"""PDF -> stable text / figure / table items (PyMuPDF).

Text is kept exactly as extracted (no lowercasing, no number stripping): the verifier later
checks quotes and numbers against it. Heading, caption and table detection are heuristics.
"""
import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

CAPTION = {
    "figure": re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.I),
    "table": re.compile(r"^\s*table\s*\d+", re.I),
}
CAPTION_PUNCT = re.compile(r"^\s*(figure|fig\.?|table)\s*\d+\s*[:.|\-–—]", re.I)
NUM_HEAD = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
ALPHA_HEAD = re.compile(r"^([A-Z]|[IVX]+)\.\s+[A-Z]")
NAMED_HEAD = re.compile(r"^(abstract|references|acknowledg(e)?ments?|appendix)\b", re.I)
MARGIN = 0.07  # top/bottom fraction of the page treated as header/footer zone
MAX_CAPTION_GAP = 150  # points between a figure/table and its caption
FIG_DPI = 150


# ---------- text blocks ----------

def _text_blocks(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0 or not b["lines"]:
            continue
        if any(abs(ln["dir"][0] - 1) > 0.1 for ln in b["lines"]):
            continue  # rotated text, e.g. the arXiv side stamp
        lines, sizes, bold, total = [], Counter(), 0, 0
        for line in b["lines"]:
            lines.append("".join(s["text"] for s in line["spans"]))
            for s in line["spans"]:
                n = len(s["text"])
                sizes[round(s["size"], 1)] += n
                total += n
                if s["flags"] & 16 or "bold" in s["font"].lower():
                    bold += n
        text = "\n".join(lines)
        if text.strip():
            out.append({
                "text": text,
                "bbox": [round(v, 1) for v in b["bbox"]],
                "size": sizes.most_common(1)[0][0],
                "bold": bool(total) and bold >= 0.8 * total,
            })
    return out


def _margin_key(b, height):
    y0, y1 = b["bbox"][1], b["bbox"][3]
    if y1 < MARGIN * height or y0 > (1 - MARGIN) * height:
        return re.sub(r"\d+", "#", b["text"].strip())
    return None


def _drop_headers(pages_blocks, heights):
    """Drop page numbers and margin text repeated on >= half of the pages."""
    seen = Counter()
    for blocks, h in zip(pages_blocks, heights, strict=True):
        for k in {_margin_key(b, h) for b in blocks} - {None}:
            seen[k] += 1
    need = max(2, math.ceil(len(pages_blocks) / 2))
    kept = []
    for blocks, h in zip(pages_blocks, heights, strict=True):
        keep = []
        for b in blocks:
            k = _margin_key(b, h)
            if k is not None and (re.fullmatch(r"\s*\d+\s*", b["text"]) or seen[k] >= need):
                continue
            keep.append(b)
        kept.append(keep)
    return kept


def _reading_order(blocks, width):
    """Full-width blocks split the page into bands; inside a band: left column, then right."""
    mid = width / 2

    def full(b):
        return b["bbox"][0] < mid - 5 and b["bbox"][2] > mid + 5

    narrow = [b for b in blocks if not full(b)]
    left = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
    if len(left) < 2 or len(narrow) - len(left) < 2:  # not a two-column page
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    out, band = [], []

    def flush():
        for side in (True, False):
            col = [b for b in band if ((b["bbox"][0] + b["bbox"][2]) / 2 < mid) == side]
            out.extend(sorted(col, key=lambda b: b["bbox"][1]))
        band.clear()

    for b in sorted(blocks, key=lambda b: b["bbox"][1]):
        if full(b):
            flush()
            out.append(b)
        else:
            band.append(b)
    flush()
    return out


def _is_heading(b, body_size):
    t = b["text"].strip()
    if len(t) > 120 or t.count("\n") > 1 or not any(c.isalpha() for c in t):
        return False
    if NUM_HEAD.match(t) or NAMED_HEAD.match(t):
        return True
    emphasised = b["bold"] or b["size"] > body_size * 1.1
    if ALPHA_HEAD.match(t):
        return emphasised
    return emphasised and len(t.split()) <= 12 and t[-1] not in ".,;:"


def _is_caption(b, kind, body_size):
    if not CAPTION[kind].match(b["text"]) or len(b["text"].split()) < 3:
        return False  # one-word "Table1" is a diagram label, not a caption
    # "Table 2 shows ..." starts body paragraphs too: require punctuation or smaller font
    return bool(CAPTION_PUNCT.match(b["text"])) or b["size"] < body_size * 0.97


# ---------- figures and tables ----------

def _merge(rects, gap=10):
    rects = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if (rects[i] + (-gap, -gap, gap, gap)).intersects(rects[j]):
                    rects[i] |= rects[j]
                    del rects[j]
                    changed = True
                    break
            if changed:
                break
    return rects


def _find_figures(page, table_rects):
    page_area = page.rect.get_area()
    cands = []
    for im in page.get_image_info():
        r = pymupdf.Rect(im["bbox"])
        if r.width >= 50 and r.height >= 50:
            cands.append(r)
    try:
        for r in page.cluster_drawings():
            if r.width >= 100 and r.height >= 60:
                cands.append(pymupdf.Rect(r))
    except Exception:  # drawings are best effort
        pass
    out = []
    for r in _merge(cands):
        if r.get_area() > 0.9 * page_area:
            continue  # page background
        if any((r & t).get_area() > 0.5 * min(r.get_area(), t.get_area()) for t in table_rects):
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r.y0, r.x0))


def _find_tables(page):
    out = []
    for t in page.find_tables(use_layout=False).tables:
        rect = pymupdf.Rect(t.bbox)
        try:
            rows = t.extract()
            cells = [c for r in rows for c in r]
            ok = (
                len(rows) >= 2 and len(rows[0]) >= 2
                and sum(1 for c in cells if c and c.strip()) >= 0.5 * len(cells)
            )
            content = t.to_markdown() if ok else page.get_text("text", clip=rect)
        except Exception:
            ok, content = False, page.get_text("text", clip=rect)
        out.append((rect, content, "ok" if ok else "low"))
    return sorted(out, key=lambda x: (x[0].y0, x[0].x0))


def _nearest_caption(rect, caps, used):
    """Nearest unused caption block below (preferred for figures) or above the rect."""
    best = None
    for b in caps:
        if id(b) in used:
            continue
        x0, y0, x1, y1 = b["bbox"]
        if min(x1, rect.x1) - max(x0, rect.x0) <= 0:
            continue
        gap = min(abs(y0 - rect.y1), abs(rect.y0 - y1))
        if gap <= MAX_CAPTION_GAP and (best is None or gap < best[0]):
            best = (gap, b)
    return best[1] if best else None


def _orphan_table(cap, ordered, excluded):
    """Caption without a detected table: keep the raw text under it, marked low quality."""
    cx0, cy0, cx1, cy1 = cap["bbox"]
    taken = []
    for b in ordered:
        x0, y0, x1, y1 = b["bbox"]
        cx = (x0 + x1) / 2
        if b is cap or id(b) in excluded or y0 < cy1 - 1 or not cx0 <= cx <= cx1:
            continue
        if len(b["text"]) > 250 and (x1 - x0) >= 0.8 * (cx1 - cx0):
            break  # body paragraph: the table is over
        taken.append(b)
        if len(taken) >= 40:
            break
    rect = pymupdf.Rect(cap["bbox"])
    for b in taken:
        rect |= pymupdf.Rect(b["bbox"])
    return rect, "\n".join(b["text"] for b in taken), taken


# ---------- parsing ----------

def parse_pdf(path, figures_dir):
    """Return (items, page_errors, n_pages). Raises if the PDF cannot be opened or is empty."""
    path, figures_dir = Path(path), Path(figures_dir)
    paper_id = path.stem
    version = (re.search(r"v\d+$", paper_id) or [None])[0]
    doc = pymupdf.open(path)
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")
    errors, raw, heights = [], [], []
    for page in doc:
        heights.append(page.rect.height)
        try:
            raw.append(_text_blocks(page))
        except Exception as e:
            errors.append({"page": page.number + 1, "error": f"text: {e!r}"})
            raw.append([])
    pages_blocks = _drop_headers(raw, heights)
    weights = Counter()
    for blocks in pages_blocks:
        for b in blocks:
            weights[b["size"]] += len(b["text"])
    body = weights.most_common(1)[0][0] if weights else 10.0

    items, count, section = [], Counter(), None

    def add(kind, page_no, bbox, content, **kw):
        item = {
            "id": f"{paper_id}:{kind}:{count[kind]}", "paper_id": paper_id, "type": kind,
            "page": page_no, "bbox": [round(v, 1) for v in bbox], "section": None,
            "content": content, "image_path": None, "caption": None, "quality": "ok",
            "source_version": version,
        }
        count[kind] += 1
        item.update(kw)
        items.append(item)

    for page, blocks in zip(doc, pages_blocks, strict=True):
        n = page.number + 1
        try:
            ordered = _reading_order(blocks, page.rect.width)
            caps = {k: [b for b in ordered if _is_caption(b, k, body)] for k in CAPTION}
            used, excluded = set(), set()

            def inside(b, rect):
                return rect.contains(pymupdf.Point(
                    (b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2))

            table_items = []
            for rect, content, quality in _find_tables(page):
                cap = _nearest_caption(rect, caps["table"], used)
                if cap is None and quality == "low":
                    continue  # uncaptioned grid that extracts badly: a drawing, handled as figure
                if cap:
                    used.add(id(cap))
                table_items.append((rect, content, quality, cap["text"] if cap else None))
            figs = _find_figures(page, [t[0] for t in table_items])
            for cap in caps["table"]:  # captions with no detected table
                if id(cap) not in used:
                    used.add(id(cap))
                    rect, content, taken = _orphan_table(cap, ordered, excluded)
                    excluded |= {id(b) for b in taken}
                    table_items.append((rect, content, "low", cap["text"]))
            fig_items = []
            for rect in figs:
                cap = _nearest_caption(rect, caps["figure"], used)
                if cap:
                    used.add(id(cap))
                fig_items.append((rect, cap["text"] if cap else None))

            for rect in [t[0] for t in table_items] + [f[0] for f in fig_items]:
                excluded |= {id(b) for b in ordered if id(b) not in used and inside(b, rect)}

            for b in ordered:
                if id(b) in excluded:
                    continue
                if _is_heading(b, body):
                    section = b["text"].strip()
                add("text", n, b["bbox"], b["text"], section=section)
            for rect, cap in fig_items:
                png = figures_dir / f"{paper_id}_{count['figure']}.png"
                figures_dir.mkdir(parents=True, exist_ok=True)
                page.get_pixmap(clip=rect, dpi=FIG_DPI).save(png)
                add("figure", n, rect, cap or "", image_path=str(png), caption=cap)
            for rect, content, quality, cap in table_items:
                add("table", n, rect, content, caption=cap, quality=quality)
        except Exception as e:
            errors.append({"page": n, "error": f"{e!r}"})
    return items, errors, doc.page_count


def _parse_one(args):
    pdf, out_dir = args
    items, errors, n_pages, error = [], [], 0, None
    try:
        items, errors, n_pages = parse_pdf(pdf, Path(out_dir) / "figures")
        if not any(i["type"] == "text" for i in items):
            error = "no text extracted (scanned PDF?)"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    if errors:
        error = error or f"{len(errors)} page(s) failed"
    return Path(pdf).stem, items, errors, n_pages, error


def parse_all(raw_dir, out_dir, workers=None):
    """Parse every PDF; never raises per paper. Writes JSONL per paper + parse_report.json."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(out_dir)) for p in sorted(raw_dir.glob("*.pdf"))]
    if workers == 1:
        results = map(_parse_one, jobs)
    else:  # find_tables is slow (~0.5 s/page), so parse papers in parallel
        results = ProcessPoolExecutor(workers).map(_parse_one, jobs)
    failed, ok = [], 0
    for stem, items, errors, n_pages, error in results:
        if items:
            with open(out_dir / f"{stem}.jsonl", "w") as f:
                f.writelines(json.dumps(i) + "\n" for i in items)
        if error:
            failed.append({
                "paper_id": stem, "error": error, "page_errors": errors,
                "salvaged": {
                    **{k: sum(1 for i in items if i["type"] == k)
                       for k in ("text", "figure", "table")},
                    "pages_total": n_pages, "pages_failed": len(errors),
                },
            })
        else:
            ok += 1
    report = {"ok": ok, "failed": failed}
    (out_dir / "parse_report.json").write_text(json.dumps(report, indent=2))
    return report
