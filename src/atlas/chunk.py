"""Merge adjacent text blocks (same paper, same section) into overlapping chunks.

A chunk record is type-agnostic so figures/tables can be added later without changing it:
  id, type, paper_id, page, pages, section, text, source_ids
Text is joined verbatim with a blank line between blocks; nothing is normalised.
"""
import json
from pathlib import Path

SEP = "\n\n"


def _len(blocks):
    return sum(len(b["content"]) for b in blocks) + len(SEP) * (len(blocks) - 1)


def chunk_paper(items, size, overlap):
    """Chunk one paper's items (in reading order). Only `text` items are used for now."""
    texts = [i for i in items if i["type"] == "text"]
    chunks, cur = [], []

    def emit():
        pages = sorted({b["page"] for b in cur})
        n = len(chunks)
        chunks.append({
            "id": f"{cur[0]['paper_id']}:chunk:{n}", "type": "text",
            "paper_id": cur[0]["paper_id"], "page": cur[0]["page"], "pages": pages,
            "section": cur[0]["section"], "text": SEP.join(b["content"] for b in cur),
            "source_ids": [b["id"] for b in cur],
        })

    for b in texts:
        if cur and b["section"] != cur[0]["section"]:
            emit()
            cur = []
        elif cur and _len(cur + [b]) > size:
            emit()
            keep = []
            for x in reversed(cur):  # trailing blocks that fit in the overlap budget
                if _len([x] + keep) > overlap:
                    break
                keep.insert(0, x)
            cur = keep if keep and _len(keep + [b]) <= size else []
        cur.append(b)
    if cur:
        emit()
    return chunks


def chunk_all(processed_dir, size, overlap):
    chunks = []
    for p in sorted(Path(processed_dir).glob("*.jsonl")):
        chunks += chunk_paper(_read(p), size, overlap)
    return chunks


def item_chunks(items):
    """One paper's figure/table items -> (chunks to index as text, all figure records).

    Same record shape as text chunks (+ `quality`, and `image_path` for figures). A figure
    without a caption has empty text: it gets no text chunk but is still in `figures`
    so it can be found through CLIP.
    """
    chunks, figures = [], []
    for i in items:
        if i["type"] not in ("figure", "table"):
            continue
        rec = {"id": i["id"], "type": i["type"], "paper_id": i["paper_id"], "page": i["page"],
               "pages": [i["page"]], "section": None, "source_ids": [i["id"]],
               "quality": i["quality"]}
        if i["type"] == "figure":
            rec |= {"text": i["caption"] or "", "image_path": i["image_path"]}
            figures.append(rec)
        else:
            rec["text"] = "\n".join(x for x in (i["caption"], i["content"]) if x)
        if rec["text"].strip():
            chunks.append(rec)
    return chunks, figures


def _read(p):
    # split on "\n" only: str.splitlines() also splits on unicode separators
    return [json.loads(ln) for ln in p.read_text().split("\n") if ln]


def item_chunks_all(processed_dir):
    chunks, figures = [], []
    for p in sorted(Path(processed_dir).glob("*.jsonl")):
        c, f = item_chunks(_read(p))
        chunks += c
        figures += f
    return chunks, figures
