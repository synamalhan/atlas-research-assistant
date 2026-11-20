"""Inspect parse output: per-paper counts with flags, or a spot-check sample with --paper."""
import argparse
import json
import statistics
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent


def load(processed: Path):
    return {p.stem: [json.loads(ln) for ln in p.read_text().split("\n") if ln]
            for p in sorted(processed.glob("*.jsonl"))}


def n_pages(raw: Path, pid, items):
    try:
        return pymupdf.open(raw / f"{pid}.pdf").page_count
    except Exception:
        return max((i["page"] for i in items), default=0)


def table(papers, raw: Path):
    rows = []
    for pid, items in papers.items():
        c = {t: sum(1 for i in items if i["type"] == t) for t in ("text", "figure", "table")}
        low = sum(1 for i in items if i["quality"] == "low")
        rows.append((pid, n_pages(raw, pid, items), c, low))
    rates = [c["text"] / max(p, 1) for _, p, c, _ in rows]
    cutoff = min(5, 0.3 * statistics.median(rates)) if rates else 0
    print(f"{'paper':<16}{'pages':>6}{'text':>7}{'fig':>5}{'tab':>5}{'low':>5}  flags")
    for pid, p, c, low in rows:
        flags = []
        if c["figure"] == 0:
            flags.append("NO-FIGURES")
        if c["table"] == 0:
            flags.append("NO-TABLES")
        if c["text"] / max(p, 1) < cutoff:
            flags.append("FEW-TEXT")
        counts = f"{p:>6}{c['text']:>7}{c['figure']:>5}{c['table']:>5}{low:>5}"
        print(f"{pid:<16}{counts}  {' '.join(flags)}")


def sample(items, n):
    texts = [i for i in items if i["type"] == "text"]
    print(f"--- first {n} text blocks ---")
    for i in texts[:n]:
        print(f"[{i['id']} p{i['page']}] section={i['section']!r}\n  {i['content'][:200]!r}")
    print("--- figures ---")
    for i in (i for i in items if i["type"] == "figure"):
        print(f"[{i['id']} p{i['page']}] {i['image_path']}\n  caption: {i['caption']!r}")
    print("--- tables ---")
    for i in (i for i in items if i["type"] == "table"):
        print(f"[{i['id']} p{i['page']}] quality={i['quality']}\n  caption: {i['caption']!r}")
        print("  " + i["content"][:400].replace("\n", "\n  "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    a = ap.parse_args()
    papers = load(a.processed)
    if a.paper:
        sample(papers[a.paper], a.n)
    else:
        table(papers, a.raw)
