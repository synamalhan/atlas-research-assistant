"""CLI: python -m atlas.ask "question" [--k N] [--mode text_only|multimodal] [...]"""
import argparse
from collections import Counter

from atlas import config, qa
from atlas import index as index_mod


def load_index(cfg, mode):
    emb = index_mod.STEmbedder(cfg["embedding_model"])
    clip = None
    if mode == "multimodal":
        from atlas.clip import ClipEmbedder

        clip = ClipEmbedder(cfg["clip_model"], cfg["clip_pretrained"])
    return index_mod.load_mode(cfg["index_dir"], mode, cfg, emb, clip)


def show(c, out):
    """Print one cited item: type, paper, page, then its figure path / table text / snippet."""
    out(f"  {c['id']}  [{c['type']}] paper {c['paper_id']}, page {c['page']}")
    if c["type"] == "figure":
        out(f"    image: {c['image_path']}\n    caption: {c['text']}")
    elif c["type"] == "table":
        warn = "  (low-quality extraction)" if c.get("quality") == "low" else ""
        out(f"    table{warn}:\n" + "\n".join("      " + ln for ln in c["text"].split("\n")))
    else:
        out(f"    {' '.join(c['text'].split())[:150]}...")


def main(argv=None, index=None, out=print):
    ap = argparse.ArgumentParser(prog="atlas.ask")
    ap.add_argument("question")
    ap.add_argument("--k", type=int)
    ap.add_argument("--mode", choices=index_mod.MODES)
    ap.add_argument("--backend")
    ap.add_argument("--model")
    ap.add_argument("--judge-backend")
    ap.add_argument("--judge-model")
    ap.add_argument("--unverified", choices=["drop", "mark"])
    a = ap.parse_args(argv)
    cfg = config.load()
    if index is None:
        index = load_index(cfg, a.mode or cfg["retrieval_mode"])
    unverified = a.unverified or cfg["unverified_claims"]
    res = qa.answer(
        a.question, index, k=a.k or cfg["k"], backend=a.backend, model=a.model,
        pool=cfg["pool"], rrf_k=cfg["rrf_k"],
        judge_backend=a.judge_backend or cfg["judge_backend"] or a.backend,
        judge_model=a.judge_model or cfg["judge_model"] or a.model,
        unverified=unverified, abstain_threshold=cfg["abstain_threshold"])
    if res["status"] != "answered":
        out(f"{res['answer'] or 'Generation failed.'}  [{res['status']}]\n"
            f"Reason: {res['reason']}")
        dense = [f"{s['id']}={s['dense']:.3f}" for s in res["scores"] if s["dense"] is not None]
        out("Dense scores: " + ", ".join(dense))
        return
    chunks = {r["chunk"]["id"]: r["chunk"] for r in res["retrieved"]}
    shown = [c for c in res["claims"] if c["verdict"] == "verified" or unverified == "mark"]
    for c in shown:
        out(f"[{c['verdict']} | {c['path']}] {c['text']}\n    cited: {c['cited_ids']}\n"
            f"    why: {c['reason']}")
    if res["dropped"]:
        out(f"({res['dropped']} unverified claim(s) dropped)")
    n = Counter(c["verdict"] for c in res["claims"])
    p = Counter(c["path"] for c in res["claims"])
    out(f"\nverified={n['verified']} unverified={n['unverified']} "
        f"no_citation={n['no_citation']} | deterministic={p['deterministic']} judge={p['judge']}")
    cited = list(dict.fromkeys(i for c in shown for i in c["cited_ids"] if i in chunks))
    out("\nCited items:" if cited else "\nCited items: none")
    for cid in cited:
        show(chunks[cid], out)


if __name__ == "__main__":
    main()
