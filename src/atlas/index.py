"""Dense (FAISS) + sparse (BM25) [+ CLIP] indexes over chunks, fused with reciprocal rank fusion.

Two modes are built and saved side by side:
  text_only   text chunks only; identical to the milestone 3 baseline.
  multimodal  text chunks + figure captions + table text in one text index, plus a CLIP image
              index. A figure can match through its caption (dense/BM25) and through CLIP.

Fusion (multimodal): every signal (dense, BM25, CLIP) gives a ranking over items; scores are
fused with RRF, score = sum 1/(rrf_k + rank). Then `min_per_type` reserves top-k slots for
figures/tables so the far more numerous text chunks cannot crowd them out by accident.
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from atlas import config
from atlas.chunk import chunk_all, item_chunks_all

CONFIG_KEYS = ("embedding_model", "chunk_size", "chunk_overlap")
CLIP_KEYS = ("clip_model", "clip_pretrained")
MODES = ("text_only", "multimodal")


class IndexConfigError(Exception):
    pass


class STEmbedder:
    """sentence-transformers wrapper; the model loads on first use."""

    def __init__(self, name):
        self.name, self._model = name, None

    def encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        v = self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=len(texts) > 100)
        return np.asarray(v, dtype="float32")


def tokenize(text):
    return re.findall(r"\w+", text.lower())  # matching only; stored text is untouched


def fuse(rankings, rrf_k=60):
    """rankings: {signal: {pos: (rank, score)}} -> [(pos, fused, {signal: (rank, score)})]."""
    fused = []
    for pos in set().union(*rankings.values()):
        per = {n: r[pos] for n, r in rankings.items() if pos in r}
        fused.append((pos, sum(1 / (rrf_k + rank) for rank, _ in per.values()), per))
    fused.sort(key=lambda t: (-t[1], t[0]))  # ties broken by position: deterministic
    return fused


def make_result(chunk, score, per, signals):
    r = {"chunk": chunk, "type": chunk["type"], "source_id": chunk["id"], "score": score}
    for n in signals:
        r[f"{n}_rank"], r[f"{n}_score"] = per.get(n, (None, None))
    return r


class Index:
    signals = ("dense", "bm25")

    def __init__(self, chunks, embedder, faiss_index, meta):
        self.chunks, self.embedder, self.faiss, self.meta = chunks, embedder, faiss_index, meta
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def rankings(self, query, pool=50):
        pool = min(pool, len(self.chunks))
        scores, ids = self.faiss.search(self.embedder.encode([query]), pool)
        dense = {int(i): (r + 1, float(s))
                 for r, (i, s) in enumerate(zip(ids[0], scores[0], strict=True)) if i >= 0}
        bm = self.bm25.get_scores(tokenize(query))
        top = [int(i) for i in np.argsort(-bm, kind="stable")[:pool] if bm[i] > 0]
        return {"dense": dense, "bm25": {i: (r + 1, float(bm[i])) for r, i in enumerate(top)}}

    def search(self, query, k=5, pool=50, rrf_k=60):
        """Top-k fused results: chunk, type, source_id, score, per-signal rank and score."""
        fused = fuse(self.rankings(query, pool), rrf_k)[:k]
        return [make_result(self.chunks[p], s, per, self.signals) for p, s, per in fused]


class MultiIndex:
    """Text index over text+caption+table chunks, plus CLIP text->image search over figures."""

    signals = ("dense", "bm25", "clip")

    def __init__(self, text, figures, clip_index, clip, min_per_type):
        self.text, self.figures, self.clip_index, self.clip = text, figures, clip_index, clip
        self.min_per_type = min_per_type
        have = {c["id"] for c in text.chunks}
        # captionless figures have no text chunk; they are reachable through CLIP only
        self.records = text.chunks + [f for f in figures if f["id"] not in have]
        self.pos = {r["id"]: i for i, r in enumerate(self.records)}
        self.meta = text.meta

    def search(self, query, k=5, pool=50, rrf_k=60):
        rankings = self.text.rankings(query, pool)
        scores, ids = self.clip_index.search(self.clip.encode_text([query]),
                                             min(pool, len(self.figures)))
        rankings["clip"] = {self.pos[self.figures[i]["id"]]: (r + 1, float(s))
                            for r, (i, s) in enumerate(zip(ids[0], scores[0], strict=True))
                            if i >= 0}
        sel = self._reserve(fuse(rankings, rrf_k), k)
        return [make_result(self.records[p], s, per, self.signals) for p, s, per in sel]

    def _reserve(self, fused, k):
        """Top-k by score, but guarantee min_per_type slots by swapping out lowest text items."""
        mins = self.min_per_type
        kind = lambda f: self.records[f[0]]["type"]  # noqa: E731
        sel = fused[:k]
        for t, need in mins.items():
            have = sum(1 for f in sel if kind(f) == t)
            for f in [f for f in fused[k:] if kind(f) == t][:max(0, need - have)]:
                for j in range(len(sel) - 1, -1, -1):  # lowest-scored item not needed by quota
                    if sum(1 for x in sel if kind(x) == kind(sel[j])) > mins.get(kind(sel[j]), 0):
                        del sel[j]
                        sel.append(f)
                        break
        return sorted(sel, key=lambda t: (-t[1], t[0]))


def build(chunks, embedder, cfg):
    vecs = embedder.encode([c["text"] for c in chunks])
    fi = faiss.IndexFlatIP(vecs.shape[1])
    fi.add(vecs)
    meta = {k: cfg[k] for k in CONFIG_KEYS} | {"num_chunks": len(chunks), "dim": vecs.shape[1]}
    return Index(chunks, embedder, fi, meta)


def build_multimodal(text_chunks, item_chunks, figures, embedder, clip, cfg):
    text = build(text_chunks + item_chunks, embedder, cfg)
    vecs = clip.encode_images([f["image_path"] for f in figures])
    ci = faiss.IndexFlatIP(vecs.shape[1])
    ci.add(vecs)
    text.meta |= {k: cfg[k] for k in CLIP_KEYS} | {"num_figures": len(figures)}
    return MultiIndex(text, figures, ci, clip, cfg.get("min_per_type", {}))


def _write_lines(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read_lines(path):
    return [json.loads(ln) for ln in path.read_text().split("\n") if ln]


def save(index, directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    base = index.text if isinstance(index, MultiIndex) else index
    faiss.write_index(base.faiss, str(d / "faiss.index"))
    _write_lines(d / "chunks.jsonl", base.chunks)
    (d / "meta.json").write_text(json.dumps(base.meta, indent=2))
    if isinstance(index, MultiIndex):
        faiss.write_index(index.clip_index, str(d / "clip.index"))
        _write_lines(d / "figures.jsonl", index.figures)


def _load_text(d, cfg, embedder, extra_keys=()):
    if not (d / "meta.json").exists():
        raise IndexConfigError(f"no index at {d}; run `make index` first")
    meta = json.loads((d / "meta.json").read_text())
    diff = {k: (meta.get(k), cfg[k]) for k in (*CONFIG_KEYS, *extra_keys)
            if meta.get(k) != cfg[k]}
    if diff:
        raise IndexConfigError(
            "index was built with a different config (saved vs current): "
            + ", ".join(f"{k}: {a!r} != {b!r}" for k, (a, b) in diff.items())
            + "; rebuild with `make index`")
    chunks = _read_lines(d / "chunks.jsonl")
    if len(chunks) != meta["num_chunks"]:
        raise IndexConfigError(f"index is corrupt: {len(chunks)} chunks, expected "
                               f"{meta['num_chunks']}; rebuild with `make index`")
    return Index(chunks, embedder, faiss.read_index(str(d / "faiss.index")), meta)


def load(directory, cfg, embedder):
    """Load a text_only index directory."""
    return _load_text(Path(directory), cfg, embedder)


def save_all(indexes, directory):
    """indexes: {mode: index}. Records which modes were built."""
    d = Path(directory)
    for mode, idx in indexes.items():
        save(idx, d / mode)
    (d / "modes.json").write_text(json.dumps({"modes": sorted(indexes)}))


def load_mode(directory, mode, cfg, embedder, clip=None):
    d = Path(directory)
    if mode not in MODES:
        raise IndexConfigError(f"unknown retrieval mode {mode!r}; choose from {list(MODES)}")
    built = json.loads((d / "modes.json").read_text())["modes"] if (d / "modes.json").exists() \
        else []
    if mode not in built:
        raise IndexConfigError(f"mode {mode!r} was not built (built: {built}); run `make index`")
    if mode == "text_only":
        return _load_text(d / mode, cfg, embedder)
    text = _load_text(d / mode, cfg, embedder, CLIP_KEYS)
    return MultiIndex(text, _read_lines(d / mode / "figures.jsonl"),
                      faiss.read_index(str(d / mode / "clip.index")), clip,
                      cfg.get("min_per_type", {}))


def main():
    from atlas.clip import ClipEmbedder

    cfg = config.load()
    text = chunk_all(cfg["processed_dir"], cfg["chunk_size"], cfg["chunk_overlap"])
    if not text:
        raise SystemExit(f"no text items in {cfg['processed_dir']}; run `make parse` first")
    items, figures = item_chunks_all(cfg["processed_dir"])
    emb = STEmbedder(cfg["embedding_model"])
    print(f"text_only: embedding {len(text)} chunks with {cfg['embedding_model']} ...")
    indexes = {"text_only": build(text, emb, cfg)}
    print(f"multimodal: {len(items)} figure/table chunks, {len(figures)} figures for CLIP ...")
    clip = ClipEmbedder(cfg["clip_model"], cfg["clip_pretrained"])
    indexes["multimodal"] = build_multimodal(text, items, figures, emb, clip, cfg)
    save_all(indexes, cfg["index_dir"])
    for mode, idx in indexes.items():
        print(f"saved {mode}: {idx.meta}")


if __name__ == "__main__":
    main()
