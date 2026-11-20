import json

import numpy as np
import pytest

from atlas import ask, llm, qa
from atlas import index as ix
from atlas.config import load as config_load

VOCAB = ["retrieval", "generation", "reranker", "precision", "accuracy", "latency", "model",
         "baseline"]
CONCEPTS = ["sunset", "graph"]
CFG = {"embedding_model": "fake", "chunk_size": 100, "chunk_overlap": 10,
       "clip_model": "fake-clip", "clip_pretrained": "none", "min_per_type": {}}


def norm(v):
    v = np.array(v, dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


class FakeEmbedder:
    def encode(self, texts):
        return norm([[t.lower().split().count(w) for w in VOCAB] for t in texts])


class FakeClip:
    """Text vectors count concept words; an image vector comes from its file name."""

    def encode_text(self, texts):
        return norm([[t.lower().split().count(w) for w in CONCEPTS] for t in texts])

    def encode_images(self, paths):
        return norm([[int(w in p) for w in CONCEPTS] for p in paths])


def rec(id_, type_, text, **kw):
    return {"id": id_, "type": type_, "paper_id": "p", "page": 1, "pages": [1],
            "section": None, "text": text, "source_ids": [id_], **kw}


TEXT = [rec(f"p:chunk:{i}", "text", t) for i, t in enumerate([
    "retrieval generation baseline model", "latency accuracy baseline", "model baseline notes"])]
FIG = rec("p:figure:0", "figure", "Figure 2: reranker precision curve", image_path="a_graph.png",
          quality="ok")
FIG_NOCAP = rec("p:figure:1", "figure", "", image_path="b_sunset.png", quality="ok")
TAB = rec("p:table:0", "table", "Table 1: latency\n|model|latency|\n|a|12ms|", quality="low")


def multi(cfg=CFG, text=TEXT):
    return ix.build_multimodal(text, [FIG, TAB], [FIG, FIG_NOCAP], FakeEmbedder(), FakeClip(), cfg)


def ids(results):
    return [r["source_id"] for r in results]


def test_figure_caption_only_evidence():
    m, t = multi(), ix.build(TEXT, FakeEmbedder(), CFG)
    assert ids(m.search("reranker precision", k=3))[0] == "p:figure:0"
    assert "p:figure:0" not in ids(t.search("reranker precision", k=3))


def test_table_only_evidence():
    m, t = multi(), ix.build(TEXT, FakeEmbedder(), CFG)
    top = m.search("latency 12ms", k=3)[0]
    assert top["source_id"] == "p:table:0" and top["chunk"]["quality"] == "low"
    assert top["type"] == "table"
    assert "p:table:0" not in ids(t.search("latency 12ms", k=3))


def test_type_tags_and_source_ids_in_both_modes():
    for idx in (multi(), ix.build(TEXT, FakeEmbedder(), CFG)):
        for r in idx.search("baseline model latency", k=5):
            assert r["type"] == r["chunk"]["type"] and r["source_id"] == r["chunk"]["id"]
    assert {r["type"] for r in multi().search("reranker latency baseline", k=6)} \
        == {"text", "figure", "table"}


def test_caption_and_clip_signals_fuse():
    r = multi().search("reranker graph", k=3)[0]
    assert r["source_id"] == "p:figure:0"
    assert r["clip_rank"] == 1 and r["bm25_rank"] == 1 and r["dense_rank"] == 1


def test_captionless_figure_found_by_clip_only_and_not_in_prompt():
    r = next(x for x in multi().search("sunset", k=5) if x["source_id"] == "p:figure:1")
    assert r["clip_rank"] == 1 and r["bm25_rank"] is None and r["dense_rank"] is None
    assert "p:figure:1" not in qa.build_prompt("sunset", [r])


def test_text_only_matches_milestone3_baseline():
    """text_only must be exactly the text-chunk-only index, whatever else the corpus holds."""
    baseline = ix.build(TEXT, FakeEmbedder(), CFG)
    modes = {"text_only": ix.build(TEXT, FakeEmbedder(), CFG), "multimodal": multi()}
    for q in ("baseline model", "latency accuracy", "reranker precision", "zzz"):
        assert modes["text_only"].search(q, k=3) == baseline.search(q, k=3)
        assert {r["type"] for r in modes["text_only"].search(q, k=3)} <= {"text"}


def test_min_per_type_reserves_slots():
    q = "baseline model"  # text chunks outrank everything else here
    plain = multi().search(q, k=2)
    assert {r["type"] for r in plain} == {"text"}
    cfg = {**CFG, "min_per_type": {"figure": 1, "table": 1}}
    r = multi(cfg).search(q, k=3)
    assert len(r) == 3 and {"figure", "table"} <= {x["type"] for x in r}
    assert r == sorted(r, key=lambda x: -x["score"])


def test_load_fails_if_mode_not_built(tmp_path):
    ix.save_all({"text_only": ix.build(TEXT, FakeEmbedder(), CFG)}, tmp_path)
    assert ix.load_mode(tmp_path, "text_only", CFG, FakeEmbedder()).meta["num_chunks"] == 3
    with pytest.raises(ix.IndexConfigError, match="not built"):
        ix.load_mode(tmp_path, "multimodal", CFG, FakeEmbedder(), FakeClip())
    with pytest.raises(ix.IndexConfigError, match="unknown"):
        ix.load_mode(tmp_path, "nope", CFG, FakeEmbedder())


def test_multimodal_roundtrip_and_clip_config_mismatch(tmp_path):
    ix.save_all({"text_only": ix.build(TEXT, FakeEmbedder(), CFG), "multimodal": multi()},
                tmp_path)
    m = ix.load_mode(tmp_path, "multimodal", CFG, FakeEmbedder(), FakeClip())
    assert "p:figure:1" in ids(m.search("sunset", k=5))
    assert ids(m.search("reranker precision", k=1)) == ["p:figure:0"]
    with pytest.raises(ix.IndexConfigError, match="clip_model"):
        ix.load_mode(tmp_path, "multimodal", {**CFG, "clip_model": "other"}, FakeEmbedder(),
                     FakeClip())


def test_cli_prints_figure_path_and_table_text(monkeypatch):
    def fake(model, prompt, system, schema, temperature):
        return json.dumps({"answerable": True, "claims": [
            {"text": "See them.", "cited_ids": ["p:figure:0", "p:table:0"],
             "quotes_directly": False}]})

    monkeypatch.setitem(llm.BACKENDS, "fake", fake)
    monkeypatch.setattr("atlas.config.load", lambda: {**config_load(), "abstain_threshold": None})
    lines = []
    ask.main(["reranker latency", "--k", "5"], index=multi(), out=lines.append)
    text = "\n".join(lines)
    assert "[figure] paper p, page 1" in text and "image: a_graph.png" in text
    assert "caption: Figure 2: reranker precision curve" in text
    assert "[table] paper p, page 1" in text and "|a|12ms|" in text and "low-quality" in text
