import numpy as np
import pytest

from atlas import index as ix

VOCAB = ["retrieval", "generation", "dense", "embeddings", "semantics", "accuracy"]
CFG = {"embedding_model": "fake", "chunk_size": 100, "chunk_overlap": 10}


class FakeEmbedder:
    """Counts known vocabulary words only; rare/unknown terms are invisible to it."""

    def encode(self, texts):
        v = np.array([[t.lower().split().count(w) for w in VOCAB] for t in texts], dtype="float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n == 0, 1, n)


def chunk(i, text):
    return {"id": f"p:chunk:{i}", "type": "text", "paper_id": "p", "page": 1, "pages": [1],
            "section": "S", "text": text, "source_ids": [f"p:text:{i}"]}


CHUNKS = [
    chunk(0, "retrieval generation improves accuracy"),
    chunk(1, "dense embeddings capture semantics"),
    chunk(2, "the zebrafish protocol xq7 yields results"),
]


@pytest.fixture
def index():
    return ix.build(CHUNKS, FakeEmbedder(), CFG)


def ids(results):
    return [r["chunk"]["id"] for r in results]


def test_dense_finds_semantic_match(index):
    r = index.search("retrieval generation", k=3)
    assert ids(r)[0] == "p:chunk:0" and r[0]["dense_rank"] == 1 and r[0]["bm25_rank"] == 1


def test_bm25_finds_exact_term_dense_misses(index):
    r = index.search("xq7", k=3)
    top = r[0]
    assert top["chunk"]["id"] == "p:chunk:2"
    assert top["bm25_rank"] == 1
    assert top["dense_score"] == 0  # fake embedder cannot see "xq7"
    assert [x["bm25_rank"] for x in r[1:]] == [None, None]  # no BM25 match elsewhere


def test_fusion_fields_and_scores(index):
    r = index.search("dense embeddings", k=2, rrf_k=60)
    assert set(r[0]) == {"chunk", "type", "source_id", "score", "dense_rank", "dense_score",
                         "bm25_rank", "bm25_score"}
    assert r[0]["score"] == pytest.approx(1 / 61 + 1 / 61)
    assert r[0]["score"] > r[1]["score"]


def test_save_load_roundtrip_and_config_mismatch(index, tmp_path):
    ix.save(index, tmp_path)
    loaded = ix.load(tmp_path, CFG, FakeEmbedder())
    assert ids(loaded.search("xq7", k=1)) == ["p:chunk:2"]
    assert loaded.meta["num_chunks"] == 3
    with pytest.raises(ix.IndexConfigError, match="embedding_model"):
        ix.load(tmp_path, {**CFG, "embedding_model": "other"}, FakeEmbedder())
    with pytest.raises(ix.IndexConfigError, match="chunk_size"):
        ix.load(tmp_path, {**CFG, "chunk_size": 999}, FakeEmbedder())


def test_load_missing_index(tmp_path):
    with pytest.raises(ix.IndexConfigError, match="make index"):
        ix.load(tmp_path / "nope", CFG, FakeEmbedder())
