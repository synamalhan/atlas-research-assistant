import json

import pytest

from atlas import ask, config, llm, qa
from atlas import index as ix
from test_index import CFG, CHUNKS, FakeEmbedder


@pytest.fixture
def index():
    return ix.build(CHUNKS, FakeEmbedder(), CFG)


def reply(claims, answerable=True):
    return json.dumps({"answerable": answerable, "claims": claims})


GOOD = reply([
    {"text": 'The protocol is called "zebrafish protocol xq7".', "cited_ids": ["p:chunk:2"],
     "quotes_directly": True},
    {"text": "Dense embeddings help.", "cited_ids": ["p:chunk:1"], "quotes_directly": False},
    {"text": "It is fast.", "cited_ids": [], "quotes_directly": False},
    {"text": "It won in 2019.", "cited_ids": ["p:chunk:2"], "quotes_directly": True},
])


@pytest.fixture
def backend(monkeypatch):
    """Generator and judge share the fake backend; they are told apart by system prompt."""
    state = {"gen": [GOOD], "judge": True, "calls": []}

    def fake(model, prompt, system, schema, temperature):
        state["calls"].append((prompt, system))
        if "judge" in system:
            return json.dumps({"supported": state["judge"], "reason": "r"})
        return state["gen"].pop(0) if len(state["gen"]) > 1 else state["gen"][0]

    monkeypatch.setitem(llm.BACKENDS, "fake", fake)
    return state


def test_claims_are_verified_and_marked(index, backend):
    res = qa.answer("what is xq7", index, k=3)
    v = [(c["verdict"], c["path"]) for c in res["claims"]]
    assert v == [("verified", "deterministic"), ("verified", "judge"),
                 ("no_citation", "none"), ("unverified", "deterministic")]
    assert "[NO CITATION] It is fast." in res["answer"]
    assert "[UNVERIFIED] It won in 2019." in res["answer"] and res["dropped"] == 0
    assert "cannot answer" in backend["calls"][0][1]


def test_drop_unverified(index, backend):
    res = qa.answer("what is xq7", index, k=3, unverified="drop")
    assert [c["verdict"] for c in res["claims"]].count("verified") == 2
    assert res["dropped"] == 2 and "It is fast" not in res["answer"]


def test_drop_all_unverified(index, backend):
    backend["judge"] = False
    backend["gen"] = [reply([{"text": "Nope.", "cited_ids": ["p:chunk:1"],
                              "quotes_directly": False}])]
    res = qa.answer("what is xq7", index, k=3, unverified="drop")
    assert res["status"] == "no_verified_claims" and res["dropped"] == 1


def test_invalid_cited_id_flagged(index, backend):
    backend["gen"] = [reply([{"text": "x", "cited_ids": ["nope:chunk:1"],
                              "quotes_directly": False}])]
    c = qa.answer("q", index)["claims"][0]
    assert c["verdict"] == "no_citation" and c["invalid_ids"] == ["nope:chunk:1"]


def test_malformed_json_retried_once_then_succeeds(index, backend):
    backend["gen"] = ["{bad json", GOOD]
    res = qa.answer("what is xq7", index, k=3)
    gens = [c for c in backend["calls"] if "judge" not in c[1]]
    assert res["status"] == "answered" and len(gens) == 2
    assert "previous response was invalid" in gens[1][0]


def test_malformed_twice_is_reported_failure(index, backend):
    backend["gen"] = ["{bad json", '{"answerable": true}']  # second is schema-invalid
    res = qa.answer("what is xq7", index, k=3)
    assert res["status"] == "generation_failed" and "twice" in res["reason"]
    assert res["claims"] == []


def test_model_says_unanswerable(index, backend):
    backend["gen"] = [reply([], answerable=False)]
    res = qa.answer("q", index)
    assert res["status"] == "abstained" and res["answer"] == qa.CANNOT


def test_abstains_below_threshold_without_calling_llm(index, backend):
    res = qa.answer("xq7", index, k=3, abstain_threshold=0.99)  # fake dense score is 0
    assert res["status"] == "abstained" and res["answer"] == qa.CANNOT
    assert backend["calls"] == [] and res["scores"] and "below threshold" in res["reason"]
    assert qa.answer("xq7", index, k=3, abstain_threshold=-1)["status"] == "answered"


@pytest.fixture
def no_abstain(monkeypatch):
    real = config.load
    monkeypatch.setattr("atlas.config.load", lambda: {**real(), "abstain_threshold": None})


def test_cli_abstains_and_prints_scores(index, backend):
    lines = []
    ask.main(["what is xq7"], index=index, out=lines.append)  # fake dense scores are 0
    text = "\n".join(lines)
    assert "Cannot answer from the corpus." in text and "[abstained]" in text
    assert "Dense scores: p:chunk:2=0.000" in text and backend["calls"] == []


def test_cli_prints_status_path_and_cited_items(index, backend, no_abstain):
    lines = []
    ask.main(["what is xq7", "--k", "3"], index=index, out=lines.append)
    text = "\n".join(lines)
    assert "[verified | deterministic]" in text and "[verified | judge]" in text
    assert "[no_citation | none]" in text and "[unverified | deterministic]" in text
    assert "verified=2 unverified=1 no_citation=1 | deterministic=2 judge=1" in text
    assert "p:chunk:2  [text] paper p, page 1" in text and "zebrafish protocol xq7" in text


def test_cli_drop_flag(index, backend, no_abstain):
    lines = []
    ask.main(["what is xq7", "--k", "3", "--unverified", "drop"], index=index, out=lines.append)
    text = "\n".join(lines)
    assert "[no_citation" not in text and "2 unverified claim(s) dropped" in text
