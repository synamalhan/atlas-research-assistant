import json

import pytest

from atlas import llm, verify
from atlas.verify import check_deterministic as chk


def ok(claim, *sources):
    res = chk(claim, list(sources))
    assert res is not None
    return res[0]


# ---- quotes
def test_exact_quote():
    assert ok('It says "Retrieval helps a lot".', "Retrieval helps a lot, said the authors.")


def test_quote_case_insensitive():
    assert ok('It says "retrieval helps".', "RETRIEVAL HELPS here")


def test_quote_across_hyphenated_line_break():
    assert ok('They "employ dense retrieval".', "They em-\nploy dense retrieval.")
    assert ok('They "employ dense retrieval".', "They em- ploy dense retrieval.")
    assert ok('A "state-of-the-art model".', "A state-of-the-\nart model.")  # real hyphen kept


def test_quote_different_whitespace_and_quote_marks():
    assert ok('He wrote “the  quick\nbrown fox”.', 'the quick brown fox')
    assert ok('He wrote "it’s a long—very long road".', "it's a long-very long road")


def test_absent_quote():
    res = chk('It says "gradient descent wins".', ["Retrieval helps a lot."])
    assert res[0] is False and "gradient descent wins" in res[1]


# ---- numbers
@pytest.mark.parametrize("claim, source", [
    ("1000 queries", "we ran 1,000 queries"), ("1,000 queries", "we ran 1000 queries"),
    ("accuracy of 50%", "accuracy 0.5"), ("accuracy of 0.5", "accuracy 50%"),
    ("score of 12", "score 12.0"), ("score of 12.0", "score 12"),
    ("42.5% accuracy", "got 42.5% accuracy"),
])
def test_number_formats_match(claim, source):
    assert ok(claim, source)


def test_number_in_markdown_table():
    table = "Table 1: Scores\n|Model|Acc|\n|---|---|\n|A|71.3|\n|B|64.8<br>66.1|"
    assert ok("Model A got 71.3", table)
    assert ok("Model B got 66.1", table)
    assert not ok("Model A got 71.4", table)


def test_number_not_matched_inside_larger_number():
    assert not ok("5 papers were used", "25 papers were used")
    assert not ok("2 points", "2.5 points")
    assert not ok("12 runs", "112 runs")
    assert ok("25 papers", "25 papers")


def test_numbers_and_quotes_must_all_be_found():
    assert not ok('"alpha beta" gives 3 gains', "alpha beta appears; 4 gains")


def test_nothing_checkable_returns_none():
    assert chk("Retrieval helps.", ["Retrieval helps."]) is None


# ---- verdict logic
SHOWN = {"a": {"id": "a", "text": "Accuracy is 71.3% on Foo."},
         "b": {"id": "b", "text": "Dense retrieval embeds passages."}}


def claim(text, ids, direct):
    return {"text": text, "cited_ids": ids, "quotes_directly": direct}


@pytest.fixture
def judge_says(monkeypatch):
    seen = []

    def set_(supported, reason="because"):
        def fake(model, prompt, system, schema, temperature):
            seen.append((prompt, system))
            return json.dumps({"supported": supported, "reason": reason})

        monkeypatch.setitem(llm.BACKENDS, "fake", fake)
        return seen

    return set_


def test_no_citation():
    v = verify.verify_claim(claim("x", [], False), SHOWN)
    assert (v["verdict"], v["path"]) == ("no_citation", "none")


def test_invalid_cited_id_is_flagged():
    v = verify.verify_claim(claim("x", ["zzz"], False), SHOWN)
    assert v["verdict"] == "no_citation" and v["invalid_ids"] == ["zzz"]
    assert "zzz" in v["reason"]


def test_deterministic_path():
    v = verify.verify_claim(claim("Accuracy is 71.3%", ["a"], True), SHOWN)
    assert (v["verdict"], v["path"]) == ("verified", "deterministic")
    v = verify.verify_claim(claim("Accuracy is 99%", ["a"], True), SHOWN)
    assert (v["verdict"], v["path"]) == ("unverified", "deterministic")


def test_judge_supported_and_not_supported(judge_says):
    seen = judge_says(True, "matches")
    v = verify.verify_claim(claim("It embeds passages.", ["b", "zzz"], False), SHOWN)
    assert (v["verdict"], v["path"]) == ("verified", "judge") and "matches" in v["reason"]
    assert v["invalid_ids"] == ["zzz"]
    prompt, _ = seen[0]  # the judge sees only the claim and the cited items
    assert "Dense retrieval embeds passages." in prompt and "71.3" not in prompt
    judge_says(False, "not stated")
    v = verify.verify_claim(claim("It embeds images.", ["b"], False), SHOWN)
    assert (v["verdict"], v["path"]) == ("unverified", "judge")


def test_direct_flag_without_checkable_content_goes_to_judge(judge_says):
    judge_says(True)
    v = verify.verify_claim(claim("It embeds passages.", ["b"], True), SHOWN)
    assert v["path"] == "judge"


def test_judge_failure_is_unverified(monkeypatch):
    monkeypatch.setitem(llm.BACKENDS, "fake", lambda *a: "not json")
    v = verify.verify_claim(claim("It embeds passages.", ["b"], False), SHOWN)
    assert v["verdict"] == "unverified" and "judge failed" in v["reason"]
