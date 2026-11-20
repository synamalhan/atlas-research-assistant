import jsonschema
import pytest

from atlas import llm

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}}


@pytest.fixture
def calls(monkeypatch):
    c = []
    real = llm.BACKENDS["fake"]

    def counting(*a):
        c.append(a)
        return real(*a)

    monkeypatch.setitem(llm.BACKENDS, "fake", counting)
    return c


def test_cache_hit_on_second_identical_call(calls):
    a = llm.generate("hi", system="s")
    b = llm.generate("hi", system="s")
    assert a == b and len(calls) == 1


def test_cache_miss_on_model_schema_prompt_change(calls):
    llm.generate("hi", schema=SCHEMA)
    llm.generate("hi", schema=SCHEMA, model="other")
    llm.generate("hi", schema={**SCHEMA, "required": ["n"]})
    llm.generate("hi2", schema=SCHEMA)
    assert len(calls) == 4


def test_cache_stores_model(tmp_path):
    llm.generate("hi", model="m1")
    f = next((tmp_path / "cache").glob("*.json"))
    assert '"model": "m1"' in f.read_text()


def test_schema_validated_json():
    out = llm.generate("hi", schema=SCHEMA)
    assert out == {"n": 0, "s": "fake"}


def test_invalid_output_raises(monkeypatch):
    monkeypatch.setitem(llm.BACKENDS, "fake", lambda *a: '{"n": "not-an-int"}')
    with pytest.raises(jsonschema.ValidationError):
        llm.generate("hi", schema=SCHEMA)


def test_retry_on_rate_limit(monkeypatch):
    n = []

    def flaky(*a):
        n.append(1)
        if len(n) < 3:
            raise llm.RateLimitError("429")
        return "ok"

    monkeypatch.setitem(llm.BACKENDS, "fake", flaky)
    assert llm.generate("hi") == "ok" and len(n) == 3


def test_missing_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.generate("hi", backend="gemini")
