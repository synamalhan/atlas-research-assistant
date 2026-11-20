"""Small LLM interface: fake / ollama / gemini backends, disk cache, retry on rate limits."""
import hashlib
import json
import os
import time
from pathlib import Path

import jsonschema
import requests

DEFAULT_MODELS = {"fake": "fake-1", "ollama": "llama3.1", "gemini": "gemini-2.5-flash"}
RETRIES = 5
BACKOFF = 1.0  # seconds; doubles each retry


class RateLimitError(Exception):
    pass


def _post(url, headers=None, **body):
    r = requests.post(url, headers=headers, json=body, timeout=300)
    if r.status_code == 429 or r.status_code >= 500:
        raise RateLimitError(f"HTTP {r.status_code} from {url}")
    r.raise_for_status()
    return r.json()


def _sample(schema):
    """Smallest valid value for a schema (fake backend)."""
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {k: _sample(v) for k, v in schema.get("properties", {}).items()}
    return {"string": "fake", "integer": 0, "number": 0.0, "boolean": False, "array": []}.get(t)


def _fake(model, prompt, system, schema, temperature):
    if schema:
        return json.dumps(_sample(schema))
    return "fake:" + hashlib.sha256(f"{system}|{prompt}".encode()).hexdigest()[:12]


def _ollama(model, prompt, system, schema, temperature):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    opts = {"temperature": temperature}
    body = {"model": model, "prompt": prompt, "stream": False, "options": opts}
    if system:
        body["system"] = system
    if schema:
        body["format"] = schema
    return _post(f"{host}/api/generate", **body)["response"]


def _gemini(model, prompt, system, schema, temperature):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; export it to use the gemini backend")
    cfg = {"temperature": temperature}
    if schema:
        cfg |= {"responseMimeType": "application/json", "responseSchema": schema}
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    out = _post(url, headers={"x-goog-api-key": key}, **body)
    return out["candidates"][0]["content"]["parts"][0]["text"]


BACKENDS = {"fake": _fake, "ollama": _ollama, "gemini": _gemini}


def _call_with_retry(fn, *args):
    for attempt in range(RETRIES):
        try:
            return fn(*args)
        except RateLimitError as e:
            if attempt == RETRIES - 1:
                raise RuntimeError(f"rate limited after {RETRIES} attempts: {e}") from e
            time.sleep(BACKOFF * 2**attempt)


def generate(prompt, *, system=None, schema=None, temperature=0.0, backend=None, model=None):
    """Return text, or schema-validated parsed JSON when `schema` is given.

    backend/model default to env ATLAS_LLM_BACKEND (fake) / ATLAS_LLM_MODEL.
    """
    backend = backend or os.environ.get("ATLAS_LLM_BACKEND", "fake")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    model = model or os.environ.get("ATLAS_LLM_MODEL") or DEFAULT_MODELS[backend]
    key = json.dumps(
        [backend, model, system, prompt, schema, temperature], sort_keys=True, default=str
    )
    path = Path(os.environ.get("ATLAS_CACHE_DIR", ".cache/llm")) / (
        hashlib.sha256(key.encode()).hexdigest() + ".json"
    )
    cached = path.exists()
    if cached:
        text = json.loads(path.read_text())["response"]
    else:
        text = _call_with_retry(BACKENDS[backend], model, prompt, system, schema, temperature)
    result = text
    if schema:
        result = json.loads(text)
        jsonschema.validate(result, schema)  # raises on invalid output; never cached
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"backend": backend, "model": model, "response": text}))
    return result
