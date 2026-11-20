"""QA with structured claims, abstention and claim verification.

NOTE: the generator sees TEXT ONLY: text chunks, figure captions and table text. It never
receives figure images (passing images to a vision model is out of scope and would not work
with every backend). A figure without a caption has no text, so it is not put in the prompt.
"""
import json

import jsonschema

from atlas import llm, verify

SYSTEM = (
    "You answer questions using ONLY the provided items. Each item starts with its ID in "
    "square brackets and its type: a text passage, a figure caption, or a table (with its "
    "caption). You cannot see the figures themselves, only their captions. Answer as a list "
    "of short claims. For each claim give `cited_ids`: the IDs of the items it relies on. "
    "Set `quotes_directly` to true if the claim copies source text in double quotation marks "
    "or states a specific number taken from the source; otherwise false (a paraphrase). If "
    "the items do not contain the answer, set `answerable` to false, return no claims and "
    "say nothing else: you cannot answer from the provided chunks. Do not use outside "
    "knowledge."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"},
                           "cited_ids": {"type": "array", "items": {"type": "string"}},
                           "quotes_directly": {"type": "boolean"}},
            "required": ["text", "cited_ids", "quotes_directly"]}},
    },
    "required": ["answerable", "claims"],
}
CANNOT = "Cannot answer from the corpus."
TAGS = {"unverified": "[UNVERIFIED] ", "no_citation": "[NO CITATION] "}


def build_prompt(question, results):
    ctx = "\n\n".join(f"[{r['chunk']['id']}] ({r['type']})\n{r['chunk']['text']}"
                      for r in results if r["chunk"]["text"].strip())
    return f"Items:\n\n{ctx}\n\nQuestion: {question}"


def score_table(results):
    return [{"id": r["source_id"], "fused": r["score"], "dense": r["dense_score"],
             "bm25": r["bm25_score"], "clip": r.get("clip_score")} for r in results]


def generate_claims(prompt, backend=None, model=None):
    """Structured generation; one retry (with the error in the prompt) on malformed output."""
    err = None
    for _ in range(2):
        p = prompt if err is None else (
            f"{prompt}\n\nYour previous response was invalid ({err}). "
            "Return valid JSON that matches the schema.")
        try:
            return llm.generate(p, system=SYSTEM, schema=SCHEMA, backend=backend,
                                model=model), None
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
    return None, err


def answer(question, index, k=5, backend=None, model=None, pool=50, rrf_k=60,
           judge_backend=None, judge_model=None, unverified="mark", abstain_threshold=None):
    """Returns {status, answer, claims, dropped, reason, scores, retrieved}.

    status: answered | abstained | no_verified_claims | generation_failed
    """
    results = index.search(question, k=k, pool=pool, rrf_k=rrf_k)
    scores = score_table(results)
    res = {"status": "answered", "answer": "", "claims": [], "dropped": 0, "reason": None,
           "scores": scores, "retrieved": results}

    def stop(status, reason, text):
        return res | {"status": status, "reason": reason, "answer": text}

    # the best dense cosine similarity decides abstention (fused RRF scores are rank-based)
    dense = [s["dense"] for s in scores if s["dense"] is not None]
    if abstain_threshold is not None and dense and max(dense) < abstain_threshold:
        return stop("abstained",
                    f"best dense score {max(dense):.3f} is below threshold {abstain_threshold}",
                    CANNOT)
    out, err = generate_claims(build_prompt(question, results), backend, model)
    if out is None:
        return stop("generation_failed", f"malformed model output twice: {err}", "")
    if not out["answerable"]:
        return stop("abstained", "model says the retrieved items do not contain the answer",
                    CANNOT)
    shown = {r["chunk"]["id"]: r["chunk"] for r in results if r["chunk"]["text"].strip()}
    claims = [verify.verify_claim(c, shown, judge_backend, judge_model) for c in out["claims"]]
    kept = [c for c in claims if c["verdict"] == "verified" or unverified == "mark"]
    res |= {"claims": claims, "dropped": len(claims) - len(kept)}
    if not any(c["verdict"] == "verified" for c in claims) and unverified == "drop":
        return res | {"status": "no_verified_claims",
                      "reason": "no claim could be verified; all were dropped",
                      "answer": "No claim could be verified against the corpus."}
    res["answer"] = " ".join(TAGS.get(c["verdict"], "") + c["text"] for c in kept)
    return res
