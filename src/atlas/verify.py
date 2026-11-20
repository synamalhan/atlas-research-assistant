"""Claim verification: deterministic quote/number checks first, an LLM judge for paraphrases.

Normalisation happens at comparison time only; stored items are never modified.
Verdicts: verified | unverified | no_citation, plus the path that decided it
(deterministic | judge | none) and a reason.
"""
import json
import re
import unicodedata
from decimal import Decimal

import jsonschema

from atlas import llm

CHAR_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "′": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
})
BREAK = re.compile(r"(?<=\w)-\s+(?=\w)")  # "em-\nploy" or "em- ploy"
NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s*%)?(?!\w)")
QUOTE = re.compile(r'"([^"]{4,})"')

JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. You are given a claim and the cited source items. "
    "Decide whether the claim is supported by the items alone. Use no outside knowledge. "
    "Answer supported=false if the items do not state or clearly imply the claim."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}


def _clean(s):
    return unicodedata.normalize("NFKC", s).translate(CHAR_MAP)


def _squash(s):
    return " ".join(s.split()).lower()


def text_variants(s):
    """Normalised forms of a text: hyphen breaks kept as-is, joined ('employ'), or as '-'."""
    s = _clean(s)
    return {_squash(s), _squash(BREAK.sub("", s)), _squash(BREAK.sub("-", s))}


def numbers(s):
    """[(raw, value, is_percent)] for whole-token numbers: 1,000 -> 1000, 5 is not in 25."""
    out = []
    for m in NUMBER.finditer(_clean(s)):
        value = Decimal(m[1].replace(",", "") + (m[2] or ""))
        out.append((m[0].strip(), value, bool(m[3])))
    return out


def _same_number(a, b):
    (va, pa), (vb, pb) = a, b
    return va == vb or (pa and not pb and va / 100 == vb) or (pb and not pa and vb / 100 == va)


def check_deterministic(claim_text, source_texts):
    """None if the claim has nothing checkable; else (ok, reason)."""
    spans = QUOTE.findall(_clean(claim_text))
    nums = numbers(claim_text)
    if not spans and not nums:
        return None
    sources = set().union(*(text_variants(t) for t in source_texts))
    src_nums = [(v, p) for t in source_texts for _, v, p in numbers(t)]
    missing = []
    for sp in spans:
        if not any(q in s for q in text_variants(sp) for s in sources):
            missing.append(f'quote "{sp}"')
    for raw, v, p in nums:
        if not any(_same_number((v, p), m) for m in src_nums):
            missing.append(f"number {raw}")
    if missing:
        return False, "not found in cited items: " + ", ".join(missing)
    found = [f'quote "{s}"' for s in spans] + [f"number {raw}" for raw, _, _ in nums]
    return True, "found in cited items: " + ", ".join(found)


def judge(claim_text, source_texts, backend=None, model=None):
    """Ask the judge LLM; it sees only the claim and the cited items. Returns (bool, reason)."""
    items = "\n\n".join(f"Item {i + 1}:\n{t}" for i, t in enumerate(source_texts))
    prompt = f"Claim: {claim_text}\n\nCited items:\n\n{items}"
    err = None
    for _ in range(2):
        p = prompt if err is None else f"{prompt}\n\nYour previous response was invalid ({err})."
        try:
            out = llm.generate(p, system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA,
                               backend=backend, model=model)
            return out["supported"], out["reason"]
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
    return False, f"judge failed twice: {err}"


def verify_claim(claim, shown, judge_backend=None, judge_model=None):
    """claim: {text, cited_ids, quotes_directly}; shown: {id: chunk} of items the model saw."""
    ids = list(dict.fromkeys(claim["cited_ids"]))
    valid = [i for i in ids if i in shown]
    invalid = [i for i in ids if i not in shown]
    out = {**claim, "invalid_ids": invalid}
    if not valid:
        why = f"cited IDs not among retrieved items: {invalid}" if invalid else "no cited IDs"
        return out | {"verdict": "no_citation", "path": "none", "reason": why}
    texts = [shown[i]["text"] for i in valid]
    note = f" (invalid cited IDs ignored: {invalid})" if invalid else ""
    if claim["quotes_directly"]:
        res = check_deterministic(claim["text"], texts)
        if res is not None:
            ok, why = res
            return out | {"verdict": "verified" if ok else "unverified",
                          "path": "deterministic", "reason": why + note}
    ok, why = judge(claim["text"], texts, judge_backend, judge_model)
    return out | {"verdict": "verified" if ok else "unverified", "path": "judge",
                  "reason": why + note}
