"""ADTC validation runner — an lm-eval-shaped adapter over backend.chat() (embedded template).
The official judge-distributed set drops in unchanged as data/adtc/official/*.jsonl; a
permissively-licensed proxy set is the interim C3 forgetting-detector (RELATIVE delta only)."""
import json
import re


def load_adtc(path) -> list[dict]:
    """Each line: {id, prompt, expected, choices?}."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _norm(s) -> str:
    return " ".join(str(s).strip().lower().split())


def score_adtc(cases, backend, label: str) -> dict:
    """Run prompts through the embedded template with explicit scoring contracts."""
    per_id, correct = {}, 0
    for c in cases:
        out = backend.chat(c["prompt"])
        if c.get("scoring") == "multiple_choice":
            match = re.search(
                r"(?:answer\s*(?:is|:)?\s*)?\b([ABCD])\b",
                str(out).upper(),
            )
            hit = bool(match and match.group(1) == str(c["expected"]).upper())
        else:
            hit = _norm(c["expected"]) in _norm(out)
        per_id[c["id"]] = hit
        correct += int(hit)
    return {"label": label, "accuracy": (correct / len(cases)) if cases else None, "per_id": per_id}


def regression_delta(pre: dict, post: dict) -> float:
    """Relative delta only (forgetting detector). Never compares absolute proxy scores."""
    return post["accuracy"] - pre["accuracy"]
