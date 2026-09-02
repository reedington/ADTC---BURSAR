"""Bare-model suite — generic-enterprise prompts + the two D14 visible prompts, run through
the model's EMBEDDED chat template (backend.chat). Automated checks are structural (non-empty +
the D6 reconciliation-JSON leak tripwire); open-ended quality goes to the human side-by-side."""
import json
from dataclasses import dataclass

_LEAK_MARKERS = ("candidate_allocations", "recommended_action")


@dataclass
class BareRecord:
    suite: str
    case_id: str
    prompt: str
    output: str
    valid: bool
    format_leak: bool


def load_bare_prompts(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _has_leak(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _LEAK_MARKERS)


def run_bare_suite(prompts, backend) -> list[BareRecord]:
    records = []
    for p in prompts:
        out = backend.chat(p["prompt"])
        records.append(BareRecord(
            suite="bare_model", case_id=p["id"], prompt=p["prompt"], output=out,
            valid=bool(out and out.strip()), format_leak=_has_leak(out)))
    return records
