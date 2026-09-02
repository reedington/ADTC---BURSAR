"""Deterministic general-enterprise examples and offline OASST1 filtering."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


_OWNED_TASKS = {
    "summarization": (
        "Summarize this fictional operations note in three bullets: {text}",
        "Key points:\n- {point1}\n- {point2}\n- {point3}",
    ),
    "drafting": (
        "Draft a concise internal email about this fictional decision: {text}",
        "Subject: {subject}\n\nHello team,\n\n{body}\n\nRegards,\nOperations",
    ),
    "analysis": (
        "Analyse this fictional business situation and identify the main risk: {text}",
        "The main risk is {risk}. The team should {action}.",
    ),
    "decision_support": (
        "Recommend one action for this fictional organisation: {text}",
        "Recommendation: {action}. Rationale: {rationale}.",
    ),
}

_SCENARIOS = (
    "a supplier lead time increased while demand stayed stable",
    "a branch has repeated invoice-entry errors",
    "customer response time rose after a staffing change",
    "monthly transport costs exceeded the approved budget",
    "two teams use conflicting versions of the same policy",
    "a school office needs a safer month-end close checklist",
    "inventory counts disagree with the receiving log",
    "a service desk backlog doubled over three weeks",
)


def project_owned_examples(count: int, *, seed: int = 3407) -> list[dict]:
    randomizer = random.Random(seed)
    categories = tuple(_OWNED_TASKS)
    rows = []
    for index in range(count):
        category = categories[index % len(categories)]
        prompt_template, completion_template = _OWNED_TASKS[category]
        scenario = _SCENARIOS[(index + randomizer.randrange(len(_SCENARIOS))) % len(_SCENARIOS)]
        values = {
            "text": f"{scenario}; example reference GE-{index:05d}",
            "point1": scenario.capitalize(),
            "point2": "The evidence is fictional and should be verified before action",
            "point3": "Assign an owner and review date",
            "subject": f"Decision GE-{index:05d}",
            "body": f"We will address {scenario}. Please confirm the owner and review date.",
            "risk": scenario,
            "action": "assign an accountable owner, verify the evidence, and set a review date",
            "rationale": "it is reversible, evidence-led, and gives the team a clear checkpoint",
        }
        rows.append({
            "id": f"owned-{category}-{index:05d}",
            "format": "chat",
            "category": category,
            "prompt": prompt_template.format(**values),
            "completion": completion_template.format(**values),
            "provenance": "project_owned",
            "license": "Apache-2.0",
        })
    return rows


def _norm_hash(prompt: str, completion: str) -> str:
    normalized = " ".join((prompt + "\n" + completion).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_oasst_pairs(
    path: str | Path, count: int, *, seed: int = 3407, revision: str
) -> list[dict]:
    messages = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                message_id = row.get("message_id")
                if message_id:
                    messages[message_id] = row
    candidates = []
    for row in messages.values():
        parent = messages.get(row.get("parent_id"))
        if (
            row.get("role") != "assistant"
            or not parent
            or parent.get("role") != "prompter"
            or row.get("lang") != "en"
            or parent.get("lang") != "en"
            or row.get("deleted")
            or parent.get("deleted")
            or row.get("rank") != 0
        ):
            continue
        prompt = str(parent.get("text") or "").strip()
        completion = str(row.get("text") or "").strip()
        if not 20 <= len(prompt) <= 1500 or not 20 <= len(completion) <= 2000:
            continue
        digest = _norm_hash(prompt, completion)
        candidates.append((digest, row, parent, prompt, completion))
    candidates.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]}".encode("utf-8")
        ).hexdigest()
    )
    selected, seen = [], set()
    for digest, row, parent, prompt, completion in candidates:
        if digest in seen:
            continue
        seen.add(digest)
        selected.append({
            "id": f"oasst1-{row['message_id']}",
            "format": "chat",
            "prompt": prompt,
            "completion": completion,
            "provenance": "OpenAssistant/oasst1",
            "license": "Apache-2.0",
            "source_revision": revision,
            "source_row_ids": [parent["message_id"], row["message_id"]],
        })
        if len(selected) == count:
            return selected
    raise ValueError(f"OASST1 filter produced {len(selected)}/{count} required pairs")
