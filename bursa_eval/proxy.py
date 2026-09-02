"""Build the pinned 200-row internal MMLU enterprise proxy offline."""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from bursa_eval.repro import sha256_file, write_json


SUBJECTS = (
    "business_ethics",
    "management",
    "marketing",
    "professional_accounting",
)
CHOICE_LABELS = ("A", "B", "C", "D")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_subject(source: Path, subject: str) -> list[dict]:
    if source.is_file():
        return [
            row for row in _load_jsonl(source)
            if row.get("subject") == subject
        ]
    jsonl = source / f"{subject}.jsonl"
    if jsonl.exists():
        return _load_jsonl(jsonl)
    csv_path = source / f"{subject}_test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"missing {jsonl} or {csv_path}; export the pinned cais/mmlu revision first"
        )
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for index, values in enumerate(csv.reader(handle)):
            if len(values) < 6:
                raise ValueError(f"{csv_path}:{index + 1}: expected six columns")
            rows.append({
                "question": values[0],
                "choices": values[1:5],
                "answer": values[5],
                "subject": subject,
                "source_row": index,
            })
    return rows


def _fetch_subject(subject: str, revision: str) -> list[dict]:
    query = urllib.parse.urlencode({
        "dataset": "cais/mmlu",
        "config": subject,
        "split": "test",
        "offset": 0,
        "length": 100,
        "revision": revision,
    })
    with urllib.request.urlopen(
        f"https://datasets-server.huggingface.co/rows?{query}", timeout=60
    ) as response:
        payload = json.loads(response.read())
    return [
        {**item["row"], "source_row": item["row_idx"]}
        for item in payload["rows"]
    ]


def _choice_label(answer) -> str:
    if isinstance(answer, int) or str(answer).isdigit():
        index = int(answer)
        if 0 <= index < 4:
            return CHOICE_LABELS[index]
    answer = str(answer).strip().upper()
    if answer in CHOICE_LABELS:
        return answer
    raise ValueError(f"invalid MMLU answer: {answer!r}")


def build_proxy(
    source: str | Path,
    output: str | Path,
    *,
    revision: str,
    seed: int = 3407,
    per_subject: int = 50,
) -> dict:
    source_path = Path(source)
    selected = []
    selected_ids = []
    for subject in SUBJECTS:
        rows = (
            _fetch_subject(subject, revision)
            if str(source) == "huggingface"
            else _load_subject(source_path, subject)
        )
        ranked = sorted(
            enumerate(rows),
            key=lambda item: hashlib.sha256(
                f"{seed}:{subject}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < per_subject:
            raise ValueError(
                f"{subject} contains {len(ranked)}/{per_subject} required rows"
            )
        for source_index, row in ranked[:per_subject]:
            choices = row.get("choices")
            if not isinstance(choices, list) or len(choices) != 4:
                raise ValueError(f"{subject}:{source_index}: expected four choices")
            case_id = f"mmlu-{subject}-{source_index:04d}"
            prompt = "\n".join([
                str(row["question"]),
                *[
                    f"{label}. {choice}"
                    for label, choice in zip(CHOICE_LABELS, choices)
                ],
                "Answer with only the letter A, B, C, or D.",
            ])
            selected.append({
                "id": case_id,
                "subject": subject,
                "prompt": prompt,
                "expected": _choice_label(row["answer"]),
                "scoring": "multiple_choice",
                "source_row": row.get("source_row", source_index),
            })
            selected_ids.append(case_id)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "label": "internal_mmlu_enterprise_proxy",
        "official": False,
        "dataset": "cais/mmlu",
        "revision": revision,
        "license": "MIT",
        "seed": seed,
        "subjects": list(SUBJECTS),
        "rows_per_subject": per_subject,
        "selected_ids": selected_ids,
        "output_sha256": sha256_file(target),
    }
    write_json(target.with_suffix(".manifest.json"), manifest)
    return manifest
