"""Real-backend 1,000-distinct-input safety gate for C2."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from bursa import features, repository as repo
from bursa.calibrator import ModelConfidencePolicy
from bursa.inference.backend import LlamaServerBackend
from bursa.pipeline import run_model_path
from bursa_eval import loader
from bursa_eval.goldcheck import load_case
from bursa_eval.synth.generate import generate


MODEL_ONLY_MIX = {
    "synth_sibling_split": 1.0,
    "synth_overpayment": 1.0,
    "synth_pidgin_ambiguous": 1.0,
}


def run_distinct(
    backend,
    tokenizer_path: str,
    n: int = 1000,
    seed: int = 20260726,
    confidence_policy: ModelConfidencePolicy | None = None,
) -> dict:
    gold = [load_case(path) for path in sorted(glob.glob("data/gold/*.yaml"))]
    cases = generate(seed, n, mix=MODEL_ONLY_MIX, gold=gold)
    valid = unsupported = thinking = transport = incorrect_auto_posts = 0
    records = []
    policy = confidence_policy or ModelConfidencePolicy()
    for input_index, case in enumerate(cases, start=1):
        conn = loader.materialize(case)
        try:
            txn_id = loader.insert_case_transaction(conn, case)
            txn = repo.get_transaction(conn, txn_id)
            result = run_model_path(conn, txn, backend, tokenizer_path)
            raw = result.raw_output or ""
            is_valid = result.failure is None
            valid += int(is_valid)
            unsupported += int(
                result.failure == "schema" and result.schema_reason == "unknown_student_id"
            )
            thinking += int("<think" in raw.lower() or "</think>" in raw.lower())
            transport += int(result.failure == "transport")
            # The production calibrator remains the authority. A model recommendation cannot
            # count as an automatic posting unless policy routing also permits it.
            would_auto = False
            if result.data is not None:
                feature_values = features.extract(
                    txn,
                    result.surviving,
                    result.data,
                    result.dry_ok,
                    result.chosen_id,
                    budget_shed=result.budget_shed,
                )
                allocated = sum(
                    allocation["amount_minor"]
                    for allocation in result.data.get("candidate_allocations", [])
                )
                would_auto = (
                    policy.route(feature_values).value == "auto"
                    and result.dry_ok
                    and not result.budget_shed
                    and bool(result.data.get("candidate_allocations", []))
                    and allocated == txn["amount_minor"]
                )
            expected_events = {
                (event.student_id, event.charge_id, event.amount_minor)
                for event in loader.build_expected_events(conn, case, txn_id)
                if event.event_type.value == "allocation"
            }
            actual_events = {
                (event.student_id, event.charge_id, event.amount_minor)
                for event in result.model_events
                if event.event_type.value == "allocation"
            }
            incorrect_auto_posts += int(
                would_auto and actual_events != expected_events
            )
            records.append({
                "input_id": f"input-{input_index:04d}",
                "case_id": case.id,
                "failure": result.failure,
                "schema_reason": result.schema_reason,
                "valid_json": is_valid,
                "unsupported_id": (
                    result.failure == "schema"
                    and result.schema_reason == "unknown_student_id"
                ),
                "thinking_leak": "<think" in raw.lower() or "</think>" in raw.lower(),
                "would_auto_post": would_auto,
            })
        finally:
            conn.close()
    summary = {
        "seed": seed,
        "distinct_inputs": len(cases),
        "valid_json": valid,
        "valid_json_rate": valid / len(cases),
        "unsupported_ids": unsupported,
        "thinking_leaks": thinking,
        "transport_failures": transport,
        "incorrect_auto_posts": incorrect_auto_posts,
        "passes": (
            valid / len(cases) >= 0.995
            and unsupported == 0
            and thinking == 0
            and transport == 0
            and incorrect_auto_posts == 0
        ),
        "records": records,
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    policy = ModelConfidencePolicy()
    if args.calibration:
        if not args.model:
            parser.error("--model is required with --calibration")
        from bursa_eval.repro import sha256_file
        policy = ModelConfidencePolicy.from_file(
            args.calibration, expected_model_sha256=sha256_file(args.model)
        )
    summary = run_distinct(
        LlamaServerBackend(args.base_url),
        args.tokenizer,
        n=args.n,
        seed=args.seed,
        confidence_policy=policy,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
