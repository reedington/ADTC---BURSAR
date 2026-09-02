"""Fit and verify Bursa's portable logistic confidence artifact."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from bursa import features, repository as repo
from bursa.calibrator import (
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_KEYS,
    CalibrationArtifact,
    ModelConfidencePolicy,
)
from bursa.features import FEATURES_VERSION
from bursa.inference.backend import LlamaServerBackend
from bursa.inference.server import LlamaServer
from bursa.pipeline import run_model_path
from bursa_eval import loader
from bursa_eval.models import GoldCase
from bursa_eval.repro import (
    canonical_sha256,
    environment_fingerprint,
    git_commit,
    require_clean,
    sha256_file,
    write_json,
)


def load_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                features = row["features"]
                label = row["label"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid calibration row") from exc
            if not isinstance(features, dict) or not isinstance(label, bool):
                raise ValueError(
                    f"{path}:{line_number}: features must be an object and label a boolean"
                )
            if int(features.get("features_version", -1)) != FEATURES_VERSION:
                raise ValueError(f"{path}:{line_number}: incompatible features_version")
            records.append(row)
    return records


def _matrix(records: list[dict]) -> tuple[list[list[float]], list[int]]:
    x = [
        [float(row["features"].get(feature, 0)) for feature in FEATURE_KEYS]
        for row in records
    ]
    y = [1 if row["label"] else 0 for row in records]
    return x, y


def _standardize(x: list[list[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    width = len(FEATURE_KEYS)
    means = [sum(row[index] for row in x) / len(x) for index in range(width)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in x) / len(x)
        scales.append(math.sqrt(variance) or 1.0)
    standardized = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in x
    ]
    return means, scales, standardized


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def fit_logistic(
    records: list[dict],
    *,
    seed: int = 3407,
    iterations: int = 4000,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> tuple[list[float], list[float], float, list[float]]:
    if not records:
        raise ValueError("no fitting records")
    x, y = _matrix(records)
    means, scales, x = _standardize(x)
    randomizer = random.Random(seed)
    weights = [randomizer.uniform(-0.001, 0.001) for _ in FEATURE_KEYS]
    positives = sum(y)
    prevalence = min(max(positives / len(y), 1e-6), 1 - 1e-6)
    intercept = math.log(prevalence / (1 - prevalence))
    for _ in range(iterations):
        grad_b = 0.0
        grad_w = [0.0] * len(weights)
        for row, label in zip(x, y):
            probability = _sigmoid(intercept + sum(w * v for w, v in zip(weights, row)))
            error = probability - label
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value
        count = len(x)
        intercept -= learning_rate * grad_b / count
        for index in range(len(weights)):
            gradient = grad_w[index] / count + l2 * weights[index]
            weights[index] -= learning_rate * gradient
    return means, scales, intercept, weights


def choose_threshold(
    policy: ModelConfidencePolicy, records: list[dict], *, starting: float = 0.90
) -> tuple[float, bool, int]:
    false_positive_scores = [
        policy.score(row["features"]) for row in records if not row["label"]
    ]
    threshold = starting
    if false_positive_scores:
        threshold = max(
            starting, math.nextafter(max(false_positive_scores), math.inf)
        )
    enabled = threshold <= 1.0
    return min(threshold, 1.0), enabled, len(false_positive_scores)


def fit_artifact(
    fit_records: list[dict],
    threshold_records: list[dict],
    *,
    model_sha256: str,
    source_manifest_sha256: str,
    seed: int = 3407,
) -> CalibrationArtifact:
    if not threshold_records:
        raise ValueError("threshold selection requires held-out records")
    positives = sum(1 for row in fit_records if row["label"])
    negatives = len(fit_records) - positives
    means, scales, intercept, coefficients = fit_logistic(fit_records, seed=seed)
    provisional = CalibrationArtifact(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        features_version=FEATURES_VERSION,
        model_sha256=model_sha256,
        feature_names=FEATURE_KEYS,
        means=tuple(means),
        scales=tuple(scales),
        intercept=intercept,
        coefficients=tuple(coefficients),
        auto_threshold=0.90,
        review_threshold=0.65,
        auto_enabled=False,
        fit_positive_count=positives,
        fit_negative_count=negatives,
        seed=seed,
        source_manifest_sha256=source_manifest_sha256,
    )
    threshold, zero_false_possible, threshold_negatives = choose_threshold(
        ModelConfidencePolicy(provisional), threshold_records
    )
    enough_samples = (
        positives >= 25
        and negatives >= 25
        and len(threshold_records) >= 200
    )
    return CalibrationArtifact(
        **{
            **provisional.__dict__,
            "auto_threshold": threshold,
            "auto_enabled": bool(enough_samples and zero_false_possible),
            "threshold_case_count": len(threshold_records),
            "threshold_negative_count": threshold_negatives,
            "threshold_false_positive_count": 0,
        }
    )


def verify_artifact(artifact: CalibrationArtifact, records: list[dict]) -> dict:
    policy = ModelConfidencePolicy(artifact)
    false_auto = [
        row.get("case_id", index)
        for index, row in enumerate(records)
        if not row["label"] and policy.route(row["features"]).value == "auto"
    ]
    return {
        "records": len(records),
        "false_auto_count": len(false_auto),
        "false_auto_case_ids": false_auto,
        "auto_enabled": artifact.auto_enabled,
        "passed": not false_auto,
    }


def _events_set(events) -> set[tuple]:
    return {
        (event.student_id, event.charge_id, event.amount_minor)
        for event in events
        if getattr(event, "event_type", None).value == "allocation"
    }


def collect_records(
    cases_path: str | Path,
    backend,
    tokenizer_path: str | None,
) -> dict[str, list[dict]]:
    groups = {"fit": [], "threshold": []}
    with Path(cases_path).open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    for source in source_rows:
        case = GoldCase.model_validate(source["case"])
        split_name = source["split"]
        if split_name not in groups:
            raise ValueError(f"{case.id}: unknown calibration split {split_name}")
        conn = loader.materialize(case)
        try:
            txn_id = loader.insert_case_transaction(conn, case)
            txn = repo.get_transaction(conn, txn_id)
            expected = _events_set(loader.build_expected_events(conn, case, txn_id))
            result = run_model_path(conn, txn, backend, tokenizer_path)
            feature_values = {"features_version": FEATURES_VERSION}
            feature_values.update({key: 0 for key in FEATURE_KEYS})
            exact = False
            safe_remainder = False
            if result.data is not None:
                feature_values = features.extract(
                    txn,
                    result.surviving,
                    result.data,
                    result.dry_ok,
                    result.chosen_id,
                    budget_shed=result.budget_shed,
                )
                exact = _events_set(result.model_events) == expected
                allocated = sum(
                    allocation["amount_minor"]
                    for allocation in result.data.get("candidate_allocations", [])
                )
                safe_remainder = allocated == txn["amount_minor"]
            label = bool(
                result.failure is None
                and exact
                and result.dry_ok
                and safe_remainder
                and not case.expected.credits
            )
            groups[split_name].append({
                "case_id": case.id,
                "label": label,
                "features": feature_values,
                "observation": {
                    "failure": result.failure,
                    "schema_reason": result.schema_reason,
                    "exact_allocation": exact,
                    "dry_run_ok": result.dry_ok,
                    "safe_remainder": safe_remainder,
                    "has_expected_credit": bool(case.expected.credits),
                },
            })
        finally:
            conn.close()
    return groups


def write_collected(
    groups: dict[str, list[dict]],
    output_dir: str | Path,
    *,
    model_sha256: str,
    cases_sha256: str,
) -> dict:
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty; calibration collection is immutable")
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for split_name, rows in groups.items():
        path = target / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        files[path.name] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "environment": environment_fingerprint(),
        "model_sha256": model_sha256,
        "cases_sha256": cases_sha256,
        "counts": {key: len(value) for key, value in groups.items()},
        "labels": {
            key: {
                "positive": sum(row["label"] for row in value),
                "negative": sum(not row["label"] for row in value),
            }
            for key, value in groups.items()
        },
        "files": files,
    }
    write_json(target / "collection_manifest.json", manifest)
    return manifest


def _model_sha(args) -> str:
    if args.model_sha256:
        return args.model_sha256.lower()
    if not args.model:
        raise ValueError("provide --model or --model-sha256")
    return sha256_file(args.model)


def _load_config(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported calibration configuration schema")
    locked = {
        "feature_version": FEATURES_VERSION,
        "minimum_positive": 25,
        "minimum_negative": 25,
        "auto_threshold_start": 0.90,
        "review_threshold": 0.65,
    }
    for key, expected in locked.items():
        if value.get(key) != expected:
            raise ValueError(f"calibration config {key} must remain {expected}")
    if value.get("fit_cases", 0) + value.get("threshold_cases", 0) != 600:
        raise ValueError("calibration config must define the isolated 400/200 corpus")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument(
        "--config", default="configs/calibration/logistic-v1.json"
    )
    collect.add_argument("--cases", required=True)
    collect.add_argument("--model", required=True)
    collect.add_argument("--tokenizer")
    collect.add_argument("--output", required=True)
    collect.add_argument("--port", type=int, default=18090)
    collect.add_argument("--allow-dirty", action="store_true")

    fit = sub.add_parser("fit")
    fit.add_argument(
        "--config", default="configs/calibration/logistic-v1.json"
    )
    fit.add_argument("--fit-data", required=True)
    fit.add_argument("--threshold-data", required=True)
    fit.add_argument("--validation-data", action="append", default=[])
    fit.add_argument("--model")
    fit.add_argument("--model-sha256")
    fit.add_argument("--source-manifest", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=3407)
    fit.add_argument("--allow-dirty", action="store_true")

    verify = sub.add_parser("verify")
    verify.add_argument(
        "--config", default="configs/calibration/logistic-v1.json"
    )
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--data", action="append", required=True)
    verify.add_argument("--model")
    verify.add_argument("--model-sha256")
    verify.add_argument("--allow-dirty", action="store_true")

    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        if args.command == "collect":
            require_clean(allow_dirty=args.allow_dirty)
            with Path(args.cases).open(encoding="utf-8") as handle:
                calibration_rows = [
                    json.loads(line) for line in handle if line.strip()
                ]
            split_counts = {
                name: sum(row.get("split") == name for row in calibration_rows)
                for name in ("fit", "threshold")
            }
            if split_counts != {
                "fit": config["fit_cases"],
                "threshold": config["threshold_cases"],
            }:
                raise ValueError(
                    f"calibration cases do not match configured 400/200 split: "
                    f"{split_counts}"
                )
            model_hash = sha256_file(args.model)
            server = LlamaServer(args.model, port=args.port, threads=4, ctx=2048)
            server.start()
            if not server.wait_until_ready():
                server.stop()
                raise RuntimeError("llama-server did not become ready")
            try:
                groups = collect_records(
                    args.cases,
                    LlamaServerBackend(
                        f"http://127.0.0.1:{args.port}", timeout=120
                    ),
                    args.tokenizer,
                )
            finally:
                server.stop()
            manifest = write_collected(
                groups,
                args.output,
                model_sha256=model_hash,
                cases_sha256=sha256_file(args.cases),
            )
            manifest["configuration_sha256"] = sha256_file(args.config)
            manifest["command_arguments"] = vars(args)
            write_json(
                Path(args.output) / "collection_manifest.json", manifest
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.command == "fit":
            require_clean(allow_dirty=args.allow_dirty)
            fitting = load_records(args.fit_data)
            thresholding = load_records(args.threshold_data)
            if len(fitting) != config["fit_cases"]:
                raise ValueError("fitting observations do not match configured count")
            if len(thresholding) != config["threshold_cases"]:
                raise ValueError("threshold observations do not match configured count")
            if args.seed != config["seed"]:
                raise ValueError("fit seed does not match calibration configuration")
            for path in args.validation_data:
                thresholding.extend(load_records(path))
            source_hash = sha256_file(args.source_manifest)
            artifact = fit_artifact(
                fitting,
                thresholding,
                model_sha256=_model_sha(args),
                source_manifest_sha256=source_hash,
                seed=args.seed,
            )
            artifact.validate()
            output = Path(args.output)
            manifest_output = output.with_suffix(output.suffix + ".manifest.json")
            if output.exists() or manifest_output.exists():
                raise FileExistsError(
                    f"calibration output is immutable: {output}"
                )
            write_json(output, artifact.to_dict())
            report = verify_artifact(artifact, thresholding)
            write_json(manifest_output, {
                "schema_version": 1,
                "git_commit": git_commit(),
                "environment": environment_fingerprint(),
                "command_arguments": vars(args),
                "configuration_sha256": sha256_file(args.config),
                "artifact_sha256": sha256_file(output),
                "fit_data_sha256": sha256_file(args.fit_data),
                "threshold_data_sha256": sha256_file(args.threshold_data),
                "validation_data_sha256": {
                    path: sha256_file(path) for path in args.validation_data
                },
                "source_manifest_sha256": source_hash,
                "model_sha256": artifact.model_sha256,
                "verification": report,
            })
            print(json.dumps(report, sort_keys=True))
            return 0 if report["passed"] else 1

        require_clean(allow_dirty=args.allow_dirty)
        artifact = CalibrationArtifact.from_dict(
            json.loads(Path(args.artifact).read_text(encoding="utf-8")),
            expected_model_sha256=_model_sha(args),
        )
        records = []
        for path in args.data:
            records.extend(load_records(path))
        report = verify_artifact(artifact, records)
        report["artifact_sha256"] = canonical_sha256(artifact.to_dict())
        report["configuration_sha256"] = sha256_file(args.config)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 1
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
