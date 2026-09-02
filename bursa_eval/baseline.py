"""Run, review, compare, and freeze Bursa's four-model zero-shot baseline."""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from bursa.inference.backend import LlamaServerBackend
from bursa.inference.server import LlamaServer
from bursa_eval.goldcheck import load_case
from bursa_eval.harness.adapters import NativeTemplateBackend, QwenEvaluationBackend
from bursa_eval.harness.adtc import load_adtc, score_adtc
from bursa_eval.harness.baremodel import load_bare_prompts, run_bare_suite
from bursa_eval.harness.runner import run_gold_suite
from bursa_eval.harness.scorecard import build_scorecard, write_run
from bursa_eval.proxy import build_proxy
from bursa_eval.repro import (
    environment_fingerprint,
    git_commit,
    immutable_run_dir,
    require_clean,
    require_disk,
    sha256_file,
    write_json,
)


BASELINE_LABEL = "internal_mmlu_enterprise_proxy"
QWEN_SMALL = "qwen3-0.6b-q4_k_m"
QWEN_LARGE = "qwen3-1.7b-q4_k_m"


def load_models(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not value.get("models"):
        raise ValueError("unsupported or empty baseline model manifest")
    runtime = value.get("runtime", {})
    if runtime.get("threads") != 4 or runtime.get("context") != 2048:
        raise ValueError("baseline runtime must remain at four threads and context 2048")
    return value


def verify_model(model: dict) -> None:
    path = Path(model["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{model['id']}: missing artifact {path}")
    if path.stat().st_size != int(model["size_bytes"]):
        raise ValueError(f"{model['id']}: size does not match pinned manifest")
    digest = sha256_file(path)
    if digest != model["sha256"]:
        raise ValueError(f"{model['id']}: SHA-256 mismatch")


def fetch_models(models: list[dict], *, accept_third_party_terms: bool) -> list[dict]:
    if any(model["license"] != "Apache-2.0" for model in models) and not (
        accept_third_party_terms
    ):
        raise ValueError(
            "Gemma/Llama downloads require --accept-third-party-terms"
        )
    authorization = os.environ.get("HF_TOKEN")
    results = []
    for model in models:
        target = Path(model["path"])
        if target.exists():
            verify_model(model)
            results.append({"id": model["id"], "status": "verified"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        filename = target.name
        url = (
            f"https://huggingface.co/{model['source']}/resolve/"
            f"{model['source_revision']}/{filename}"
        )
        request = urllib.request.Request(url)
        if authorization:
            request.add_header("Authorization", f"Bearer {authorization}")
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                "wb"
            ) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if partial.stat().st_size != int(model["size_bytes"]):
                raise ValueError(f"{model['id']}: downloaded size mismatch")
            if sha256_file(partial) != model["sha256"]:
                raise ValueError(f"{model['id']}: downloaded SHA-256 mismatch")
            partial.replace(target)
            results.append({"id": model["id"], "status": "downloaded"})
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise
    return results


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _temperature_c() -> float | None:
    values = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            value = float(Path(path).read_text().strip())
            values.append(value / 1000 if value > 500 else value)
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def _rss_mib(pid: int | None) -> float | None:
    if pid is None:
        return None
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip()) / 1024
    except ValueError:
        return None


def run_llama_bench(model_path: str, *, threads: int = 4) -> dict:
    command = [
        "llama-bench",
        "--model", model_path,
        "--threads", str(threads),
        "--n-prompt", "512",
        "--n-gen", "128",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--offline",
        "--repetitions", "1",
        "--output", "json",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"llama-bench failed: {result.stderr.strip()}")
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("llama-bench did not emit valid JSON") from exc
    if isinstance(rows, dict):
        rows = [rows]
    generation = [
        row for row in rows
        if int(row.get("n_gen", row.get("n_gen_tokens", 0)) or 0) > 0
    ]
    prompt = [
        row for row in rows
        if int(row.get("n_prompt", row.get("n_prompt_tokens", 0)) or 0) > 0
        and int(row.get("n_gen", row.get("n_gen_tokens", 0)) or 0) == 0
    ]

    def speed(rows):
        values = [
            float(row.get("avg_ts", row.get("tokens_per_second", 0)) or 0)
            for row in rows
        ]
        return sum(values) / len(values) if values else None

    return {
        "generation_tokens_per_second": speed(generation),
        "prompt_tokens_per_second": speed(prompt),
        "threads": threads,
        "context": 2048,
        "kv_cache": "q8_0",
        "elapsed_seconds": time.perf_counter() - started,
        "raw": rows,
    }


def _selected_cases(
    gold_dir: str,
    manifest_path: str | None,
    allow_unfrozen: bool,
    *,
    split: str = "development",
):
    cases = [
        load_case(path) for path in sorted(Path(gold_dir).glob("*.yaml"))
    ]
    if not manifest_path:
        if not allow_unfrozen or split == "test":
            raise ValueError("a frozen dataset manifest is required")
        return cases
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("frozen") and (not allow_unfrozen or split == "test"):
        raise ValueError("dataset manifest is not frozen")
    test_ids = set(manifest.get("test_case_ids", []))
    if split == "test":
        if not test_ids:
            raise ValueError("frozen dataset manifest has no sealed test IDs")
        selected = [case for case in cases if case.id in test_ids]
        if {case.id for case in selected} != test_ids:
            raise ValueError("one or more sealed test cases are missing from the gold set")
        return selected
    return [case for case in cases if case.id not in test_ids]


def _backend_for(model: dict, base: LlamaServerBackend):
    if model["family"] == "qwen3":
        return QwenEvaluationBackend(base)
    return NativeTemplateBackend(base, model["family"])


def run_one(
    model: dict,
    *,
    output_dir: Path,
    gold_cases: list,
    enterprise_path: str,
    bare_path: str,
    tokenizer_path: str | None,
    port: int,
) -> dict:
    verify_model(model)
    bench = run_llama_bench(model["path"], threads=4)
    server = LlamaServer(model["path"], port=port, threads=4, ctx=2048)
    server.start()
    if not server.wait_until_ready():
        server.stop()
        raise RuntimeError(f"{model['id']}: llama-server did not become ready")
    try:
        base = LlamaServerBackend(f"http://127.0.0.1:{port}", timeout=120)
        backend = _backend_for(model, base)
        records = run_gold_suite(
            gold_cases,
            backend,
            tokenizer_path if model["family"] == "qwen3" else None,
        )
        enterprise = score_adtc(
            load_adtc(enterprise_path), backend, label=BASELINE_LABEL
        )
        bare = run_bare_suite(load_bare_prompts(bare_path), backend)
        bench["server_rss_mib"] = _rss_mib(server.process_id)
        bench["temperature_c"] = _temperature_c()
        cpu = _cpu_name()
        bench["measured_on"] = cpu
        bench["qualifying_i5"] = bool(re.search(r"\bi5[-\s]", cpu, re.IGNORECASE))
        provenance = {
            "git_commit": git_commit(),
            "model_id": model["id"],
            "model_path": model["path"],
            "model_sha256": model["sha256"],
            "source": model["source"],
            "source_revision": model["source_revision"],
            "license": model["license"],
            "production_candidate": model["production_candidate"],
            "enterprise_label": BASELINE_LABEL,
            "environment": environment_fingerprint(),
        }
        scorecard = build_scorecard(
            records,
            adtc_res=enterprise,
            bare_records=bare,
            provenance=provenance,
            perf=bench,
        )
        write_run(output_dir, records, scorecard)
        return scorecard
    finally:
        server.stop()


def create_review_packet(
    run_root: str | Path,
    output: str | Path,
    *,
    seed: int = 3407,
) -> tuple[Path, Path]:
    root = Path(run_root)
    responses = []
    mapping = {}
    for scorecard_path in sorted(root.glob("*/scorecard.json")):
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        model_id = scorecard["provenance"]["model_id"]
        for record in scorecard.get("bare_model") or []:
            responses.append((model_id, record))
    random.Random(seed).shuffle(responses)
    packet = []
    for index, (model_id, record) in enumerate(responses, 1):
        review_id = f"blind-{index:04d}"
        mapping[review_id] = model_id
        packet.append({
            "review_id": review_id,
            "case_id": record["case_id"],
            "prompt": record["prompt"],
            "response": record["output"],
            "scores": {
                "correctness": None,
                "coherence": None,
                "no_schema_leak": None,
                "appropriate_abstention": None,
            },
            "reviewer_notes": "",
        })
    output_path = Path(output)
    _write_packet(output_path, packet)
    key_path = output_path.with_suffix(".key.json")
    write_json(key_path, {"seed": seed, "mapping": mapping})
    return output_path, key_path


def _write_packet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score_review(packet: str | Path, key: str | Path) -> dict:
    mapping = json.loads(Path(key).read_text(encoding="utf-8"))["mapping"]
    totals = {}
    counts = {}
    with Path(packet).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scores = row["scores"]
            values = list(scores.values())
            if any(not isinstance(value, int) or not 0 <= value <= 2 for value in values):
                raise ValueError(f"{row['review_id']}: every rubric score must be 0, 1, or 2")
            model_id = mapping[row["review_id"]]
            totals[model_id] = totals.get(model_id, 0) + sum(values)
            counts[model_id] = counts.get(model_id, 0) + 8
    return {
        model_id: round(100 * totals[model_id] / counts[model_id], 4)
        for model_id in sorted(totals)
    }


def c2_decision(
    run_root: str | Path,
    review_scores: dict,
    *,
    model_config: dict | None = None,
) -> dict:
    scorecards = {}
    for path in Path(run_root).glob("*/scorecard.json"):
        scorecard = json.loads(path.read_text(encoding="utf-8"))
        scorecards[scorecard["provenance"]["model_id"]] = scorecard
    if len(scorecards) != 4:
        raise ValueError("C2 requires scorecards for all four baseline models")
    if model_config is not None:
        expected = {
            model["id"]: model["sha256"] for model in model_config["models"]
        }
        actual = {
            model_id: scorecard.get("provenance", {}).get("model_sha256")
            for model_id, scorecard in scorecards.items()
        }
        if actual != expected:
            raise ValueError("baseline scorecards do not match the pinned model hashes")
    if set(review_scores) != set(scorecards):
        raise ValueError("review scores must cover exactly the four baseline models")
    small, large = scorecards[QWEN_SMALL], scorecards[QWEN_LARGE]

    def gold(card):
        value = card["bursa_gold"]["exact_allocation_accuracy"]
        if value is None:
            raise ValueError("Qwen gold exact-allocation score is missing")
        return 100 * value

    large_tps = large["perf"].get("generation_tokens_per_second")
    if large_tps is None:
        raise ValueError("Qwen3-1.7B generation speed is missing")
    if not large["perf"].get("qualifying_i5"):
        raise ValueError("C2 can only freeze from qualifying i5-class measurements")
    if large_tps < 13:
        selected, reason = QWEN_SMALL, "1.7B generation speed below 13 tokens/s"
    elif (
        gold(large) - gold(small) <= 5
        and review_scores[large["provenance"]["model_id"]]
        - review_scores[small["provenance"]["model_id"]] <= 5
        and not small["gates_failed"]
    ):
        selected, reason = QWEN_SMALL, "0.6B is within five gold and bare-quality points"
    else:
        selected, reason = QWEN_LARGE, "1.7B retains the locked quality advantage"
    four_model_table = {
        model_id: {
            "production_candidate": bool(
                scorecard["provenance"].get("production_candidate")
            ),
            "model_sha256": scorecard["provenance"].get("model_sha256"),
            "gold_exact_allocation_accuracy": scorecard["bursa_gold"].get(
                "exact_allocation_accuracy"
            ),
            "gold_action_accuracy": scorecard["bursa_gold"].get(
                "action_accuracy"
            ),
            "enterprise_proxy_accuracy": (
                scorecard.get("adtc") or {}
            ).get("accuracy"),
            "bare_human_score": review_scores[model_id],
            "generation_tokens_per_second": (
                scorecard.get("perf") or {}
            ).get("generation_tokens_per_second"),
            "prompt_tokens_per_second": (
                scorecard.get("perf") or {}
            ).get("prompt_tokens_per_second"),
            "server_rss_mib": (
                scorecard.get("perf") or {}
            ).get("server_rss_mib"),
            "temperature_c": (
                scorecard.get("perf") or {}
            ).get("temperature_c"),
            "gates_failed": scorecard.get("gates_failed", []),
        }
        for model_id, scorecard in sorted(scorecards.items())
    }
    return {
        "schema_version": 1,
        "checkpoint": "C2",
        "selected_model_id": selected,
        "reason": reason,
        "frozen_runtime": {
            "quantization": "Q4_K_M",
            "context": 2048,
            "threads": 4,
            "temperature": 0,
            "kv_cache": "q8_0",
        },
        "qwen": {
            model_id: {
                "gold_exact_allocation_points": gold(scorecards[model_id]),
                "bare_human_score": review_scores[model_id],
                "generation_tokens_per_second": scorecards[model_id]["perf"][
                    "generation_tokens_per_second"
                ],
                "gates_failed": scorecards[model_id]["gates_failed"],
            }
            for model_id in (QWEN_SMALL, QWEN_LARGE)
        },
        "four_model_table": four_model_table,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    proxy = sub.add_parser("prepare-proxy")
    proxy.add_argument(
        "--source",
        default="huggingface",
        help="Local MMLU export or 'huggingface' for the pinned datasets-server API",
    )
    proxy.add_argument("--output", default="data/adtc/proxy/mmlu_enterprise.jsonl")
    proxy.add_argument("--revision", required=True)
    proxy.add_argument("--seed", type=int, default=3407)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--config", default="configs/baseline/models.json")
    fetch.add_argument("--model", default="all")
    fetch.add_argument("--accept-third-party-terms", action="store_true")
    fetch.add_argument("--minimum-free-gib", type=float, default=12.0)

    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/baseline/models.json")
    run.add_argument("--model", default="all")
    run.add_argument("--gold-dir", default="data/gold")
    run.add_argument("--dataset-manifest", default="data/manifest.json")
    run.add_argument("--enterprise", default="data/adtc/proxy/mmlu_enterprise.jsonl")
    run.add_argument("--bare", default="data/bare/prompts.jsonl")
    run.add_argument("--tokenizer", default="model/tokenizer.json")
    run.add_argument("--out", default="runs/baselines")
    run.add_argument("--allow-unfrozen", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--minimum-free-gib", type=float, default=12.0)
    run.add_argument("--port", type=int, default=18080)
    run.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    run.add_argument(
        "--test-receipt-dir", default="artifacts/frozen-test-receipts"
    )

    review = sub.add_parser("review-packet")
    review.add_argument("--run-root", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--seed", type=int, default=3407)

    score = sub.add_parser("score-review")
    score.add_argument("--packet", required=True)
    score.add_argument("--key", required=True)
    score.add_argument("--output", required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--config", default="configs/baseline/models.json")
    compare.add_argument("--run-root", required=True)
    compare.add_argument("--review-scores", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--allow-dirty", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-proxy":
            print(json.dumps(build_proxy(
                args.source,
                args.output,
                revision=args.revision,
                seed=args.seed,
            ), indent=2, sort_keys=True))
            return 0
        if args.command == "fetch":
            require_disk(".", args.minimum_free_gib)
            config = load_models(args.config)
            selected = config["models"] if args.model == "all" else [
                model for model in config["models"] if model["id"] == args.model
            ]
            if not selected:
                raise ValueError(f"unknown model id: {args.model}")
            print(json.dumps({
                "results": fetch_models(
                    selected,
                    accept_third_party_terms=args.accept_third_party_terms,
                )
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "review-packet":
            paths = create_review_packet(
                args.run_root, args.output, seed=args.seed
            )
            print(json.dumps({"packet": str(paths[0]), "key": str(paths[1])}))
            return 0
        if args.command == "score-review":
            scores = score_review(args.packet, args.key)
            write_json(args.output, scores)
            print(json.dumps(scores, indent=2, sort_keys=True))
            return 0
        if args.command == "compare":
            require_clean(allow_dirty=args.allow_dirty)
            scores = json.loads(Path(args.review_scores).read_text(encoding="utf-8"))
            decision = c2_decision(
                args.run_root,
                scores,
                model_config=load_models(args.config),
            )
            decision["git_commit"] = git_commit()
            decision["environment"] = environment_fingerprint()
            decision["command_arguments"] = vars(args)
            decision["configuration_sha256"] = sha256_file(args.config)
            decision["review_scores_sha256"] = sha256_file(args.review_scores)
            output = Path(args.output)
            if output.exists():
                raise FileExistsError(f"C2 decision is immutable: {output}")
            write_json(output, decision)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return 0

        require_clean(allow_dirty=args.allow_dirty)
        require_disk(".", args.minimum_free_gib)
        config = load_models(args.config)
        selected = config["models"] if args.model == "all" else [
            model for model in config["models"] if model["id"] == args.model
        ]
        if not selected:
            raise ValueError(f"unknown model id: {args.model}")
        for model in selected:
            verify_model(model)
        if args.split == "test" and len(selected) != 1:
            raise ValueError("sealed test evaluation runs exactly one artifact at a time")
        manifest_path = (
            args.dataset_manifest if Path(args.dataset_manifest).exists() else None
        )
        cases = _selected_cases(
            args.gold_dir,
            manifest_path,
            args.allow_unfrozen,
            split=args.split,
        )
        test_receipt = None
        if args.split == "test":
            dataset_sha = sha256_file(manifest_path)
            model_sha = selected[0]["sha256"]
            test_receipt = (
                Path(args.test_receipt_dir) / f"{dataset_sha}-{model_sha}.json"
            )
            if test_receipt.exists():
                raise FileExistsError(
                    "this frozen dataset/model pair has already consumed its test run: "
                    f"{test_receipt}"
                )
        root = immutable_run_dir(args.out, "four-model-zero-shot")
        results = {}
        for index, model in enumerate(selected):
            results[model["id"]] = run_one(
                model,
                output_dir=root / model["id"],
                gold_cases=cases,
                enterprise_path=args.enterprise,
                bare_path=args.bare,
                tokenizer_path=args.tokenizer,
                port=args.port + index,
            )
        summary = {
            "schema_version": 1,
            "run_root": str(root),
            "models": list(results),
            "gold_case_ids": [case.id for case in cases],
            "split": args.split,
            "dataset_manifest": manifest_path,
            "dataset_manifest_sha256": (
                sha256_file(manifest_path) if manifest_path else None
            ),
            "enterprise_label": BASELINE_LABEL,
            "enterprise_path": args.enterprise,
            "enterprise_sha256": sha256_file(args.enterprise),
            "bare_path": args.bare,
            "bare_sha256": sha256_file(args.bare),
            "configuration_sha256": sha256_file(args.config),
            "git_commit": git_commit(),
            "environment": environment_fingerprint(),
            "command_arguments": vars(args),
        }
        write_json(root / "baseline_manifest.json", summary)
        if test_receipt is not None:
            write_json(test_receipt, {
                "schema_version": 1,
                "consumed": True,
                "dataset_manifest_sha256": sha256_file(manifest_path),
                "model_id": selected[0]["id"],
                "model_sha256": selected[0]["sha256"],
                "run_manifest": str(root / "baseline_manifest.json"),
                "run_manifest_sha256": sha256_file(root / "baseline_manifest.json"),
                "git_commit": git_commit(),
            })
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
