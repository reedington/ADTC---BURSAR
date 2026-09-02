"""Pinned Unsloth QLoRA, merge, and GGUF export workflow."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

from bursa.inference.backend import LlamaServerBackend
from bursa.inference.server import LlamaServer
from bursa_eval.repro import (
    environment_fingerprint,
    git_commit,
    immutable_run_dir,
    require_clean,
    require_disk,
    sha256_file,
    write_json,
)

BASELINE_MODEL_IDS = {
    "qwen3-0.6b": "qwen3-0.6b-q4_k_m",
    "qwen3-1.7b": "qwen3-1.7b-q4_k_m",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported training configuration schema")
    if config.get("max_seq_length") != 2048:
        raise ValueError("training max_seq_length must remain 2048")
    lora = config["lora"]
    if lora["r"] != 16 or lora["alpha"] != 32:
        raise ValueError("locked LoRA parameters are r=16, alpha=32")
    trainer = config["trainer"]
    if (
        trainer["micro_batch_size"] != 4
        or trainer["gradient_accumulation_steps"] != 8
        or trainer["epochs"] != 2
        or trainer["packing"] is not False
    ):
        raise ValueError("training batch/epoch/packing configuration changed")
    return config


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_dataset(
    train_path: str | Path,
    validation_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("frozen"):
        raise ValueError("training requires a frozen dataset manifest")
    expected = manifest.get("files", {})
    for path in (train_path, validation_path):
        name = Path(path).name
        if name not in expected or sha256_file(path) != expected[name]:
            raise ValueError(f"{name} does not match the frozen dataset manifest")
    train = _load_jsonl(train_path)
    validation = _load_jsonl(validation_path)
    if not train or not validation:
        raise ValueError("training and validation data must both be non-empty")
    formats = {row.get("format") for row in train}
    if not {"app", "chat"} <= formats:
        raise ValueError("training data must include app and conversational formats")
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "formats": sorted(str(value) for value in formats),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "general_fraction": manifest.get("mixture", {}).get(
            "general_fraction_target"
        ),
    }


def render_training_text(row: dict, tokenizer) -> str:
    prompt = str(row["prompt"])
    completion = str(row["completion"])
    if row.get("format") == "app":
        if not prompt.endswith("<|im_start|>assistant"):
            raise ValueError(f"{row.get('id')}: app prompt is not the production Qwen envelope")
        return prompt + "\n" + completion + "<|im_end|>"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _gpu_preflight(config: dict) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("training is supported only in the pinned Linux CUDA container")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch/CUDA training dependencies are not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / (1024 ** 3)
    if memory_gib < float(config["minimum_gpu_memory_gib"]):
        raise RuntimeError(
            f"GPU has {memory_gib:.1f} GiB; "
            f"{config['minimum_gpu_memory_gib']} GiB required"
        )
    return {
        "name": properties.name,
        "memory_gib": memory_gib,
        "cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _pip_freeze() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def _artifact_hashes(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            result[str(path.relative_to(root))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def semantic_app_output(text: str) -> dict | None:
    """Reduce a smoke output to the financially meaningful app decision."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        allocations = value.get("candidate_allocations")
        action = value.get("recommended_action")
        if not isinstance(allocations, list) or not isinstance(action, str):
            continue
        try:
            normalized = sorted(
                (
                    str(item["student_id"]),
                    int(item["amount_minor"]),
                )
                for item in allocations
            )
        except (KeyError, TypeError, ValueError):
            continue
        return {
            "candidate_allocations": normalized,
            "recommended_action": action,
        }
    return None


def run_merged_gguf_smoke(
    merged_path: str | Path,
    gguf_path: str | Path,
    validation_path: str | Path,
    *,
    port: int = 18095,
    cases: int = 5,
) -> dict:
    """Compare fixed app decisions from the merged checkpoint and Q4 GGUF."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [
        row
        for row in _load_jsonl(validation_path)
        if row.get("format") == "app"
    ][:cases]
    if len(rows) != cases:
        raise ValueError(f"smoke set requires {cases} app validation rows")
    tokenizer = AutoTokenizer.from_pretrained(
        merged_path, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        local_files_only=True,
        device_map="auto",
        dtype="auto",
    )
    model.eval()
    server = LlamaServer(str(gguf_path), port=port, threads=4, ctx=2048)
    server.start()
    if not server.wait_until_ready():
        server.stop()
        raise RuntimeError("GGUF smoke server did not become ready")
    backend = LlamaServerBackend(f"http://127.0.0.1:{port}", timeout=120)
    records = []
    try:
        for row in rows:
            encoded = tokenizer(row["prompt"], return_tensors="pt")
            encoded = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            generated = model.generate(
                **encoded,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0][encoded["input_ids"].shape[1]:]
            merged_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            gguf_text = backend.completion(
                row["prompt"], max_tokens=512
            )
            expected = semantic_app_output(row["completion"])
            merged_semantic = semantic_app_output(merged_text)
            gguf_semantic = semantic_app_output(gguf_text)
            records.append({
                "case_id": row["case_id"],
                "expected": expected,
                "merged": merged_semantic,
                "gguf": gguf_semantic,
                "merged_raw": merged_text,
                "gguf_raw": gguf_text,
                "equivalent": (
                    expected is not None
                    and merged_semantic == expected
                    and gguf_semantic == expected
                ),
            })
    finally:
        server.stop()
    return {
        "schema_version": 1,
        "cases": len(records),
        "passed": bool(records) and all(row["equivalent"] for row in records),
        "records": records,
    }


def _train_one(
    model_id: str,
    model_config: dict,
    config: dict,
    train_rows: list[dict],
    validation_rows: list[dict],
    output: Path,
    gpu: dict,
) -> dict:
    # Imports remain lazy so local CPU machines can validate configs and datasets.
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    max_length = config["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["source"],
        revision=model_config["revision"],
        max_seq_length=max_length,
        load_in_4bit=True,
    )
    if not tokenizer.chat_template:
        raise RuntimeError(f"{model_id}: tokenizer has no embedded chat template")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing=config["lora"]["gradient_checkpointing"],
        random_state=config["seed"],
        use_rslora=False,
        loftq_config=None,
    )
    train = Dataset.from_list([
        {"text": render_training_text(row, tokenizer)} for row in train_rows
    ])
    validation = Dataset.from_list([
        {"text": render_training_text(row, tokenizer)} for row in validation_rows
    ])
    trainer_config = config["trainer"]
    checkpoints = output / "checkpoints"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=validation,
        dataset_text_field="text",
        max_seq_length=max_length,
        packing=False,
        args=SFTConfig(
            output_dir=str(checkpoints),
            per_device_train_batch_size=trainer_config["micro_batch_size"],
            gradient_accumulation_steps=trainer_config[
                "gradient_accumulation_steps"
            ],
            num_train_epochs=trainer_config["epochs"],
            learning_rate=trainer_config["learning_rate"],
            lr_scheduler_type=trainer_config["lr_scheduler_type"],
            warmup_ratio=trainer_config["warmup_ratio"],
            weight_decay=trainer_config["weight_decay"],
            optim=trainer_config["optim"],
            save_steps=trainer_config["save_steps"],
            eval_steps=trainer_config["eval_steps"],
            eval_strategy="steps",
            logging_steps=trainer_config["logging_steps"],
            seed=config["seed"],
            bf16=gpu["bf16_supported"],
            fp16=not gpu["bf16_supported"],
            report_to="none",
        ),
    )
    result = trainer.train()
    adapter = output / "adapter"
    merged = output / "merged-16bit"
    gguf_f16 = output / "gguf-f16"
    gguf_q4 = output / "gguf-q4_k_m"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    model.save_pretrained_merged(
        str(merged), tokenizer, save_method="merged_16bit"
    )
    model.save_pretrained_gguf(
        str(gguf_f16), tokenizer, quantization_method="f16"
    )
    model.save_pretrained_gguf(
        str(gguf_q4), tokenizer, quantization_method="q4_k_m"
    )
    metrics = dict(result.metrics)
    write_json(output / "training_metrics.json", metrics)
    (output / "chat_template.txt").write_text(
        tokenizer.chat_template, encoding="utf-8"
    )
    return {
        "model_id": model_id,
        "source": model_config["source"],
        "source_revision": model_config["revision"],
        "metrics": metrics,
        "exports": {
            "adapter": str(adapter),
            "merged_16bit": str(merged),
            "gguf_f16": str(gguf_f16),
            "gguf_q4_k_m": str(gguf_q4),
            "q5_k_m": "not_generated_pending_q4_accuracy_tripwire",
        },
    }


def c3_decision(
    zero_shot: dict,
    candidate: dict,
    stress: dict,
    *,
    zero_bare_score: float,
    candidate_bare_score: float,
    merged: dict | None = None,
) -> dict:
    zero_gold = zero_shot["bursa_gold"]["exact_allocation_accuracy"]
    candidate_gold = candidate["bursa_gold"]["exact_allocation_accuracy"]
    zero_proxy = zero_shot["adtc"]["accuracy"]
    candidate_proxy = candidate["adtc"]["accuracy"]
    if None in (zero_gold, candidate_gold, zero_proxy, candidate_proxy):
        raise ValueError("C3 requires complete gold and enterprise proxy metrics")
    proxy_regression_points = 100 * (zero_proxy - candidate_proxy)
    bare_regression_points = zero_bare_score - candidate_bare_score
    q4_loss_points = None
    if merged is not None:
        merged_gold = merged["bursa_gold"]["exact_allocation_accuracy"]
        if merged_gold is None:
            raise ValueError("merged scorecard lacks exact-allocation accuracy")
        q4_loss_points = 100 * (merged_gold - candidate_gold)
    gates = {
        "gold_improved": candidate_gold > zero_gold,
        "enterprise_regression_le_2": proxy_regression_points <= 2,
        "bare_regression_le_5": bare_regression_points <= 5,
        "valid_json_ge_99_5": stress.get("valid_json_rate", 0) >= .995,
        "unsupported_ids_zero": stress.get("unsupported_ids") == 0,
        "thinking_leaks_zero": stress.get("thinking_leaks") == 0,
        "incorrect_auto_posts_zero": stress.get("incorrect_auto_posts") == 0,
        "stress_passed": stress.get("passes") is True,
        "scorecard_safety_gates": not candidate.get("gates_failed"),
        "q4_vs_merged_within_3": (
            q4_loss_points is None or q4_loss_points <= 3
        ),
    }
    fine_tuned_passes = all(gates.values())
    q5_required = q4_loss_points is not None and q4_loss_points > 3
    return {
        "schema_version": 1,
        "checkpoint": "C3",
        "ship": (
            "q5_required"
            if q5_required
            else ("fine_tuned" if fine_tuned_passes else "zero_shot")
        ),
        "fine_tuned_passes": fine_tuned_passes,
        "gates": gates,
        "deltas_points": {
            "gold_exact_allocation": 100 * (candidate_gold - zero_gold),
            "enterprise_proxy_regression": proxy_regression_points,
            "bare_regression": bare_regression_points,
            "q4_vs_merged_loss": q4_loss_points,
        },
        "retry_60_40": proxy_regression_points > 2,
        "q5_required": q5_required,
        "frozen_test_status": "not_run",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/training/qwen3-lora.toml")
    run.add_argument("--model", choices=("qwen3-0.6b", "qwen3-1.7b", "all"), default="all")
    run.add_argument("--train", default="data/build/train.jsonl")
    run.add_argument("--validation", default="data/build/val.jsonl")
    run.add_argument("--dataset-manifest", default="data/manifest.json")
    run.add_argument("--c2-decision", required=True)
    run.add_argument("--out", default="artifacts/training")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--preflight-only", action="store_true")
    run.add_argument(
        "--attempt", choices=("primary", "60-40-retry"), default="primary"
    )
    run.add_argument("--retry-decision")
    run.add_argument(
        "--retry-receipt",
        default="artifacts/training/60-40-retry-receipt.json",
    )

    compare = sub.add_parser("compare")
    compare.add_argument("--zero-shot", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--merged", required=True)
    compare.add_argument("--stress", required=True)
    compare.add_argument("--zero-bare-score", type=float, required=True)
    compare.add_argument("--candidate-bare-score", type=float, required=True)
    compare.add_argument("--c2-decision", required=True)
    compare.add_argument(
        "--model", choices=tuple(BASELINE_MODEL_IDS), required=True
    )
    compare.add_argument("--output", required=True)
    compare.add_argument("--allow-dirty", action="store_true")
    compare.add_argument(
        "--attempt", choices=("primary", "60-40-retry"), default="primary"
    )

    q5 = sub.add_parser("quantize-q5")
    q5.add_argument("--f16-gguf", required=True)
    q5.add_argument("--c3-decision", required=True)
    q5.add_argument("--output", required=True)
    q5.add_argument("--allow-dirty", action="store_true")

    smoke = sub.add_parser("smoke-compare")
    smoke.add_argument("--config", default="configs/training/qwen3-lora.toml")
    smoke.add_argument("--merged", required=True)
    smoke.add_argument("--gguf", required=True)
    smoke.add_argument("--validation", default="data/build/val.jsonl")
    smoke.add_argument("--dataset-manifest", default="data/manifest.json")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--port", type=int, default=18095)
    smoke.add_argument("--cases", type=int, default=5)
    smoke.add_argument("--allow-dirty", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "smoke-compare":
            load_config(args.config)
            require_clean(allow_dirty=args.allow_dirty)
            manifest = json.loads(
                Path(args.dataset_manifest).read_text(encoding="utf-8")
            )
            expected_hash = manifest.get("files", {}).get(
                Path(args.validation).name
            )
            if (
                not manifest.get("frozen")
                or expected_hash != sha256_file(args.validation)
            ):
                raise ValueError(
                    "smoke validation data does not match the frozen manifest"
                )
            output = Path(args.output)
            if output.exists():
                raise FileExistsError(f"smoke output is immutable: {output}")
            result = run_merged_gguf_smoke(
                args.merged,
                args.gguf,
                args.validation,
                port=args.port,
                cases=args.cases,
            )
            result["git_commit"] = git_commit()
            result["environment"] = environment_fingerprint()
            result["command_arguments"] = vars(args)
            result["hashes"] = {
                "gguf": sha256_file(args.gguf),
                "validation": sha256_file(args.validation),
                "dataset_manifest": sha256_file(args.dataset_manifest),
            }
            write_json(output, result)
            print(json.dumps({
                "cases": result["cases"],
                "passed": result["passed"],
                "output": str(output),
            }, indent=2))
            return 0 if result["passed"] else 1
        if args.command == "quantize-q5":
            require_clean(allow_dirty=args.allow_dirty)
            decision = json.loads(
                Path(args.c3_decision).read_text(encoding="utf-8")
            )
            if decision.get("q5_required") is not True:
                raise ValueError("Q5_K_M tripwire did not fire")
            output = Path(args.output)
            manifest_output = output.with_suffix(output.suffix + ".manifest.json")
            if output.exists() or manifest_output.exists():
                raise FileExistsError(f"Q5 artifact is immutable: {output}")
            result = subprocess.run(
                [
                    "llama-quantize",
                    args.f16_gguf,
                    str(output),
                    "Q5_K_M",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"llama-quantize failed: {result.stderr.strip()}"
                )
            write_json(manifest_output, {
                "schema_version": 1,
                "git_commit": git_commit(),
                "environment": environment_fingerprint(),
                "command_arguments": vars(args),
                "source_f16_sha256": sha256_file(args.f16_gguf),
                "c3_decision_sha256": sha256_file(args.c3_decision),
                "quantization": "Q5_K_M",
                "output_sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
            })
            print(json.dumps({
                "output": str(output),
                "sha256": sha256_file(output),
            }))
            return 0
        if args.command == "compare":
            require_clean(allow_dirty=args.allow_dirty)
            c2 = json.loads(Path(args.c2_decision).read_text(encoding="utf-8"))
            if c2.get("checkpoint") != "C2":
                raise ValueError("a valid C2 decision is required")
            if c2.get("selected_model_id") != BASELINE_MODEL_IDS[args.model]:
                raise ValueError(
                    "the non-selected Qwen family is evidence-only and cannot enter C3"
                )
            result = c3_decision(
                json.loads(Path(args.zero_shot).read_text(encoding="utf-8")),
                json.loads(Path(args.candidate).read_text(encoding="utf-8")),
                json.loads(Path(args.stress).read_text(encoding="utf-8")),
                zero_bare_score=args.zero_bare_score,
                candidate_bare_score=args.candidate_bare_score,
                merged=(
                    json.loads(Path(args.merged).read_text(encoding="utf-8"))
                    if args.merged
                    else None
                ),
            )
            result["c2_selected_model_id"] = c2["selected_model_id"]
            result["evaluated_training_model"] = args.model
            result["attempt"] = args.attempt
            if args.attempt == "60-40-retry":
                result["retry_exhausted"] = True
                result["retry_60_40"] = False
            result["git_commit"] = git_commit()
            result["environment"] = environment_fingerprint()
            result["command_arguments"] = vars(args)
            result["input_hashes"] = {
                "zero_shot": sha256_file(args.zero_shot),
                "candidate": sha256_file(args.candidate),
                "merged": sha256_file(args.merged) if args.merged else None,
                "stress": sha256_file(args.stress),
                "c2_decision": sha256_file(args.c2_decision),
            }
            output = Path(args.output)
            if output.exists():
                raise FileExistsError(f"C3 decision is immutable: {output}")
            write_json(output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["fine_tuned_passes"] else 1
        config = load_config(args.config)
        require_clean(allow_dirty=args.allow_dirty)
        require_disk(".", float(config["minimum_free_gib"]))
        dataset = validate_dataset(
            args.train, args.validation, args.dataset_manifest
        )
        c2 = json.loads(Path(args.c2_decision).read_text(encoding="utf-8"))
        if c2.get("checkpoint") != "C2" or not c2.get("selected_model_id"):
            raise ValueError("a valid C2 freeze decision is required")
        retry_receipt = None
        if args.attempt == "primary":
            if dataset["general_fraction"] != 0.30:
                raise ValueError("primary training requires the frozen 70/30 corpus")
        else:
            if dataset["general_fraction"] != 0.40:
                raise ValueError("the forgetting retry requires the frozen 60/40 corpus")
            if args.model == "all":
                raise ValueError("the forgetting retry may train only the C2-selected family")
            if BASELINE_MODEL_IDS[args.model] != c2["selected_model_id"]:
                raise ValueError("the forgetting retry may not train the evidence-only family")
            if not args.retry_decision:
                raise ValueError("--retry-decision is required for the 60/40 attempt")
            retry_decision = json.loads(
                Path(args.retry_decision).read_text(encoding="utf-8")
            )
            if retry_decision.get("retry_60_40") is not True:
                raise ValueError("C3 did not authorize the 60/40 forgetting retry")
            retry_receipt = Path(args.retry_receipt)
            if retry_receipt.exists():
                raise FileExistsError("the single 60/40 retry has already completed")
        gpu = _gpu_preflight(config)
        preflight = {
            "config": str(args.config),
            "dataset": dataset,
            "c2_selected_model_id": c2["selected_model_id"],
            "gpu": gpu,
            "environment": environment_fingerprint(),
        }
        if args.preflight_only:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0

        root = immutable_run_dir(args.out, "qwen3-lora")
        train_rows = _load_jsonl(args.train)
        validation_rows = _load_jsonl(args.validation)
        model_ids = list(config["models"]) if args.model == "all" else [args.model]
        results = []
        for model_id in model_ids:
            model_output = root / model_id
            model_output.mkdir()
            trained = _train_one(
                model_id,
                config["models"][model_id],
                config,
                train_rows,
                validation_rows,
                model_output,
                gpu,
            )
            trained["shipping_eligible_at_c3"] = (
                BASELINE_MODEL_IDS[model_id] == c2["selected_model_id"]
            )
            results.append(trained)
        manifest = {
            "schema_version": 1,
            "git_commit": git_commit(),
            "command_arguments": vars(args),
            "preflight": preflight,
            "training_config": config,
            "results": results,
            "pip_freeze": _pip_freeze(),
            "verification": {
                "merged_vs_gguf_smoke": "pending",
                "ollama": "pending",
                "lm_studio": "pending",
                "development_scorecards": "pending",
                "stress_1000": "pending",
            },
        }
        write_json(root / "run_manifest.json", manifest)
        manifest["artifacts"] = _artifact_hashes(root)
        write_json(root / "run_manifest.json", manifest)
        if retry_receipt is not None:
            write_json(retry_receipt, {
                "schema_version": 1,
                "completed": True,
                "run_manifest": str(root / "run_manifest.json"),
                "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
                "retry_decision_sha256": sha256_file(args.retry_decision),
                "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
                "model": args.model,
                "git_commit": git_commit(),
            })
        print(json.dumps({"run_root": str(root), "models": model_ids}, indent=2))
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
