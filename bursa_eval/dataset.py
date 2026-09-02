import argparse
import glob
import json
import os
import math
from pathlib import Path
from collections import defaultdict
from bursa_eval.synth.seeds import stable_seed
from bursa import normalize
from bursa_eval.repro import (
    canonical_sha256,
    environment_fingerprint,
    git_commit,
    require_clean,
    sha256_file,
    write_json,
)

CALIBRATION_MIX = {
    "synth_sibling_split": 1.0,
    "synth_overpayment": 1.0,
    "synth_pidgin_ambiguous": 1.0,
    "synth_no_candidate": 1.0,
}
RELEASE_TRAINING_MIX = {
    # Duplicate references are stopped before model inference, so they do not
    # have a truthful constrained-app or bare-model training target.
    "synth_exact_id": 1.0,
    "synth_sibling_split": 1.0,
    "synth_overpayment": 1.0,
    # This weight yields roughly 28% true abstentions across the two
    # abstaining and three allocating template families.
    "synth_pidgin_ambiguous": 7 / 12,
    "synth_no_candidate": 7 / 12,
}


def near_dup_signature(case) -> str:
    toks = " ".join(sorted(normalize.narration_tokens(case.transaction.narration)))
    names = " ".join(sorted(normalize.normalize_name(s.name) for s in case.setup.students))
    amt = case.transaction.amount_naira
    bucket = (amt // 10000) if isinstance(amt, int) else 0
    return f"{case.scenario_family}|{toks}|{bucket}|{names}"


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _components(cases):
    uf = _UF()
    by_gfam, by_tfam, by_sig = defaultdict(list), defaultdict(list), defaultdict(list)
    for c in cases:
        uf.find(c.id)
        by_gfam[c.guardian_family].append(c.id)
        by_tfam[c.template_family].append(c.id)
        by_sig[near_dup_signature(c)].append(c.id)
    for group in list(by_gfam.values()) + list(by_tfam.values()) + list(by_sig.values()):
        for other in group[1:]:
            uf.union(group[0], other)
    comps = defaultdict(list)
    for c in cases:
        comps[uf.find(c.id)].append(c)
    return list(comps.values())


def split(cases, base_seed, targets=(0.7, 0.15, 0.15), pinned=None):
    pinned = pinned or {}
    pinned_ids = set(pinned.get("val", [])) | set(pinned.get("test", []))
    comps = _components(cases)
    gold_total = sum(1 for c in cases if c.provenance == "team_authored")
    assign = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    def comp_pin(comp):
        splits = {s for c in comp for s in ("val", "test") if c.id in set(pinned.get(s, []))}
        if len(splits) > 1:
            raise ValueError(f"component straddles pinned splits: {[c.id for c in comp]}")
        return next(iter(splits)) if splits else None

    def is_synth(comp):
        return any(c.provenance == "synthetic" for c in comp)

    for comp in comps:
        pin = comp_pin(comp)
        if pin:
            intruders = [c.id for c in comp if c.id not in pinned_ids]
            if is_synth(comp) or intruders:
                raise ValueError(f"new edge pulls pinned {pin} component toward train: {intruders}")
            assign[pin].extend(c.id for c in comp)
            counts[pin] += sum(1 for c in comp if c.provenance == "team_authored")

    remaining = [comp for comp in comps if comp_pin(comp) is None]
    synth_comps = [c for c in remaining if is_synth(c)]
    gold_comps = sorted([c for c in remaining if not is_synth(c)],
                        key=lambda comp: (-len(comp), comp[0].id))
    for comp in synth_comps:              # synthetic is training-only
        assign["train"].extend(c.id for c in comp)
    tgt = {"train": targets[0], "val": targets[1], "test": targets[2]}
    for comp in gold_comps:
        size = len(comp)
        if gold_total and size > 0.10 * gold_total:
            print(f"WARN: component {comp[0].id} is {size}/{gold_total} (>10% of gold)")
        best = min(("train", "val", "test"),
                   key=lambda s: (counts[s] - tgt[s] * gold_total, stable_seed(base_seed, s, size)))
        assign[best].extend(c.id for c in comp)
        counts[best] += size
    return assign


def _coverage(cases, assignment):
    report = {}
    for split_name, ids in assignment.items():
        cell = {}
        for cid in ids:
            c = cases[cid]
            cell.setdefault(c.scenario_family, {}).setdefault(c.language, 0)
            cell[c.scenario_family][c.language] += 1
        report[split_name] = cell
    for split_name in ("val", "test"):
        if not report.get(split_name):
            print(f"WARN: {split_name} split is empty — steer authoring before the freeze")
    return report


def build(base_seed, n_synth, gold_dir="data/gold", out_dir="data/build", pinned=None):
    """Assemble gold + synthetic, split, and write {train,val,test}.jsonl (D6 dual-format).
    Does NOT freeze — call freeze() explicitly when gold authoring completes (pre-Phase-3)."""
    from bursa_eval.goldcheck import load_case
    from bursa_eval.synth.generate import generate
    from bursa_eval.synth import render
    gold = [load_case(p) for p in sorted(glob.glob(f"{gold_dir}/*.yaml"))]
    synth = generate(base_seed=base_seed, n=n_synth, gold=tuple(gold))
    cases = {c.id: c for c in gold + synth}
    assignment = split(gold + synth, base_seed=base_seed, pinned=pinned)
    os.makedirs(out_dir, exist_ok=True)
    for split_name, ids in assignment.items():
        with open(f"{out_dir}/{split_name}.jsonl", "w") as f:
            for cid in ids:
                c = cases[cid]
                for fmt in (render.to_app_format(c), render.to_chat_format(c)):
                    if fmt is not None:
                        f.write(json.dumps({"id": cid, "provenance": c.provenance, **fmt}) + "\n")
    return {"assignment": assignment, "gold": len(gold), "synth": len(synth),
            "coverage": _coverage(cases, assignment)}


def freeze(assignment, path="data/manifest.json", **meta):
    """Freeze the split: pin val AND test case ids (immovable). Trigger: gold authoring
    complete, immediately before Phase 3 training."""
    manifest = {"frozen": True, "val_case_ids": sorted(assignment["val"]),
                "test_case_ids": sorted(assignment["test"]), **meta}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_manifest(path="data/manifest.json"):
    with open(path) as f:
        return json.load(f)


def _render_rows(cases, assignment, out_dir):
    from bursa_eval.synth import render

    rows_by_split = {}
    for split_name, ids in assignment.items():
        rows = []
        for case_id in ids:
            case = cases[case_id]
            for format_name, rendered in (
                ("app", render.to_app_format(case)),
                ("chat", render.to_chat_format(case)),
            ):
                if rendered is not None:
                    rows.append({
                        "id": f"{case_id}:{format_name}",
                        "case_id": case_id,
                        "format": format_name,
                        "language": case.language,
                        "scenario_family": case.scenario_family,
                        "provenance": case.provenance,
                        **rendered,
                    })
        rows_by_split[split_name] = rows
    return rows_by_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _leakage_errors(cases, assignment) -> list[str]:
    owner = {
        case_id: split_name
        for split_name, ids in assignment.items()
        for case_id in ids
    }
    errors = []
    for component in _components(cases):
        splits = {owner.get(case.id) for case in component if owner.get(case.id)}
        if len(splits) > 1:
            errors.append(
                f"component spans {sorted(splits)}: {sorted(case.id for case in component)}"
            )
    return errors


def build_release(
    *,
    base_seed: int,
    n_synth: int = 4000,
    n_calibration: int = 600,
    gold_dir: str = "data/gold",
    out_dir: str = "data/build",
    pinned: dict | None = None,
    oasst_path: str,
    oasst_revision: str,
    general_fraction: float = 0.30,
    require_final_gold: bool = True,
) -> dict:
    """Build the locked dual-format 70/30 corpus and isolated calibration cases."""
    from bursa_eval.authoring import validate_authored_set
    from bursa_eval.general import load_oasst_pairs, project_owned_examples
    from bursa_eval.goldcheck import load_case
    from bursa_eval.synth.generate import generate

    target = Path(out_dir)
    if general_fraction not in (0.30, 0.40):
        raise ValueError("general_fraction must be the locked 0.30 or one-time retry 0.40")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty; build outputs are immutable")
    target.mkdir(parents=True, exist_ok=True)

    gold = [load_case(path) for path in sorted(Path(gold_dir).glob("*.yaml"))]
    gold_report = validate_authored_set(gold, require_final=require_final_gold)
    if not gold_report["valid"]:
        raise ValueError("authored-set gate failed: " + "; ".join(gold_report["errors"]))

    synthetic = generate(
        base_seed=base_seed,
        n=n_synth,
        mix=RELEASE_TRAINING_MIX,
        gold=tuple(gold),
    )
    synthetic_generation = {
        "attempts": generate.last_attempts,
        "dropped_near_duplicate_or_invalid": generate.last_attempts - len(synthetic),
        "drop_rate": generate.last_drop_rate,
    }
    calibration = generate(
        base_seed=base_seed + 1,
        n=n_calibration,
        mix=CALIBRATION_MIX,
        gold=tuple(gold + synthetic),
    )
    calibration_generation = {
        "attempts": generate.last_attempts,
        "dropped_near_duplicate_or_invalid": generate.last_attempts - len(calibration),
        "drop_rate": generate.last_drop_rate,
    }
    for index, case in enumerate(calibration):
        case.id = f"calibration-{index:04d}-{case.id}"
        case.guardian_family = f"calibration-{case.guardian_family}"
        case.template_family = f"calibration-{case.template_family}"

    all_cases = {case.id: case for case in gold + synthetic}
    assignment = split(gold + synthetic, base_seed=base_seed, pinned=pinned)
    leakage = _leakage_errors(gold + synthetic, assignment)
    if leakage:
        raise ValueError("split leakage: " + "; ".join(leakage))

    rows = _render_rows(all_cases, assignment, target)
    rendered_synthetic = {
        row["case_id"]
        for row in rows["train"]
        if row["provenance"] == "synthetic"
    }
    if rendered_synthetic != {case.id for case in synthetic}:
        missing = sorted({case.id for case in synthetic} - rendered_synthetic)
        raise ValueError(
            f"{len(missing)} generated cases lack a required rendered format"
        )
    synthetic_row_counts = {
        case.id: sum(
            row["case_id"] == case.id
            for row in rows["train"]
        )
        for case in synthetic
    }
    if any(count != 2 for count in synthetic_row_counts.values()):
        raise ValueError("every generated reconciliation case must have app and chat rows")
    reconciliation_train_rows = len(rows["train"])
    general_count = round(
        reconciliation_train_rows * general_fraction / (1 - general_fraction)
    )
    owned_count = math.ceil(general_count / 2)
    oasst_count = general_count - owned_count
    general_rows = project_owned_examples(owned_count, seed=base_seed)
    general_rows.extend(
        load_oasst_pairs(
            oasst_path,
            oasst_count,
            seed=base_seed,
            revision=oasst_revision,
        )
    )
    rows["train"].extend(general_rows)

    for split_name, split_rows in rows.items():
        _write_jsonl(target / f"{split_name}.jsonl", split_rows)
    fit_count = int(n_calibration * 2 / 3)
    calibration_rows = [
        {
            "case_id": case.id,
            "split": "fit" if index < fit_count else "threshold",
            "case": case.model_dump(mode="json", by_alias=True),
        }
        for index, case in enumerate(calibration)
    ]
    _write_jsonl(target / "calibration_cases.jsonl", calibration_rows)

    file_hashes = {
        path.name: sha256_file(path)
        for path in sorted(target.glob("*.jsonl"))
    }
    manifest = {
        "schema_version": 1,
        "frozen": False,
        "base_seed": base_seed,
        "regeneration_command": (
            f"bursa-data build --seed {base_seed} --n-synth {n_synth} "
            f"--n-calibration {n_calibration} --gold-dir {gold_dir} "
            f"--out {out_dir} --oasst {oasst_path} "
            f"--oasst-revision {oasst_revision} "
            f"--general-fraction {general_fraction}"
        ),
        "assignment": {key: sorted(value) for key, value in assignment.items()},
        "authored": gold_report,
        "synthetic_case_count": n_synth,
        "synthetic_rendered_row_count": 2 * n_synth,
        "synthetic_training_mix": RELEASE_TRAINING_MIX,
        "synthetic_language_counts": {
            language: sum(case.language == language for case in synthetic)
            for language in ("en", "pcm", "yo")
        },
        "synthetic_abstention_count": sum(
            case.is_abstention() for case in synthetic
        ),
        "calibration_case_count": n_calibration,
        "calibration_fit_count": fit_count,
        "calibration_threshold_count": n_calibration - fit_count,
        "calibration_language_counts": {
            language: sum(case.language == language for case in calibration)
            for language in ("en", "pcm", "yo")
        },
        "calibration_abstention_count": sum(
            case.is_abstention() for case in calibration
        ),
        "duplicate_filter_report": {
            "synthetic": synthetic_generation,
            "calibration": calibration_generation,
            "split_leakage_errors": leakage,
        },
        "reconciliation_train_rows": reconciliation_train_rows,
        "general_train_rows": general_count,
        "mixture": {
            "reconciliation_fraction": (
                reconciliation_train_rows
                / (reconciliation_train_rows + general_count)
            ),
            "project_owned_rows": owned_count,
            "oasst1_rows": oasst_count,
            "general_fraction_target": general_fraction,
        },
        "oasst1": {
            "dataset": "OpenAssistant/oasst1",
            "revision": oasst_revision,
            "license": "Apache-2.0",
            "source_sha256": sha256_file(oasst_path),
            "selected_ids": [
                row["id"] for row in general_rows if row["provenance"] == "OpenAssistant/oasst1"
            ],
            "filters": {
                "roles": ["prompter", "assistant"],
                "language": "en",
                "assistant_rank": [0],
                "deleted": False,
                "prompt_char_range": [20, 1500],
                "completion_char_range": [20, 2000],
                "deduplication": "sha256-normalized-prompt-plus-completion",
            },
        },
        "split_leakage_errors": leakage,
        "files": file_hashes,
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    write_json(target / "build_manifest.json", manifest)
    return manifest


def freeze_build(
    build_manifest_path: str | Path,
    *,
    output: str | Path = "data/manifest.json",
) -> dict:
    source = json.loads(Path(build_manifest_path).read_text(encoding="utf-8"))
    if source.get("frozen"):
        raise ValueError("build manifest is already frozen")
    assignment = source["assignment"]
    manifest = {
        **source,
        "frozen": True,
        "val_case_ids": sorted(assignment["val"]),
        "test_case_ids": sorted(assignment["test"]),
        "source_build_manifest_sha256": sha256_file(build_manifest_path),
    }
    manifest["manifest_content_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    )
    write_json(output, manifest)
    return manifest


def _load_pin(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _load_release_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported data configuration schema")
    required = {
        "seed",
        "authored_cases",
        "languages",
        "abstentions",
        "synthetic_cases",
        "calibration_cases",
        "calibration_fit_cases",
        "calibration_threshold_cases",
        "general_fraction",
        "allowed_forgetting_retry_fraction",
        "oasst1_revision",
    }
    if not required <= set(config):
        raise ValueError("data configuration is incomplete")
    if (
        config["calibration_fit_cases"]
        + config["calibration_threshold_cases"]
        != config["calibration_cases"]
    ):
        raise ValueError("calibration split counts do not sum to the corpus size")
    return config


def _validate_release_arguments(args, config: dict) -> None:
    expected = {
        "seed": config["seed"],
        "n_synth": config["synthetic_cases"],
        "n_calibration": config["calibration_cases"],
        "oasst_revision": config["oasst1_revision"],
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(
                f"{name}={getattr(args, name)!r} does not match data config {value!r}"
            )
    if args.general_fraction not in (
        config["general_fraction"],
        config["allowed_forgetting_retry_fraction"],
    ):
        raise ValueError("general fraction is not authorized by the data config")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bursa reproducible data workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    drafts = sub.add_parser("author-drafts")
    drafts.add_argument("--gold-dir", default="data/gold")
    drafts.add_argument("--draft-dir", default="data/gold_drafts")
    drafts.add_argument("--seed", type=int, default=3407)

    promote = sub.add_parser("promote")
    promote.add_argument("--draft", required=True)
    promote.add_argument("--gold-dir", default="data/gold")
    promote.add_argument("--reviewer", required=True)
    promote.add_argument("--reviewed-at")

    validate = sub.add_parser("validate")
    validate.add_argument("--config", default="configs/data/release.json")
    validate.add_argument("--gold-dir", default="data/gold")
    validate.add_argument("--require-final", action="store_true")

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--config", default="configs/data/release.json")
    build_parser.add_argument("--seed", type=int, default=3407)
    build_parser.add_argument("--n-synth", type=int, default=4000)
    build_parser.add_argument("--n-calibration", type=int, default=600)
    build_parser.add_argument("--gold-dir", default="data/gold")
    build_parser.add_argument("--out", default="data/build")
    build_parser.add_argument("--pinned")
    build_parser.add_argument("--oasst", required=True)
    build_parser.add_argument("--oasst-revision", required=True)
    build_parser.add_argument(
        "--general-fraction", type=float, choices=(0.30, 0.40), default=0.30
    )
    build_parser.add_argument("--allow-incomplete-gold", action="store_true")
    build_parser.add_argument("--allow-dirty", action="store_true")

    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--config", default="configs/data/release.json")
    freeze_parser.add_argument("--build-manifest", required=True)
    freeze_parser.add_argument("--output", default="data/manifest.json")
    freeze_parser.add_argument("--allow-dirty", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "author-drafts":
            from bursa_eval.authoring import build_drafts, write_drafts
            paths = write_drafts(
                build_drafts(args.gold_dir, seed=args.seed), args.draft_dir
            )
            print(json.dumps({"drafts": len(paths), "directory": args.draft_dir}))
            return 0
        if args.command == "promote":
            from bursa_eval.authoring import promote_draft
            path = promote_draft(
                args.draft,
                args.gold_dir,
                reviewer=args.reviewer,
                reviewed_at=args.reviewed_at,
            )
            print(path)
            return 0
        if args.command == "validate":
            from bursa_eval.authoring import validate_authored_set
            from bursa_eval.goldcheck import load_case
            cases = [
                load_case(path) for path in sorted(Path(args.gold_dir).glob("*.yaml"))
            ]
            report = validate_authored_set(cases, require_final=args.require_final)
            config = _load_release_config(args.config)
            report["configuration_sha256"] = sha256_file(args.config)
            if args.require_final:
                locked = {
                    "total": config["authored_cases"],
                    "languages": config["languages"],
                    "abstentions": config["abstentions"],
                }
                for key, expected in locked.items():
                    if report[key] != expected:
                        report["errors"].append(
                            f"{key} does not match configured value {expected}"
                        )
                report["valid"] = not report["errors"]
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["valid"] else 1
        if args.command == "build":
            require_clean(allow_dirty=args.allow_dirty)
            release_config = _load_release_config(args.config)
            _validate_release_arguments(args, release_config)
            manifest = build_release(
                base_seed=args.seed,
                n_synth=args.n_synth,
                n_calibration=args.n_calibration,
                gold_dir=args.gold_dir,
                out_dir=args.out,
                pinned=_load_pin(args.pinned),
                oasst_path=args.oasst,
                oasst_revision=args.oasst_revision,
                general_fraction=args.general_fraction,
                require_final_gold=not args.allow_incomplete_gold,
            )
            manifest["configuration_sha256"] = sha256_file(args.config)
            manifest["command_arguments"] = vars(args)
            manifest["git_commit"] = git_commit()
            manifest["environment"] = environment_fingerprint()
            manifest["manifest_content_sha256"] = canonical_sha256({
                key: value
                for key, value in manifest.items()
                if key != "manifest_content_sha256"
            })
            write_json(Path(args.out) / "build_manifest.json", manifest)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        require_clean(allow_dirty=args.allow_dirty)
        release_config = _load_release_config(args.config)
        source = json.loads(
            Path(args.build_manifest).read_text(encoding="utf-8")
        )
        if (
            source.get("synthetic_case_count") != release_config["synthetic_cases"]
            or source.get("calibration_case_count")
            != release_config["calibration_cases"]
            or source.get("calibration_fit_count")
            != release_config["calibration_fit_cases"]
            or source.get("calibration_threshold_count")
            != release_config["calibration_threshold_cases"]
            or source.get("oasst1", {}).get("revision")
            != release_config["oasst1_revision"]
            or source.get("mixture", {}).get("general_fraction_target")
            not in (
                release_config["general_fraction"],
                release_config["allowed_forgetting_retry_fraction"],
            )
        ):
            raise ValueError("build manifest does not match the data configuration")
        manifest = freeze_build(args.build_manifest, output=args.output)
        manifest["configuration_sha256"] = sha256_file(args.config)
        manifest["command_arguments"] = vars(args)
        manifest["git_commit"] = git_commit()
        manifest["environment"] = environment_fingerprint()
        manifest["manifest_content_sha256"] = canonical_sha256({
            key: value
            for key, value in manifest.items()
            if key != "manifest_content_sha256"
        })
        write_json(args.output, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
