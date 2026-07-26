"""One-command scorecard across the three suites. Two hard gates map to the exit code.
The perf section ingests profiler numbers ONLY (--perf); harness timings never populate it."""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from bursa_eval.goldcheck import load_case
from bursa_eval.harness.runner import run_gold_suite
from bursa_eval.harness.metrics import compute_metrics, evaluate_gates, unexercised_gates
from bursa_eval.harness.adtc import load_adtc, score_adtc
from bursa_eval.harness.baremodel import load_bare_prompts, run_bare_suite


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha256(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scorecard(records, adtc_res=None, bare_records=None, provenance=None, perf=None) -> dict:
    metrics = compute_metrics(records)
    return {
        "bursa_gold": metrics,
        "adtc": adtc_res,
        "bare_model": ([asdict(b) for b in bare_records] if bare_records else None),
        "perf": perf,   # profiler-ingested ONLY; harness timings never populate this
        "provenance": provenance or {},
        "gates_failed": evaluate_gates(metrics),
        "gates_not_exercised": unexercised_gates(records),
    }


def write_run(out_dir, records, scorecard) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    with open(os.path.join(out_dir, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, sort_keys=True)


def _load_records(run_dir):
    out = {}
    with open(os.path.join(run_dir, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["case_id"]] = r
    return out


def diff_runs(dir_a, dir_b) -> list[dict]:
    """Per-case regressions (a True metric in A that became False in B)."""
    a, b = _load_records(dir_a), _load_records(dir_b)
    regressed = []
    for cid in sorted(set(a) & set(b)):
        for key in ("top1_hit", "exact_alloc_hit", "correct_action", "dup_blocked", "pool_recall_hit"):
            if a[cid].get(key) is True and b[cid].get(key) is False:
                regressed.append({"case_id": cid, "metric": key, "from": True, "to": False})
    return regressed


def sidebyside(zeroshot_dir, candidate_dir) -> str:
    """Compare TWO run dirs' bare-model outputs for the human rubric."""
    def bare(run_dir):
        with open(os.path.join(run_dir, "scorecard.json"), encoding="utf-8") as f:
            sc = json.load(f)
        return {b["case_id"]: b for b in (sc.get("bare_model") or [])}
    z, c = bare(zeroshot_dir), bare(candidate_dir)
    lines = ["# Bare-model side-by-side",
             "_Score each: correct · coherent · no format leakage · appropriate abstention_\n"]
    for cid in sorted(set(z) & set(c)):
        lines += [f"## {cid}", f"**Prompt:** {z[cid]['prompt']}", "",
                  f"**Zero-shot:** {z[cid]['output']}", "",
                  f"**Candidate:** {c[cid]['output']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fake", "llama"], default="fake")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--label", default="run")
    ap.add_argument("--gold-dir", default="data/gold")
    ap.add_argument("--adtc", default="data/adtc/proxy/sample.jsonl")
    ap.add_argument("--adtc-label", default="proxy")
    ap.add_argument("--bare", default="data/bare/prompts.jsonl")
    ap.add_argument("--perf")
    ap.add_argument("--model-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    ap.add_argument("--sidebyside", nargs=2, metavar=("ZERO", "CAND"))
    args = ap.parse_args(argv)

    if args.diff:
        for row in diff_runs(*args.diff):
            print(json.dumps(row))
        return 0
    if args.sidebyside:
        print(sidebyside(*args.sidebyside))
        return 0

    if args.backend == "fake":
        from bursa.inference.backend import FakeBackend
        backend = FakeBackend(response="{}")
    else:
        from bursa.inference.backend import LlamaServerBackend
        backend = LlamaServerBackend()

    cases = [load_case(p) for p in sorted(glob.glob(os.path.join(args.gold_dir, "*.yaml")))]
    records = run_gold_suite(cases, backend, args.tokenizer)

    adtc_res = None
    if args.adtc and os.path.exists(args.adtc):
        adtc_res = score_adtc(load_adtc(args.adtc), backend, label=args.adtc_label)
    bare_records = None
    if args.bare and os.path.exists(args.bare):
        bare_records = run_bare_suite(load_bare_prompts(args.bare), backend)

    perf = json.load(open(args.perf, encoding="utf-8")) if args.perf else None
    provenance = {"git_commit": _git_commit(), "model_path": args.model_path,
                  "model_sha256": _sha256(args.model_path), "backend": args.backend, "seeds": {}}
    sc = build_scorecard(records, adtc_res=adtc_res, bare_records=bare_records,
                         provenance=provenance, perf=perf)
    write_run(os.path.join(args.out, args.label), records, sc)

    g = sc["bursa_gold"]
    print(f"incorrect_auto_posts   = {g['incorrect_auto_posts']}")
    print(f"duplicate_blocked_rate = {g['duplicate_blocked_rate']}")
    print(f"action_accuracy        = {g['action_accuracy']}")
    print(f"top1_student_accuracy  = {g['top1_student_accuracy']}")
    for gate in sc["gates_not_exercised"]:
        print(f"WARN: gate not exercised — {gate} (no qualifying cases; vacuous, NOT a pass)",
              file=sys.stderr)
    if sc["gates_failed"]:
        print(f"GATES FAILED: {sc['gates_failed']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
