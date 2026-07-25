import glob
import json
import os
from collections import defaultdict
from bursa_eval.synth.seeds import stable_seed
from bursa import normalize


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
