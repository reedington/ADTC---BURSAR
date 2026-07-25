import random
from bursa_eval.synth.seeds import stable_seed
from bursa_eval.synth.templates import TEMPLATES
from bursa_eval.goldcheck import check_case

# abstention templates (empty allocations) — used to guarantee the >=25% floor.
_ABSTAIN = {"synth_pidgin_ambiguous", "synth_no_candidate", "synth_duplicate"}


def _sig(case):
    from bursa_eval.dataset import near_dup_signature   # lazy: avoids a cycle with dataset
    return near_dup_signature(case)


def generate(base_seed: int, n: int, mix=None, gold=()):
    """Pure, deterministic, validator-gated generator. Same (base_seed, n, mix) -> identical
    output across processes. Drops any case that fails check_case or near-dups an existing or
    gold case; guarantees >=25% abstention by construction."""
    weights = mix or {tid: 1.0 for tid in TEMPLATES}
    tids = list(weights)
    w = [weights[t] for t in tids]
    abstain_tids = sorted(_ABSTAIN & set(tids)) or tids
    gold_sigs = {_sig(g) for g in gold}
    out, sigs = [], set()
    n_abstain_target = -(-n // 4)   # ceil(n/4) == 25%
    attempts = 0

    def _try_add(tid, idx):
        nonlocal attempts
        attempts += 1
        case = TEMPLATES[tid](random.Random(stable_seed(base_seed, tid, idx)))
        sig = _sig(case)
        if sig in sigs or sig in gold_sigs or check_case(case):
            return False
        sigs.add(sig)
        out.append(case)
        return True

    i = 0
    while (sum(1 for c in out if c.is_abstention()) < n_abstain_target
           and len(out) < n and i < n * 50):
        _try_add(abstain_tids[i % len(abstain_tids)], i)
        i += 1
    while len(out) < n and i < n * 100:
        tid = random.Random(stable_seed(base_seed, "pick", i)).choices(tids, weights=w, k=1)[0]
        _try_add(tid, i)
        i += 1

    generate.last_attempts = attempts   # for gate-rate inspection
    return out[:n]
