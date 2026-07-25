import random
from bursa_eval.synth.seeds import stable_seed
from bursa_eval.synth.templates import TEMPLATES
from bursa_eval.goldcheck import check_case

# abstention templates (empty allocations).
_ABSTAIN = {"synth_pidgin_ambiguous", "synth_no_candidate", "synth_duplicate"}
# weight that puts the abstention SHARE in the 25-30% band (3 abstain + 3 non-abstain templates):
#   share = 3*w / (3*w + 3*1) = w / (w + 1) = 0.28  =>  w = 0.389
_ABSTAIN_WEIGHT = 0.389


class PoolExhaustionError(RuntimeError):
    """The name/amount pools could not yield n unique valid cases within the attempt budget."""


def default_mix():
    return {tid: (_ABSTAIN_WEIGHT if tid in _ABSTAIN else 1.0) for tid in TEMPLATES}


def _sig(case):
    from bursa_eval.dataset import near_dup_signature   # lazy: avoids a cycle with dataset
    return near_dup_signature(case)


def generate(base_seed: int, n: int, mix=None, gold=(), max_attempts=None):
    """Pure, deterministic, validator-gated generator. Same (base_seed, n, mix) -> identical
    output across processes. Drops any case that fails check_case or near-dups an existing/gold
    case. Abstention share is set by the weighted mix (25-30% band), NOT a forced floor.
    Raises PoolExhaustionError if it cannot reach n within max_attempts (default n*40)."""
    weights = mix or default_mix()
    tids = list(weights)
    w = [weights[t] for t in tids]
    gold_sigs = {_sig(g) for g in gold}
    out, sigs = [], set()
    max_attempts = max_attempts if max_attempts is not None else n * 40
    i = attempts = 0
    while len(out) < n and attempts < max_attempts:
        tid = random.Random(stable_seed(base_seed, "pick", i)).choices(tids, weights=w, k=1)[0]
        i += 1
        attempts += 1
        case = TEMPLATES[tid](random.Random(stable_seed(base_seed, tid, i)))
        sig = _sig(case)
        if sig in sigs or sig in gold_sigs or check_case(case):
            continue
        sigs.add(sig)
        out.append(case)

    generate.last_attempts = attempts
    generate.last_drop_rate = (attempts - len(out)) / attempts if attempts else 0.0
    if len(out) < n:
        raise PoolExhaustionError(
            f"produced {len(out)}/{n} unique valid cases in {attempts} attempts "
            f"(drop rate {generate.last_drop_rate:.1%}); expand name/amount pools or reduce n")
    return out
