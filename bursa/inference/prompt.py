from bursa.inference.constants import PROMPT_TOKEN_BUDGET

SYSTEM_PROMPT = (
    "You are Bursa's reconciliation assistant. Given a bank transaction and up to five "
    "candidate students, decide the allocation. Output ONLY JSON matching this contract: "
    '{"transaction_id","interpretation","candidate_allocations":[{"student_id","amount_minor",'
    '"reason_codes"}],"recommended_action":"auto|review|unmatched","explanation","ambiguities"}. '
    "student_id must be one of the provided candidates; amounts are integer minor units (kobo); "
    "never invent IDs; prefer review when uncertain."
)


def _candidate_block(c, include_aliases, include_history) -> str:
    parts = [f"id={c.student_id} name={c.name}"]
    alias_fired = "fuzzy_name" in c.fired_signals or "alias_token_overlap" in c.fired_signals
    # keep aliases when the section is on OR an alias signal fired (never shed fired evidence)
    if c.aliases and (include_aliases or alias_fired):
        parts.append("aliases=" + ",".join(c.aliases))
    if c.guardians:
        parts.append("guardian=" + ",".join(c.guardians))
    if c.outstanding:
        parts.append("outstanding=" + ",".join(f"{cid}:{bal}" for cid, bal in c.outstanding))
    if include_history:
        parts.append(f"prior_payer={c.is_prior_payer}")
    elif c.is_prior_payer:
        parts.append("prior_payer=true")   # keep the fired marker even with history dropped
    return "  - " + " ".join(parts)


def _assemble(txn, cands, allowed_codes, include_aliases, include_history) -> str:
    user_lines = [
        f"Transaction: id={txn['transaction_id']} payer={txn['payer_name']} "
        f"amount_minor={txn['amount_minor']}",
        f"Narration: {txn['narration']}",
        "Allowed reason_codes: " + ",".join(allowed_codes),
        "Candidates:",
    ]
    for c in cands:
        user_lines.append(_candidate_block(c, include_aliases, include_history))
    user = "\n".join(user_lines) + " /no_think"
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant")


def build(txn, candidates, counter, allowed_codes, budget=PROMPT_TOKEN_BUDGET):
    """Static-prefix-first prompt with a deterministic overflow ladder. Returns
    (raw_prompt, surviving_candidates), or (None, []) on PROMPT_BUDGET_EXCEEDED.
    Never truncates narration or the contract; never sheds fired evidence."""
    cands = list(candidates)
    include_history = True
    include_aliases = True

    def fits():
        raw = _assemble(txn, cands, allowed_codes, include_aliases, include_history)
        return raw if counter.count(raw) <= budget else None

    raw = fits()                       # 0: full
    if raw:
        return raw, cands
    include_history = False            # 1: drop history section (markers preserved)
    raw = fits()
    if raw:
        return raw, cands
    include_aliases = False            # 2: trim aliases (fired ones preserved)
    raw = fits()
    if raw:
        return raw, cands
    while len(cands) > 3:              # 3: reduce candidates 5->4->3 (drop lowest score)
        cands = cands[:-1]
        print(f"[prompt] txn={txn['transaction_id']} reduced candidates to {len(cands)}")
        raw = fits()
        if raw:
            return raw, cands
    print(f"[prompt] txn={txn['transaction_id']} PROMPT_BUDGET_EXCEEDED")   # 4: floor
    return None, []
