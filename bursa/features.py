from bursa import normalize

FEATURES_VERSION = 1


def extract(txn, surviving, model_data, dry_ok, chosen_id, budget_shed=False) -> dict:
    chosen = next((c for c in surviving if c.student_id == chosen_id), None)
    payer_tokens = set(normalize.normalize_name(txn.get("payer_name") or "").split())
    narr_tokens = set(normalize.narration_tokens(txn.get("narration")))
    probe = payer_tokens | narr_tokens

    name_sim = 0.0
    guardian_rel = 0
    amt_agree = 0.0
    prior = 0
    if chosen is not None:
        targets = [normalize.normalize_name(chosen.name)] + \
                  [normalize.normalize_name(a) for a in chosen.aliases]
        for tok in probe:
            for t in targets:
                if t:
                    name_sim = max(name_sim, normalize.jaro_winkler(tok, t))
        guardian_rel = 1 if any(normalize.normalize_name(g) and
                                normalize.normalize_name(g).split()[0] in probe
                                for g in chosen.guardians) else 0
        prior = 1 if chosen.is_prior_payer else 0
        alloc = next((a for a in model_data.get("candidate_allocations", [])
                      if a.get("student_id") == chosen_id), None)
        if alloc and chosen.outstanding:
            bal = chosen.outstanding[0][1]
            amt = alloc.get("amount_minor", 0)
            amt_agree = 1 - min(1, abs(amt - bal) / max(bal, 1))

    top = surviving[0].student_id if surviving else None
    second = surviving[1].score if len(surviving) > 1 else 0
    top_score = surviving[0].score if surviving else 1
    separation = (top_score - second) / max(top_score, 1)

    return {
        "features_version": FEATURES_VERSION,
        "name_alias_similarity": round(name_sim, 4),
        "guardian_relationship": guardian_rel,
        "amount_to_balance_agreement": round(amt_agree, 4),
        "historical_payer_consistency": prior,
        "candidate_separation": round(separation, 4),
        "llm_ranking_consistency": 1 if chosen_id == top else 0,
        "constraint_validation_result": 1 if dry_ok else 0,
        "budget_shed": 1 if budget_shed else 0,
        "candidate_count": len(surviving),
    }
