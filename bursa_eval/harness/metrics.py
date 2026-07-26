def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute_metrics(records) -> dict:
    m = {}
    # HARD GATE 1: an auto-post with the wrong money.
    m["incorrect_auto_posts"] = sum(
        1 for r in records if r.would_auto_post and r.exact_alloc_hit is False)
    # HARD GATE 2: an unblocked duplicate = money doubling.
    dup = [r.dup_blocked for r in records if r.dup_blocked is not None]
    m["duplicate_blocked_rate"] = (sum(1 for d in dup if d) / len(dup)) if dup else None

    m["action_accuracy"] = _mean([r.correct_action for r in records])
    m["valid_json_rate"] = _mean([r.valid_json for r in records])
    m["top1_student_accuracy"] = _mean([r.top1_hit for r in records])
    m["exact_allocation_accuracy"] = _mean([r.exact_alloc_hit for r in records])
    m["sibling_split_accuracy"] = _mean(
        [r.exact_alloc_hit for r in records if r.family == "sibling_split"])
    m["pool_recall"] = _mean([r.pool_recall_hit for r in records])
    m["unsupported_id_rate"] = _mean([r.unsupported_id for r in records])

    # Abstention precision/recall: model_abstains ({} or unmatched) vs true_abstention.
    abstained = [r for r in records if r.model_abstains is True]
    truths = [r for r in records if r.true_abstention]
    m["abstention_precision"] = (
        sum(1 for r in abstained if r.true_abstention) / len(abstained)) if abstained else None
    m["abstention_recall"] = (
        sum(1 for r in truths if r.model_abstains) / len(truths)) if truths else None

    langs = {r.language for r in records}
    m["language_subset_accuracy"] = {
        lg: _mean([r.top1_hit for r in records if r.language == lg]) for lg in sorted(langs)}
    return m


def evaluate_gates(metrics: dict) -> list[str]:
    """Return the names of failing hard gates ([] = pass). The scorecard maps this to exit code."""
    fails = []
    if metrics.get("incorrect_auto_posts", 0) != 0:
        fails.append("incorrect_auto_posts")
    rate = metrics.get("duplicate_blocked_rate")
    if rate is not None and rate < 1.0:
        fails.append("duplicate_blocked_rate")
    return fails
