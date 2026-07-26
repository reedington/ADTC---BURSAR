import glob
import sys
import yaml
from collections import Counter
from bursa import distribute, ledger, normalize, repository as repo
from bursa.errors import InvariantViolation
from bursa_eval import loader
from bursa_eval.models import GoldCase, SCENARIO_FAMILIES, naira_to_minor


def load_case(path) -> GoldCase:
    with open(path) as f:
        return GoldCase(**yaml.safe_load(f))


def check_case(case: GoldCase) -> list[str]:
    """Return a list of problems (empty = valid). Reuses the production ledger/constraint
    engine as the validity oracle: a valid gold answer must POST without invariant violation."""
    problems = []
    try:
        conn = loader.materialize(case)
    except Exception as exc:
        return [f"setup/history failed to materialize: {exc}"]
    try:
        for sid in case.pool_truth():
            if not repo.student_exists(conn, sid):
                problems.append(f"references unknown student {sid}")

        if case.expected.outcome == "duplicate_blocked":
            canon = normalize.canonicalize_reference(case.transaction.reference)
            if canon is None:
                problems.append("duplicate_blocked requires a transaction reference")
            elif conn.execute("SELECT 1 FROM transactions WHERE reference=?",
                              (canon,)).fetchone() is None:
                problems.append("duplicate_blocked but the reference is not present in setup.history")
        else:
            txn_id = loader.insert_case_transaction(conn, case)
            events = loader.build_expected_events(conn, case, txn_id)
            if events:
                try:
                    ledger.post(conn, txn_id, events, "goldset", "expected", txn_id, "gold")
                except InvariantViolation as exc:
                    problems.append(f"expected answer violates invariants: {exc.violations}")
                except Exception as exc:
                    problems.append(f"expected answer failed to post: {exc}")
    finally:
        conn.close()
    return problems


def _alloc_credit_set(events):
    return {(e.student_id, e.charge_id, e.amount_minor) for e in events}


def distribute_divergence(case: GoldCase) -> str | None:
    """Warn (not fail) when the authored expected split differs from what distribute() produces
    for the same per-student totals. A perfect model yields distribute()'s natural split, so an
    authored bursar override makes exact_allocation_accuracy unreachable for this case."""
    if case.expected.outcome == "duplicate_blocked" or not case.expected.allocations:
        return None
    conn = loader.materialize(case)
    try:
        txn_id = loader.insert_case_transaction(conn, case)
        expected = _alloc_credit_set(loader.build_expected_events(conn, case, txn_id))
        totals = {}
        for a in case.expected.allocations:
            totals[a.student_id] = totals.get(a.student_id, 0) + naira_to_minor(a.amount_naira)
        # A student-level model proposal contains the full amount intended for that student.
        # When the authored answer includes overpayment credit, include it in the amount sent to
        # distribute(); otherwise the natural path can never reproduce the credit event.
        for credit in case.expected.credits:
            if repo.student_exists(conn, credit.holder):
                totals[credit.holder] = (
                    totals.get(credit.holder, 0) + naira_to_minor(credit.amount_naira))
        natural = []
        for sid, tot in totals.items():
            evs, _ = distribute.distribute(conn, txn_id, sid, tot, "goldcheck")
            natural.extend(evs)
        if _alloc_credit_set(natural) != expected:
            return (f"authored split diverges from distribute(); "
                    f"exact_allocation_accuracy unreachable for {case.id}")
        return None
    finally:
        conn.close()


def check_dir(path="data/gold") -> int:
    paths = sorted(glob.glob(f"{path}/*.yaml"))
    families, langs, n_abstain, n_nonauto, n_fail = Counter(), Counter(), 0, 0, 0
    for p in paths:
        try:
            case = load_case(p)
        except Exception as exc:
            print(f"FAIL {p}: schema error: {exc}")
            n_fail += 1
            continue
        problems = check_case(case)
        families[case.scenario_family] += 1
        langs[case.language] += 1
        n_abstain += 1 if case.is_abstention() else 0
        n_nonauto += 1 if case.expected.outcome != "auto" else 0
        if problems:
            n_fail += 1
            for prob in problems:
                print(f"FAIL {p}: {prob}")
        else:
            print(f"ok   {p}  [{case.scenario_family}/{case.language}/{case.expected.outcome}]")
            divergence = distribute_divergence(case)
            if divergence:
                print(f"WARN {p}: {divergence}")

    total = len(paths)
    print(f"\n{total - n_fail}/{total} valid | families {len(families)}/{len(SCENARIO_FAMILIES)} "
          f"| langs {sorted(langs)} | abstention {n_abstain}/{total} | non-auto {n_nonauto}/{total}")
    missing = SCENARIO_FAMILIES - set(families)
    if missing:
        print(f"coverage gap — families not yet represented: {sorted(missing)}")
    if total and n_nonauto / total < 0.25:
        print("WARN: non-auto (review/unmatched/duplicate_blocked) share < 25%")
    return 1 if n_fail else 0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/gold"
    raise SystemExit(check_dir(path))


if __name__ == "__main__":
    main()
