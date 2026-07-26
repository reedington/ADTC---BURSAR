from dataclasses import dataclass, field
from collections import defaultdict
from bursa import projections as proj, repository as repo
from bursa.models import LedgerEventInput


@dataclass
class ValidationResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def validate(conn, txn, proposed: list[LedgerEventInput]) -> ValidationResult:
    v: list[str] = []

    # INV-03: integer, non-negative amounts
    for ev in proposed:
        if not isinstance(ev.amount_minor, int) or ev.amount_minor < 0:
            v.append("INV-03")
            break

    # INV-09: provenance present
    for ev in proposed:
        if not (ev.actor and ev.source and ev.evidence_ref and ev.decision_path):
            v.append("INV-09")
            break

    # INV-04: referenced ids exist
    for ev in proposed:
        if ev.charge_id is not None and not repo.charge_exists(conn, ev.charge_id):
            v.append("INV-04")
            break
        if ev.student_id is not None and not repo.student_exists(conn, ev.student_id):
            v.append("INV-04")
            break
        if ev.event_type == "credit_grant" and (
            not ev.holder or not repo.holder_exists(conn, ev.holder)
        ):
            v.append("INV-04")
            break

    # INV-01: transaction capacity (allocations + credit grants funded by txn)
    txn_id = txn["transaction_id"]
    if txn_id is not None:
        proposed_txn = sum(ev.amount_minor for ev in proposed
                           if ev.transaction_id == txn_id
                           and ev.event_type in ("allocation", "credit_grant"))
        if proj.txn_used(conn, txn_id) + proposed_txn > txn["amount_minor"]:
            v.append("INV-01")

    # INV-05: no charge overfilled
    by_charge = defaultdict(int)
    for ev in proposed:
        if ev.charge_id and ev.event_type in ("allocation", "credit_application"):
            by_charge[ev.charge_id] += ev.amount_minor
    for charge_id, add in by_charge.items():
        if repo.charge_exists(conn, charge_id):
            if proj.charge_paid(conn, charge_id) + add > proj.charge_billed(conn, charge_id):
                v.append("INV-05")
                break

    # Credit sufficiency: applications <= available credit per holder
    by_holder = defaultdict(int)
    for ev in proposed:
        if ev.event_type == "credit_application" and ev.holder:
            by_holder[ev.holder] += ev.amount_minor
    for holder, add in by_holder.items():
        if add > proj.holder_credit(conn, holder):
            v.append("INV-05")  # credit non-negativity is an INV-05-class floor
            break

    return ValidationResult(ok=(len(v) == 0), violations=v)
