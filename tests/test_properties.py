import os
import tempfile
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition
from bursa.db import connect, init_db
from bursa import ledger, repository as repo, projections as proj, distribute
from bursa.errors import InvariantViolation
from bursa.models import CanonicalTransaction


def _fresh_db(billed=10_000_000):
    conn = connect(os.path.join(tempfile.mkdtemp(), "d.db"))
    init_db(conn)
    conn.execute("INSERT INTO terms VALUES ('T1','s','t',1)")
    conn.execute("INSERT INTO students VALUES ('STU-1','n','n','c','T1')")
    conn.execute("INSERT INTO fee_items VALUES ('FEE-1','f','T1',10)")
    ledger.create_charge(conn, "CHG-1", "STU-1", "FEE-1", "T1", billed, "i", "f", "B")
    return conn


@given(amount=st.integers(min_value=0, max_value=20_000_000))
@settings(max_examples=50, deadline=None)
def test_conservation(amount):
    conn = _fresh_db()
    repo.insert_transaction(conn, CanonicalTransaction(transaction_id="TXN-1",
        source="bank_csv", posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount,
        direction="credit", dedup_hash="h1"))
    events, _ = distribute.distribute(conn, "TXN-1", "STU-1", amount, "e")
    try:
        ledger.post(conn, "TXN-1", events, "e", "d", "TXN-1", "auto")
    except InvariantViolation:
        conn.close()
        return
    txn = repo.get_transaction(conn, "TXN-1")
    # money is conserved: paid-to-charges + credit + unapplied == transaction amount
    total = (proj.charge_paid(conn, "CHG-1") + proj.holder_credit(conn, "STU-1")
             + proj.txn_unapplied(conn, txn))
    conn.close()
    assert total == amount


class LedgerMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.conn = _fresh_db(billed=8_000_000)
        self.txn_seq = 0
        self.posted_event_ids = []

    @rule(amount=st.integers(min_value=1, max_value=3_000_000))
    def post_payment(self, amount):
        self.txn_seq += 1
        tid = f"TXN-{self.txn_seq}"
        repo.insert_transaction(self.conn, CanonicalTransaction(transaction_id=tid,
            source="bank_csv", posted_at="2026-02-14T00:00:00+00:00", amount_minor=amount,
            direction="credit", dedup_hash=f"h{self.txn_seq}"))
        events, _ = distribute.distribute(self.conn, tid, "STU-1", amount, "e")
        try:
            self.posted_event_ids.extend(ledger.post(self.conn, tid, events, "e", "d",
                                                      tid, "auto"))
        except InvariantViolation:
            pass

    @precondition(lambda self: self.posted_event_ids)
    @rule()
    def reverse_last(self):
        ev = self.posted_event_ids.pop()
        row = repo.event_by_id(self.conn, ev)
        if row and row["event_type"] != "reversal":
            try:
                ledger.reverse(self.conn, ev, "bursar", "sm")
            except InvariantViolation:
                pass

    @invariant()
    def charge_never_negative(self):
        assert proj.charge_balance(self.conn, "CHG-1") >= 0

    @invariant()
    def credit_never_negative(self):
        assert proj.holder_credit(self.conn, "STU-1") >= 0

    def teardown(self):
        self.conn.close()


TestLedgerMachine = LedgerMachine.TestCase
TestLedgerMachine.settings = settings(max_examples=20, stateful_step_count=15, deadline=None)
