from fastapi.testclient import TestClient

from bursa import db, projections
from bursa.web import create_app


def test_dashboard_is_runnable_and_seeded(tmp_path):
    with TestClient(create_app(tmp_path / "demo.db")) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Bursa" in response.text
        assert "STU-1042" in response.text
        assert client.get("/health").json()["offline"] is True


def test_exact_demo_transaction_reconciles_through_real_core(tmp_path):
    with TestClient(create_app(tmp_path / "demo.db")) as client:
        response = client.post("/reconcile/TXN-NIPDEMO001", follow_redirects=False)
        assert response.status_code == 303
        page = client.get("/")
        assert "status-auto" in page.text


def test_csv_upload_rejects_non_csv(tmp_path):
    with TestClient(create_app(tmp_path / "demo.db")) as client:
        response = client.post(
            "/import/students",
            files={"file": ("students.txt", b"student_id,name\nS1,Ada\n", "text/plain")},
        )
        assert response.status_code == 415
        assert "Only .csv files are accepted" in response.text


def test_sibling_review_can_be_split_approved_and_persists_snapshot(tmp_path):
    path = tmp_path / "demo.db"
    with TestClient(create_app(path)) as client:
        response = client.post(
            "/transactions/TXN-NIPDEMO002/reconcile", follow_redirects=False
        )
        assert response.status_code == 303
        conn = db.connect(str(path))
        proposal = conn.execute(
            "SELECT * FROM proposals WHERE transaction_id='TXN-NIPDEMO002' "
            "AND status='pending'"
        ).fetchone()
        assert proposal is not None
        proposal_id = proposal["proposal_id"]
        assert "STU-1042" in proposal["candidate_snapshot_json"]
        conn.close()

        approved = client.post(
            f"/proposals/{proposal_id}/approve",
            data={"amount_STU-1042": "40000", "amount_STU-1188": "35000"},
            follow_redirects=False,
        )
        assert approved.status_code == 303

    conn = db.connect(str(path))
    assert conn.execute(
        "SELECT routing_state FROM transactions WHERE transaction_id='TXN-NIPDEMO002'"
    ).fetchone()[0] == "approved"
    assert projections.txn_unapplied(
        conn, conn.execute(
            "SELECT * FROM transactions WHERE transaction_id='TXN-NIPDEMO002'"
        ).fetchone()
    ) == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM ledger_events WHERE transaction_id='TXN-NIPDEMO002' "
        "AND event_type='allocation'"
    ).fetchone()[0] == 2
    conn.close()


def test_auto_allocation_can_be_reversed_with_reason(tmp_path):
    path = tmp_path / "demo.db"
    with TestClient(create_app(path)) as client:
        client.post("/transactions/TXN-NIPDEMO001/reconcile")
        conn = db.connect(str(path))
        event_id = conn.execute(
            "SELECT event_id FROM ledger_events WHERE transaction_id='TXN-NIPDEMO001' "
            "AND event_type='allocation'"
        ).fetchone()[0]
        conn.close()
        response = client.post(
            f"/ledger/events/{event_id}/reverse",
            data={"reason": "Bank transfer was recalled"},
            follow_redirects=False,
        )
        assert response.status_code == 303

    conn = db.connect(str(path))
    assert projections.charge_balance(conn, "CHG-STU-1042-FEE-TUITION-T1") == 4_000_000
    reversal = conn.execute(
        "SELECT decision_path FROM ledger_events WHERE reverses_event_id=?", (event_id,)
    ).fetchone()
    assert reversal["decision_path"] == "Bank transfer was recalled"
    assert conn.execute(
        "SELECT routing_state FROM transactions WHERE transaction_id='TXN-NIPDEMO001'"
    ).fetchone()[0] == "reversed"
    conn.close()


def test_statement_column_mapping_and_import_summary(tmp_path):
    path = tmp_path / "mapped.db"
    with TestClient(create_app(path, seed_demo=False)) as client:
        response = client.post(
            "/import/statement",
            data={
                "map_reference": "Bank Ref",
                "map_date": "Value Date",
                "map_amount": "Credit",
                "map_direction": "Kind",
                "map_payer": "Sender",
                "map_narration": "Details",
            },
            files={
                "file": (
                    "bank.csv",
                    b"Bank Ref,Value Date,Credit,Kind,Sender,Details\n"
                    b"ABC1,2026-03-01,12000,credit,Ada,fees\n",
                    "text/csv",
                )
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "1 accepted, 0 duplicates, 0 rejected" in response.text
        assert "bank.csv" in response.text

    conn = db.connect(str(path))
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    batch = conn.execute(
        "SELECT kind, mapping_json FROM import_batches ORDER BY imported_at DESC LIMIT 1"
    ).fetchone()
    assert batch["kind"] == "statement"
    assert '"reference": "Bank Ref"' in batch["mapping_json"]
    conn.close()
