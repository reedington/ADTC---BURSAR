"""Offline, single-user Bursa web application."""
from __future__ import annotations

import csv
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bursa import db, ledger, money, pipeline, projections, repository as repo, review
from bursa.config import Config
from bursa.errors import InvariantViolation
from bursa.importers.fees import import_fees
from bursa.importers.setup import import_guardians, import_relationships, import_terms
from bursa.importers.statement import import_statement
from bursa.importers.students import import_students
from bursa.runtime import AppRuntime


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path("data/local/bursa-demo.db")
MAX_UPLOAD_BYTES = 1_000_000

IMPORTERS = {
    "terms": import_terms,
    "students": import_students,
    "guardians": import_guardians,
    "relationships": import_relationships,
    "fees": import_fees,
    "statement": import_statement,
}

MAPPABLE_FIELDS = {
    "terms": ("term_id", "session", "term_name", "is_active"),
    "students": ("student_id", "name", "class", "term_id", "aliases"),
    "guardians": ("guardian_id", "name", "phone_suffix"),
    "relationships": ("student_id", "guardian_id"),
    "fees": ("student_id", "fee_id", "fee_name", "term_id", "amount", "priority"),
    "statement": ("reference", "date", "amount", "direction", "payer", "narration"),
}


def _ensure_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(str(path))
    try:
        db.init_db(conn)
    finally:
        conn.close()


def _seed_demo(path: Path) -> None:
    """Idempotently seed fictional data through the same import and ledger paths as uploads."""
    conn = db.connect(str(path))
    try:
        if conn.execute("SELECT 1 FROM terms LIMIT 1").fetchone():
            return
        import_terms(
            conn,
            [{
                "term_id": "T1",
                "session": "2025/2026",
                "term_name": "second_term",
                "is_active": "1",
            }],
            "demo_terms.csv",
        )
        import_students(conn, [
            {"student_id": "STU-1042", "name": "Chidi Okafor", "class": "JSS1",
             "term_id": "T1", "aliases": "Chi"},
            {"student_id": "STU-1188", "name": "Somtochukwu Okafor", "class": "JSS2",
             "term_id": "T1", "aliases": "Somto"},
            {"student_id": "STU-2001", "name": "Adaeze Nwosu", "class": "JSS1",
             "term_id": "T1", "aliases": "Ada"},
        ], "demo_students.csv")
        import_guardians(conn, [
            {"guardian_id": "GUA-OKAFOR", "name": "C N Okafor", "phone_suffix": "1042"},
            {"guardian_id": "GUA-NWOSU", "name": "Ngozi Nwosu", "phone_suffix": "2001"},
        ], "demo_guardians.csv")
        import_relationships(conn, [
            {"student_id": "STU-1042", "guardian_id": "GUA-OKAFOR"},
            {"student_id": "STU-1188", "guardian_id": "GUA-OKAFOR"},
            {"student_id": "STU-2001", "guardian_id": "GUA-NWOSU"},
        ], "demo_relationships.csv")
        import_fees(conn, [
            {"student_id": "STU-1042", "fee_id": "FEE-TUITION", "fee_name": "Tuition",
             "term_id": "T1", "amount": "40000", "priority": "10"},
            {"student_id": "STU-1188", "fee_id": "FEE-TUITION", "fee_name": "Tuition",
             "term_id": "T1", "amount": "35000", "priority": "10"},
            {"student_id": "STU-2001", "fee_id": "FEE-TUITION", "fee_name": "Tuition",
             "term_id": "T1", "amount": "30000", "priority": "10"},
        ], "demo_fees.csv")
        import_statement(conn, [
            {"reference": "NIPDEMO001", "date": "2026-02-14", "amount": "40000",
             "payer": "C N OKAFOR", "narration": "SCH FEES STU-1042", "direction": "credit"},
            {"reference": "NIPDEMO002", "date": "2026-02-15", "amount": "75000",
             "payer": "C N OKAFOR", "narration": "CHI AND SOMTO SCH FEE",
             "direction": "credit"},
            {"reference": "NIPDEMO003", "date": "2026-02-16", "amount": "30000",
             "payer": "N NWOSO", "narration": "ada nwoso school fee", "direction": "credit"},
        ], "demo_statement.csv")
    finally:
        conn.close()


def _read_csv(upload: UploadFile, raw: bytes) -> list[dict]:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV exceeds the 1 MB demonstration limit")
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Only .csv files are accepted")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, "CSV must be UTF-8 encoded") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV has no header row")
    return list(reader)


def _format_event(row) -> dict:
    return {
        **dict(row),
        "amount_display": money.format_naira(row["amount_minor"]),
    }


def _dashboard(path: Path, runtime: AppRuntime, request: Request) -> dict:
    conn = db.connect(str(path))
    try:
        state_filter = request.query_params.get("state", "").strip()
        where = "WHERE routing_state=?" if state_filter else ""
        params = (state_filter,) if state_filter else ()
        transactions = [
            {
                **dict(row),
                "amount_display": money.format_naira(row["amount_minor"]),
                "unapplied_display": money.format_naira(projections.txn_unapplied(conn, row)),
            }
            for row in conn.execute(
                f"SELECT * FROM transactions {where} "
                "ORDER BY posted_at DESC, transaction_id",
                params,
            ).fetchall()
        ]
        students = []
        for row in conn.execute("SELECT * FROM students ORDER BY name").fetchall():
            charges = repo.charges_for_student(conn, row["student_id"])
            balance = sum(projections.charge_balance(conn, c["charge_id"]) for c in charges)
            students.append({
                **dict(row),
                "balance_display": money.format_naira(balance),
                "credit_display": money.format_naira(
                    projections.holder_credit(conn, row["student_id"])
                ),
                "status": projections.student_status(conn, row["student_id"]),
            })

        events = repo.live_events(conn)
        billed = sum(e["amount_minor"] for e in events if e["event_type"] == "charge_created")
        allocated = sum(e["amount_minor"] for e in events if e["event_type"] == "allocation")
        credit_applied = sum(
            e["amount_minor"] for e in events if e["event_type"] == "credit_application"
        )
        credit_granted = sum(
            e["amount_minor"] for e in events if e["event_type"] == "credit_grant"
        )
        credited = credit_granted - credit_applied
        received = conn.execute(
            "SELECT COALESCE(SUM(amount_minor),0) FROM transactions WHERE direction='credit'"
        ).fetchone()[0]
        import_batches = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM import_batches ORDER BY imported_at DESC LIMIT 12"
            ).fetchall()
        ]
        selected_batch = None
        batch_id = request.query_params.get("batch")
        if batch_id:
            batch = conn.execute(
                "SELECT * FROM import_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if batch:
                selected_batch = {
                    **dict(batch),
                    "errors": [
                        dict(row)
                        for row in conn.execute(
                            "SELECT * FROM import_errors WHERE batch_id=? ORDER BY row_number",
                            (batch_id,),
                        ).fetchall()
                    ],
                }
        return {
            "transactions": transactions,
            "students": students,
            "event_count": conn.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0],
            "review_count": conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE routing_state='review'"
            ).fetchone()[0],
            "database": str(path),
            "model_health": runtime.health(),
            "state_filter": state_filter,
            "import_batches": import_batches,
            "selected_batch": selected_batch,
            "flash": request.query_params.get("message"),
            "error": request.query_params.get("error"),
            "totals": {
                "billed": money.format_naira(billed),
                "received": money.format_naira(received),
                "allocated": money.format_naira(allocated),
                "outstanding": money.format_naira(
                    max(0, billed - allocated - credit_applied)
                ),
                "credited": money.format_naira(credited),
                "unapplied": money.format_naira(
                    max(0, received - allocated - credit_granted)
                ),
            },
        }
    finally:
        conn.close()


def _transaction_detail(path: Path, txn_id: str) -> dict | None:
    conn = db.connect(str(path))
    try:
        txn = repo.get_transaction(conn, txn_id)
        if txn is None:
            return None
        proposals = []
        for row in repo.proposals_for_transaction(conn, txn_id):
            item = dict(row)
            item["candidates"] = json.loads(item["candidate_snapshot_json"] or "[]")
            item["ambiguities"] = json.loads(item["ambiguities_json"] or "[]")
            original_allocations = [
                {
                    **dict(allocation),
                    "amount_display": money.format_naira(allocation["amount_minor"]),
                    "amount_input": money.format_naira_input(allocation["amount_minor"]),
                    "reason_codes": json.loads(allocation["reason_codes"] or "[]"),
                }
                for allocation in repo.proposal_allocations(conn, row["proposal_id"])
            ]
            decided_allocations = json.loads(item["decision_allocations_json"] or "[]")
            item["allocations"] = (
                [
                    {
                        **allocation,
                        "amount_display": money.format_naira(allocation["amount_minor"]),
                        "amount_input": money.format_naira_input(allocation["amount_minor"]),
                        "reason_codes": ["HUMAN_APPROVED"],
                    }
                    for allocation in decided_allocations
                ]
                if item["status"] == "approved" and decided_allocations
                else original_allocations
            )
            item["confidence_display"] = (
                f"{row['confidence'] * 100:.1f}%" if row["confidence"] is not None else "—"
            )
            proposals.append(item)
        audit_rows = conn.execute(
            "SELECT * FROM ledger_events WHERE transaction_id=? "
            "OR reverses_event_id IN "
            "(SELECT event_id FROM ledger_events WHERE transaction_id=?) "
            "ORDER BY event_id DESC",
            (txn_id, txn_id),
        ).fetchall()
        reversed_ids = {
            row["reverses_event_id"] for row in audit_rows if row["reverses_event_id"] is not None
        }
        audit = [
            {**_format_event(row), "is_reversed": row["event_id"] in reversed_ids}
            for row in audit_rows
        ]
        return {
            "transaction": {
                **dict(txn),
                "amount_display": money.format_naira(txn["amount_minor"]),
                "unapplied_display": money.format_naira(projections.txn_unapplied(conn, txn)),
            },
            "proposals": proposals,
            "audit": audit,
        }
    finally:
        conn.close()


def _student_detail(path: Path, student_id: str) -> dict | None:
    conn = db.connect(str(path))
    try:
        student = conn.execute(
            "SELECT * FROM students WHERE student_id=?", (student_id,)
        ).fetchone()
        if student is None:
            return None
        charges = []
        for charge in repo.charges_for_student(conn, student_id):
            billed = projections.charge_billed(conn, charge["charge_id"])
            paid = projections.charge_paid(conn, charge["charge_id"])
            charges.append({
                **dict(charge),
                "billed_display": money.format_naira(billed),
                "paid_display": money.format_naira(paid),
                "balance_display": money.format_naira(billed - paid),
            })
        events = [
            _format_event(row)
            for row in conn.execute(
                "SELECT * FROM ledger_events WHERE student_id=? OR holder=? "
                "ORDER BY event_id DESC",
                (student_id, student_id),
            ).fetchall()
        ]
        return {
            "student": dict(student),
            "charges": charges,
            "events": events,
            "status": projections.student_status(conn, student_id),
            "credit_display": money.format_naira(projections.holder_credit(conn, student_id)),
        }
    finally:
        conn.close()


def create_app(
    database_path: str | Path | None = None,
    seed_demo: bool = True,
    runtime: AppRuntime | None = None,
    config: Config | None = None,
) -> FastAPI:
    path = Path(database_path or os.environ.get("BURSA_DB_PATH", DEFAULT_DB_PATH))
    app_runtime = runtime or AppRuntime.from_environment()
    app_config = config or Config()

    @asynccontextmanager
    async def lifespan(_app):
        _ensure_database(path)
        if seed_demo:
            _seed_demo(path)
        app_runtime.start()
        try:
            yield
        finally:
            app_runtime.stop()

    app = FastAPI(title="Bursa", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.database_path = path
    app.state.runtime = app_runtime
    app.state.config = app_config
    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    def render_dashboard(request: Request, *, error=None, status_code=200):
        context = _dashboard(path, app_runtime, request)
        if error:
            context["error"] = error
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context=context,
            status_code=status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return render_dashboard(request)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "offline": True,
            "database": str(path),
            "model": app_runtime.health(),
            "schema_version": db.SCHEMA_VERSION,
        }

    def run_reconciliation(txn_id: str):
        conn = db.connect(str(path))
        try:
            if repo.get_transaction(conn, txn_id) is None:
                raise HTTPException(404, "Transaction not found")
            pipeline.reconcile(
                conn,
                txn_id,
                app_config,
                backend=app_runtime.backend,
                tokenizer_path=app_runtime.tokenizer_path,
            )
        finally:
            conn.close()
        return RedirectResponse(f"/transactions/{txn_id}", status_code=303)

    @app.post("/transactions/{txn_id}/reconcile")
    async def reconcile_transaction(txn_id: str):
        return run_reconciliation(txn_id)

    @app.post("/reconcile/{txn_id}")
    async def reconcile_compatibility_alias(txn_id: str):
        return run_reconciliation(txn_id)

    @app.post("/transactions/{txn_id}/retry")
    async def retry_transaction(txn_id: str):
        return run_reconciliation(txn_id)

    @app.get("/transactions/{txn_id}", response_class=HTMLResponse)
    async def transaction_detail(request: Request, txn_id: str):
        context = _transaction_detail(path, txn_id)
        if context is None:
            raise HTTPException(404, "Transaction not found")
        context["model_health"] = app_runtime.health()
        context["error"] = request.query_params.get("error")
        context["message"] = request.query_params.get("message")
        return templates.TemplateResponse(
            request=request, name="transaction.html", context=context
        )

    @app.get("/students/{student_id}", response_class=HTMLResponse)
    async def student_detail(request: Request, student_id: str):
        context = _student_detail(path, student_id)
        if context is None:
            raise HTTPException(404, "Student not found")
        return templates.TemplateResponse(
            request=request, name="student.html", context=context
        )

    @app.post("/proposals/{proposal_id}/approve")
    async def approve_proposal(request: Request, proposal_id: str):
        form = await request.form()
        allocations = []
        for key, value in form.multi_items():
            if not key.startswith("amount_") or not str(value).strip():
                continue
            try:
                amount_minor = money.parse_naira(str(value))
            except ValueError as exc:
                query = urlencode({"error": f"{key}: {exc}"})
                conn = db.connect(str(path))
                proposal = repo.get_proposal(conn, proposal_id)
                conn.close()
                txn_id = proposal["transaction_id"] if proposal else "unknown"
                return RedirectResponse(
                    f"/transactions/{txn_id}?{query}", status_code=303
                )
            allocations.append({
                "student_id": key.removeprefix("amount_"),
                "amount_minor": amount_minor,
            })
        credit_holder = str(form.get("credit_holder") or "").strip() or None
        conn = db.connect(str(path))
        try:
            proposal = repo.get_proposal(conn, proposal_id)
            if proposal is None:
                raise HTTPException(404, "Proposal not found")
            txn_id = proposal["transaction_id"]
            try:
                review.approve(
                    conn,
                    proposal_id,
                    allocations,
                    app_runtime.actor,
                    credit_holder=credit_holder,
                )
            except InvariantViolation as exc:
                query = urlencode({"error": f"Posting blocked: {', '.join(exc.violations)}"})
                return RedirectResponse(
                    f"/transactions/{txn_id}?{query}", status_code=303
                )
        finally:
            conn.close()
        return RedirectResponse(
            f"/transactions/{txn_id}?message=Proposal+approved", status_code=303
        )

    @app.post("/proposals/{proposal_id}/reject")
    async def reject_proposal(proposal_id: str):
        conn = db.connect(str(path))
        try:
            proposal = repo.get_proposal(conn, proposal_id)
            if proposal is None:
                raise HTTPException(404, "Proposal not found")
            txn_id = proposal["transaction_id"]
            review.reject(conn, proposal_id, app_runtime.actor)
        finally:
            conn.close()
        return RedirectResponse(
            f"/transactions/{txn_id}?message=Proposal+rejected", status_code=303
        )

    @app.post("/ledger/events/{event_id}/reverse")
    async def reverse_event(request: Request, event_id: int):
        form = await request.form()
        reason = str(form.get("reason") or "").strip()
        target_txn = None
        conn = db.connect(str(path))
        try:
            event = repo.event_by_id(conn, event_id)
            if event is None:
                raise HTTPException(404, "Ledger event not found")
            target_txn = event["transaction_id"]
            if not reason:
                query = urlencode({"error": "A reversal reason is required"})
                return RedirectResponse(
                    f"/transactions/{target_txn}?{query}", status_code=303
                )
            try:
                ledger.reverse(conn, event_id, app_runtime.actor, reason)
                txn = repo.get_transaction(conn, target_txn)
                if txn is not None and projections.txn_used(conn, target_txn) == 0:
                    repo.set_routing_state(conn, target_txn, "reversed")
            except InvariantViolation as exc:
                query = urlencode({"error": f"Reversal blocked: {', '.join(exc.violations)}"})
                return RedirectResponse(
                    f"/transactions/{target_txn}?{query}", status_code=303
                )
        finally:
            conn.close()
        return RedirectResponse(
            f"/transactions/{target_txn}?message=Compensating+reversal+recorded",
            status_code=303,
        )

    async def do_import(request: Request, kind: str):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return render_dashboard(request, error="Choose a CSV file.", status_code=400)
        try:
            raw = await upload.read(MAX_UPLOAD_BYTES + 1)
            rows = _read_csv(upload, raw)
        except HTTPException as exc:
            return render_dashboard(request, error=exc.detail, status_code=exc.status_code)
        mapping = {
            field: str(form.get(f"map_{field}") or "").strip()
            for field in MAPPABLE_FIELDS[kind]
            if str(form.get(f"map_{field}") or "").strip()
        }
        conn = db.connect(str(path))
        try:
            result = IMPORTERS[kind](
                conn,
                rows,
                upload.filename,
                column_mapping=mapping or None,
            )
        finally:
            conn.close()
        query = urlencode({
            "batch": result["batch_id"],
            "message": (
                f"Import complete: {result.get('accepted', 0)} accepted, "
                f"{result.get('duplicate', 0)} duplicates, "
                f"{result.get('rejected', 0)} rejected"
            ),
        })
        return RedirectResponse(f"/?{query}", status_code=303)

    @app.post("/import/{kind}")
    async def upload_csv(request: Request, kind: str):
        if kind not in IMPORTERS:
            raise HTTPException(404, "Unknown import type")
        return await do_import(request, kind)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("bursa.web:app", host="127.0.0.1", port=8000, reload=False)
