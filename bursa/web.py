"""Small offline demonstration surface for Bursa's real financial core.

This is intentionally a local single-user interface. It does not claim production authentication,
OCR, or multi-tenant deployment. All state is stored in a local SQLite file.
"""
from __future__ import annotations

import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bursa import db, money, pipeline, projections
from bursa.importers.fees import import_fees
from bursa.importers.statement import import_statement
from bursa.importers.students import import_students


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path("data/local/bursa-demo.db")
MAX_UPLOAD_BYTES = 1_000_000


def _ensure_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(str(path))
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='terms'").fetchone()
        if not exists:
            db.init_db(conn)
    finally:
        conn.close()


def _seed_demo(path: Path) -> None:
    """Idempotently seed fictional data through the same import and ledger paths as uploads."""
    conn = db.connect(str(path))
    try:
        if conn.execute("SELECT 1 FROM terms LIMIT 1").fetchone():
            return
        with db.transaction(conn):
            conn.execute("INSERT INTO terms VALUES ('T1','2025/2026','second_term',1)")
        import_students(conn, [
            {"student_id": "STU-1042", "name": "Chidi Okafor", "class": "JSS1",
             "term_id": "T1", "aliases": "Chi"},
            {"student_id": "STU-1188", "name": "Somtochukwu Okafor", "class": "JSS2",
             "term_id": "T1", "aliases": "Somto"},
            {"student_id": "STU-2001", "name": "Adaeze Nwosu", "class": "JSS1",
             "term_id": "T1", "aliases": "Ada"},
        ], "demo_students.csv")
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


def _dashboard(path: Path) -> dict:
    conn = db.connect(str(path))
    try:
        transactions = [
            {
                **dict(row),
                "amount_display": money.format_naira(row["amount_minor"]),
            }
            for row in conn.execute(
                "SELECT * FROM transactions ORDER BY posted_at DESC, transaction_id").fetchall()
        ]
        students = []
        for row in conn.execute("SELECT * FROM students ORDER BY name").fetchall():
            charges = conn.execute(
                "SELECT charge_id FROM charges WHERE student_id=?", (row["student_id"],)).fetchall()
            balance = sum(projections.charge_balance(conn, c["charge_id"]) for c in charges)
            students.append({
                **dict(row),
                "balance_display": money.format_naira(balance),
                "status": projections.student_status(conn, row["student_id"]),
            })
        event_count = conn.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0]
        review_count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE routing_state='review'").fetchone()[0]
        return {
            "transactions": transactions,
            "students": students,
            "event_count": event_count,
            "review_count": review_count,
            "database": str(path),
        }
    finally:
        conn.close()


def create_app(database_path: str | Path | None = None, seed_demo: bool = True) -> FastAPI:
    path = Path(database_path or os.environ.get("BURSA_DB_PATH", DEFAULT_DB_PATH))

    @asynccontextmanager
    async def lifespan(_app):
        _ensure_database(path)
        if seed_demo:
            _seed_demo(path)
        yield

    app = FastAPI(title="Bursa", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.database_path = path
    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(
            request=request, name="dashboard.html", context=_dashboard(path))

    @app.get("/health")
    async def health():
        return {"status": "ok", "offline": True, "database": str(path)}

    @app.post("/reconcile/{txn_id}")
    async def reconcile_transaction(txn_id: str):
        conn = db.connect(str(path))
        try:
            if conn.execute(
                    "SELECT 1 FROM transactions WHERE transaction_id=?", (txn_id,)).fetchone() is None:
                raise HTTPException(404, "Transaction not found")
            pipeline.reconcile(conn, txn_id)
        finally:
            conn.close()
        return RedirectResponse("/", status_code=303)

    async def do_import(upload: UploadFile, kind: str):
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
        rows = _read_csv(upload, raw)
        conn = db.connect(str(path))
        try:
            if kind == "students":
                result = import_students(conn, rows, upload.filename)
            elif kind == "fees":
                result = import_fees(conn, rows, upload.filename)
            else:
                result = import_statement(conn, rows, upload.filename)
        finally:
            conn.close()
        return RedirectResponse(
            f"/?imported={result.get('accepted', 0)}&rejected={result.get('rejected', 0)}",
            status_code=303)

    @app.post("/import/students")
    async def upload_students(file: UploadFile = File(...)):
        return await do_import(file, "students")

    @app.post("/import/fees")
    async def upload_fees(file: UploadFile = File(...)):
        return await do_import(file, "fees")

    @app.post("/import/statement")
    async def upload_statement(file: UploadFile = File(...)):
        return await do_import(file, "statement")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("bursa.web:app", host="127.0.0.1", port=8000, reload=False)
