import re
from dataclasses import dataclass, field
from bursa import repository as repo, projections as proj, normalize

W_STRONG, W_MEDIUM, W_WEAK = 10, 5, 2
FUZZY_NAME_THRESHOLD = 0.88
MAX_CANDIDATES = 5


@dataclass
class Candidate:
    student_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    guardians: list[str] = field(default_factory=list)
    outstanding: list[tuple] = field(default_factory=list)
    is_prior_payer: bool = False
    siblings: list[str] = field(default_factory=list)
    score: int = 0
    fired_signals: dict = field(default_factory=dict)


def _all_students(conn):
    return conn.execute("SELECT * FROM students").fetchall()


def _aliases(conn, sid):
    return conn.execute("SELECT alias, normalized_alias FROM student_aliases "
                        "WHERE student_id = ?", (sid,)).fetchall()


def _guardians(conn, sid):
    return conn.execute(
        "SELECT g.* FROM guardians g JOIN student_guardians sg ON g.guardian_id = sg.guardian_id "
        "WHERE sg.student_id = ?", (sid,)).fetchall()


def _siblings(conn, sid):
    return [r["student_id"] for r in conn.execute(
        "SELECT DISTINCT sg2.student_id FROM student_guardians sg1 "
        "JOIN student_guardians sg2 ON sg1.guardian_id = sg2.guardian_id "
        "WHERE sg1.student_id = ? AND sg2.student_id != ?", (sid, sid)).fetchall()]


def generate(conn, txn) -> list[Candidate]:
    payer = normalize.normalize_name(txn["payer_name"] or "")
    narr_tokens = set(normalize.narration_tokens(txn["narration"]))
    narr_digits = set(re.findall(r"\d{3,}", txn["narration"] or ""))
    prior = proj.payer_history(conn, payer) if payer else set()
    amount = txn["amount_minor"]
    probe = narr_tokens | set(payer.split())

    scored: dict[str, tuple] = {}   # sid -> (score, fired)

    def bump(sid, weight, signal):
        s, fired = scored.get(sid, (0, {}))
        if signal not in fired:
            fired[signal] = weight
            scored[sid] = (s + weight, fired)

    for st in _all_students(conn):
        sid = st["student_id"]
        norm = st["normalized_name"]
        name_tokens = set(norm.split())
        aliases = _aliases(conn, sid)
        alias_norms = [a["normalized_alias"] for a in aliases]

        if name_tokens & probe:
            bump(sid, W_STRONG, "name_token_overlap")
        for an in alias_norms:
            if an in probe:
                bump(sid, W_STRONG, "alias_token_overlap")

        targets = [norm] + alias_norms
        for tok in probe:
            if any(normalize.jaro_winkler(tok, t) >= FUZZY_NAME_THRESHOLD for t in targets if t):
                bump(sid, W_STRONG, "fuzzy_name")
                break

        for g in _guardians(conn, sid):
            gname = g["normalized_name"]
            if gname and (gname in (payer or "") or set(gname.split()) & probe):
                bump(sid, W_STRONG, "guardian_name")
            if g["phone_suffix"] and any(d.endswith(g["phone_suffix"]) for d in narr_digits):
                bump(sid, W_STRONG, "phone_suffix")

        if sid in prior:
            bump(sid, W_STRONG, "prior_payer")

        for c in repo.charges_for_student(conn, sid):
            if amount > 0 and proj.charge_balance(conn, c["charge_id"]) == amount:
                bump(sid, W_STRONG, "exact_balance")
                break

    for sid in list(scored.keys()):
        for sib in _siblings(conn, sid):
            if sib not in scored:
                bump(sib, W_MEDIUM, "sibling_rider")

    cands = []
    for sid, (score, fired) in scored.items():
        st = conn.execute("SELECT * FROM students WHERE student_id = ?", (sid,)).fetchone()
        cands.append(Candidate(
            student_id=sid, name=st["name"],
            aliases=[a["alias"] for a in _aliases(conn, sid)],
            guardians=[g["name"] for g in _guardians(conn, sid)],
            outstanding=[(c["charge_id"], proj.charge_balance(conn, c["charge_id"]))
                         for c in repo.charges_for_student(conn, sid)
                         if proj.charge_balance(conn, c["charge_id"]) > 0],
            is_prior_payer=(sid in prior),
            siblings=_siblings(conn, sid),
            score=score, fired_signals=fired))

    cands.sort(key=lambda c: (-c.score, c.student_id))
    if len(cands) > MAX_CANDIDATES:
        cut = cands[MAX_CANDIDATES - 1].score
        dropped = len(cands) - MAX_CANDIDATES
        print(f"[candidates] txn={txn['transaction_id']} dropped {dropped} candidates at score<={cut}")
        cands = cands[:MAX_CANDIDATES]
    return cands
