from bursa_eval.models import GoldCase
from bursa_eval.synth import namepools, perturb


def _term():
    return {"id": "T1", "session": "2025/2026", "name": "second_term"}


def _uid(rng):
    return rng.randint(1000, 9999)


def gen_exact_id(rng) -> GoldCase:
    first, last = namepools.pick_name(rng, "en")
    sid = f"STU-{_uid(rng)}"
    amt = rng.choice([25000, 30000, 40000, 50000])
    fam = f"synth-{sid}"
    return GoldCase(
        id=f"synth-exact-{sid}", scenario_family="name_match", language="en", difficulty="easy",
        guardian_family=fam, template_family="synth-exact_id", provenance="synthetic",
        setup={"term": _term(), "students": [{"id": sid, "name": f"{first} {last}",
               "charges": [{"fee_id": "FEE-TUITION", "amount_naira": amt}]}]},
        transaction={"reference": f"NIP{_uid(rng)}", "date": "2026-02-14", "amount_naira": amt,
                     "payer_name": last, "narration": f"sch fee {sid} tuition"},
        expected={"outcome": "auto",
                  "allocations": [{"student_id": sid, "fee_id": "FEE-TUITION", "amount_naira": amt}],
                  "rationale": "Exact ID in narration; amount equals balance."})


def gen_sibling_split(rng) -> GoldCase:
    first1, last = namepools.pick_name(rng, "ig")
    first2, _ = namepools.pick_name(rng, "ig")
    s1, s2 = f"STU-{_uid(rng)}", f"STU-{_uid(rng)}"
    a1, a2 = rng.choice([30000, 40000]), rng.choice([25000, 35000])
    g = f"synth-{last}-{_uid(rng)}"
    return GoldCase(
        id=f"synth-sib-{s1}-{s2}", scenario_family="sibling_split", language="en", difficulty="hard",
        guardian_family=g, template_family="synth-sibling_split", provenance="synthetic",
        setup={"term": _term(),
               "guardians": [{"id": g, "name": f"{last} guardian"}],
               "students": [
                   {"id": s1, "name": f"{first1} {last}", "aliases": [first1[:3]], "guardians": [g],
                    "charges": [{"fee_id": "FEE-TUITION", "amount_naira": a1}]},
                   {"id": s2, "name": f"{first2} {last}", "aliases": [first2[:3]], "guardians": [g],
                    "charges": [{"fee_id": "FEE-TUITION", "amount_naira": a2}]}]},
        transaction={"reference": f"NIP{_uid(rng)}", "date": "2026-02-14", "amount_naira": a1 + a2,
                     "payer_name": last, "narration": f"{first1[:3]} and {first2[:3]} sch fee"},
        expected={"outcome": "review", "allocations": [
                      {"student_id": s1, "fee_id": "FEE-TUITION", "amount_naira": a1},
                      {"student_id": s2, "fee_id": "FEE-TUITION", "amount_naira": a2}],
                  "rationale": "Two sibling aliases; combined balance equals payment."})


def gen_overpayment(rng) -> GoldCase:
    first, last = namepools.pick_name(rng, "yo")
    sid = f"STU-{_uid(rng)}"
    bal = rng.choice([30000, 40000])
    over = rng.choice([5000, 10000])
    fam = f"synth-{sid}"
    return GoldCase(
        id=f"synth-over-{sid}", scenario_family="overpayment", language="en", difficulty="medium",
        guardian_family=fam, template_family="synth-overpayment", provenance="synthetic",
        setup={"term": _term(), "students": [{"id": sid, "name": f"{first} {last}",
               "charges": [{"fee_id": "FEE-TUITION", "amount_naira": bal}]}]},
        transaction={"reference": f"NIP{_uid(rng)}", "date": "2026-02-14", "amount_naira": bal + over,
                     "payer_name": last, "narration": f"{first} fees full"},
        expected={"outcome": "review",
                  "allocations": [{"student_id": sid, "fee_id": "FEE-TUITION", "amount_naira": bal}],
                  "credits": [{"holder": sid, "amount_naira": over}],
                  "rationale": "Payment exceeds balance; surplus becomes credit."})


def gen_pidgin_ambiguous(rng) -> GoldCase:
    first1, last1 = namepools.pick_name(rng, "pcm")
    first2, last2 = namepools.pick_name(rng, "pcm")
    s1, s2 = f"STU-{_uid(rng)}", f"STU-{_uid(rng)}"
    amt = rng.choice([20000, 25000])
    return GoldCase(
        id=f"synth-pcm-{s1}-{s2}", scenario_family="ambiguous_candidates", language="pcm",
        difficulty="hard", guardian_family=f"synth-amb-{_uid(rng)}",
        template_family="synth-pidgin_ambiguous", provenance="synthetic",
        setup={"term": _term(), "students": [
                   {"id": s1, "name": f"{first1} {last1}", "aliases": [first1],
                    "charges": [{"fee_id": "FEE-TUITION", "amount_naira": amt}]},
                   {"id": s2, "name": f"{first2} {last2}", "aliases": [first1],  # colliding alias
                    "charges": [{"fee_id": "FEE-TUITION", "amount_naira": amt}]}]},
        transaction={"reference": f"NIP{_uid(rng)}", "date": "2026-02-14", "amount_naira": amt,
                     "payer_name": "unknown",
                     "narration": perturb.to_pidgin(rng, [first1], amt)},
        expected={"outcome": "review", "allocations": [],
                  "pool_must_include": [s1, s2],
                  "rationale": "Pidgin narration names an alias shared by two students; ambiguous."})


def gen_no_candidate(rng) -> GoldCase:
    first, last = namepools.pick_name(rng, "en")
    sid = f"STU-{_uid(rng)}"
    return GoldCase(
        id=f"synth-none-{sid}", scenario_family="no_candidate", language="en", difficulty="medium",
        guardian_family=f"synth-{sid}", template_family="synth-no_candidate", provenance="synthetic",
        setup={"term": _term(), "students": [{"id": sid, "name": f"{first} {last}",
               "charges": [{"fee_id": "FEE-TUITION", "amount_naira": 30000}]}]},
        transaction={"reference": f"NIP{_uid(rng)}", "date": "2026-02-14", "amount_naira": 99999,
                     "payer_name": "Stranger Zzz", "narration": "unrelated deposit zzz"},
        expected={"outcome": "unmatched", "allocations": [],
                  "rationale": "No name/alias/amount match to any student."})


def gen_duplicate(rng) -> GoldCase:
    first, last = namepools.pick_name(rng, "en")
    sid = f"STU-{_uid(rng)}"
    amt = rng.choice([30000, 40000])
    ref = f"NIP{_uid(rng)}{_uid(rng)}"
    fam = f"synth-{sid}"
    return GoldCase(
        id=f"synth-dup-{sid}", scenario_family="duplicate_reference", language="en",
        difficulty="medium", guardian_family=fam, template_family="synth-duplicate",
        provenance="synthetic",
        setup={"term": _term(), "students": [{"id": sid, "name": f"{first} {last}",
               "charges": [{"fee_id": "FEE-TUITION", "amount_naira": amt}]}],
               "history": [{"transaction": {"reference": ref, "date": "2026-01-10",
                            "amount_naira": amt, "payer_name": last},
                            "allocations": [{"student_id": sid, "fee_id": "FEE-TUITION",
                                             "amount_naira": amt}]}]},
        transaction={"reference": ref, "date": "2026-02-14", "amount_naira": amt,
                     "payer_name": last, "narration": f"{first} fees"},
        expected={"outcome": "duplicate_blocked", "allocations": [],
                  "rationale": "Reference already posted in history; duplicate."})


TEMPLATES = {
    "synth_exact_id": gen_exact_id,
    "synth_sibling_split": gen_sibling_split,
    "synth_overpayment": gen_overpayment,
    "synth_pidgin_ambiguous": gen_pidgin_ambiguous,
    "synth_no_candidate": gen_no_candidate,
    "synth_duplicate": gen_duplicate,
}
