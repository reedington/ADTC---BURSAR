import argparse
from bursa_eval.models import SCENARIO_FAMILIES, LANGUAGES

_TEMPLATE = '''# __ID__-__LANG__.yaml — fill in the blanks, then run: python -m bursa_eval.goldcheck
id: __ID__
scenario_family: __FAMILY__
language: __LANG__
difficulty: __DIFF__            # easy | medium | hard
guardian_family: CHANGE_ME     # groups cases sharing a guardian family (split key)
template_family: CHANGE_ME     # groups cases from one narration template (split key)
provenance: team_authored
setup:
  term: { id: T1, session: "2025/2026", name: second_term }
  guardians:
    - { id: G1, name: "GUARDIAN NAME", phone_suffix: "0000" }
  students:
    - id: STU-1
      name: "STUDENT NAME"
      aliases: []
      guardians: [G1]
      charges: [{ fee_id: FEE-TUITION, amount_naira: 0 }]   # int or "0.00" — never an unquoted float
  history: []                  # optional prior txns replayed through the ledger, e.g.:
  # history:
  #   - transaction: { reference: NIP-OLD, date: "2026-01-10", amount_naira: 0, payer_name: "..." }
  #     allocations: [{ student_id: STU-1, fee_id: FEE-TUITION, amount_naira: 0 }]
transaction:
  reference: NIP0000           # omit if the statement row genuinely has none
  date: "2026-02-14"
  amount_naira: 0
  payer_name: "PAYER NAME"
  narration: "NARRATION TEXT"
expected:
  outcome: review              # auto | review | unmatched | duplicate_blocked
  allocations:
    - { student_id: STU-1, fee_id: FEE-TUITION, amount_naira: 0 }
  credits: []                  # optional overpayment surplus: [{ holder: STU-1, amount_naira: 0 }]
  # pool_must_include: [STU-1] # set when allocations is empty (abstention) to define pool-recall truth
  rationale: "WHY this is the correct outcome."
'''


def render(case_id: str, family: str, lang: str, difficulty: str = "medium") -> str:
    return (_TEMPLATE.replace("__ID__", case_id).replace("__FAMILY__", family)
            .replace("__LANG__", lang).replace("__DIFF__", difficulty))


def main():
    ap = argparse.ArgumentParser(description="Scaffold a new Bursa gold case YAML.")
    ap.add_argument("--family", required=True, choices=sorted(SCENARIO_FAMILIES))
    ap.add_argument("--lang", default="en", choices=sorted(LANGUAGES))
    ap.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--id", default=None)
    ap.add_argument("--out", default="data/gold")
    args = ap.parse_args()
    case_id = args.id or f"gold-XXXX-{args.family}"
    fname = f"{args.out}/{case_id}-{args.lang}.yaml"
    with open(fname, "w") as f:
        f.write(render(case_id, args.family, args.lang, args.difficulty))
    print(f"wrote {fname}  — fill it in, then run: python -m bursa_eval.goldcheck")


if __name__ == "__main__":
    main()
