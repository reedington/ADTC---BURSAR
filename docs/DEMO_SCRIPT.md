# Bursa two-minute demo script

The final recording must show the real local application and profiler. Do not substitute mock
screens for the working model or invent benchmark values.

## 0:00–0:15 — Problem

Show a shared Nigerian school bank statement beside a manually maintained student spreadsheet.

Narration:

> Schools receive parent-named transfers, sibling lump sums, nicknames, instalments, and unclear
> narrations. Bursa reconciles them on the laptop the school already owns.

## 0:15–0:35 — Offline setup and import

- Disable networking or show the network-off indicator.
- Launch `.venv/bin/bursa-web`.
- Import `demo/students.csv`, `demo/fees.csv`, and `demo/statement.csv`, or use the fictional
  pre-seeded data.
- Show the offline indicator and local database path.

## 0:35–0:55 — Deterministic fast path

- Open the `STU-1042` transaction.
- Reconcile it.
- Show the transaction moving to `auto` and Chidi's balance changing.
- Explain that the exact student ID never requires model inference.

## 0:55–1:20 — Ambiguous sibling payment

- Show `CHI AND SOMTO SCH FEE`.
- Display the candidate evidence and the local model's proposed NGN 40,000 / NGN 35,000 split.
- Emphasize that the model proposes and explains; it cannot write to the ledger.

## 1:20–1:35 — Safety and abstention

- Run visible prompt `tp_002`.
- Show the model refusing to guess between Amina and Sadiq.
- Show the payment in review/unmatched state with all NGN 20,000 still visible.

## 1:35–1:48 — Audit trail

- Show the derived student ledger and append-only event count.
- Explain integer minor units, duplicate blocking, explicit credit, and reversal events.

## 1:48–2:00 — Hardware proof

- Show the final `submission.json` or profiler terminal output.
- Read the measured tokens/s, peak RSS, temperature, and crash count.
- End with: “AI that the bursar can run offline, and financial rules the model cannot break.”

## Recording checklist

- Final duration is no more than two minutes.
- The real GGUF is visible running locally.
- No real student, guardian, bank, email, or phone data appears.
- Profiler numbers match the committed `submission.json`.
- Captions are readable at 1080p.
- Export to `media/bursa-demo.mp4`; do not commit an unfinished placeholder video.
