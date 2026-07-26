# Privacy and local-data policy

Bursa is designed for offline use on a school-controlled laptop.

- Student, guardian, bank-statement, receipt, and ledger data stay on the local device.
- The application performs no analytics, telemetry, crash reporting, or cloud inference.
- `llama-server` is contacted only over loopback (`127.0.0.1`).
- `data/local/`, local CSV files, SQLite databases, model weights, and environment files are
  excluded from Git.
- Public evaluation and training examples must be fictional or irreversibly de-identified.
- Raw payment narrations are treated as untrusted data and cannot override application rules.
- The language model never writes to the ledger; all proposed IDs and allocations are validated
  against supplied candidates and financial invariants.

## Operator responsibilities

The school controls local retention, backup, device access, and deletion. Production deployments
should use full-disk encryption, a dedicated operating-system account, encrypted backups, and a
documented retention period appropriate to the school's legal obligations.

## Incident reporting

Do not include student or bank data in a public issue. Report security or privacy concerns
privately to the repository owner.
