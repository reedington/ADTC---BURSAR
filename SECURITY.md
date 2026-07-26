# Security policy

## Supported version

The current `main` branch is the only supported pre-release version.

## Trust boundaries

- CSV rows, filenames, payment narrations, OCR text, and model outputs are untrusted.
- Money is represented in integer minor units.
- SQL uses parameterized statements.
- Model outputs are schema-validated and restricted to supplied candidate IDs.
- Financial invariants are enforced before posting.
- Ledger history is append-only; corrections use compensating reversals.
- Model or OCR failure routes to review and cannot post money.

## Reporting

Report vulnerabilities privately to the repository owner. Do not place real student, guardian,
bank, or payment information in a public issue.

## Pre-release limitations

The current prototype does not provide application authentication or database encryption. It is
intended for a local demonstration environment. A school deployment must add device access
controls and full-disk encryption.
