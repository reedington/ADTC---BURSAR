# Bursa demonstration fixtures

These files are fictional and contain no student or bank personal data.

Run the local application:

```bash
.venv/bin/bursa-web
```

Open `http://127.0.0.1:8000`. The application seeds the same fictional scenario automatically,
or the CSV files in this directory can be imported manually in this order:

1. `students.csv`
2. `fees.csv`
3. `statement.csv`

The statement demonstrates:

- an exact student-ID fast path;
- an ambiguous sibling transfer;
- a nickname/misspelling case that must remain reviewable.

The broader authored gold set in `data/gold/` covers every evaluation family. Any case can become
a persistent, fictional demo database without hand-editing fixtures:

```bash
.venv/bin/bursa-demo-case data/gold/gold-0003-sibling-split-en.yaml \
  --output data/local/sibling-demo.db
BURSA_DB_PATH=data/local/sibling-demo.db .venv/bin/bursa-web
```

Use a new output path for every run. The command refuses to overwrite an existing ledger.
