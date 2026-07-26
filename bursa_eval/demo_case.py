"""Materialize any authored gold scenario as a persistent fictional Bursa database."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from bursa_eval.goldcheck import load_case
from bursa_eval import loader


def materialize_to_file(case_path: str, output_path: str) -> dict:
    case = load_case(case_path)
    source = loader.materialize(case)
    transaction_id = None
    duplicate_blocked = False
    try:
        try:
            transaction_id = loader.insert_case_transaction(source, case)
        except sqlite3.IntegrityError:
            if case.expected.outcome != "duplicate_blocked":
                raise
            duplicate_blocked = True
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(
                f"{output} already exists; choose a new path to preserve the existing ledger"
            )
        destination = sqlite3.connect(output)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    return {
        "case_id": case.id,
        "scenario_family": case.scenario_family,
        "language": case.language,
        "transaction_id": transaction_id,
        "duplicate_blocked": duplicate_blocked,
        "database_path": str(Path(output_path)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="Path to one data/gold/*.yaml case")
    parser.add_argument("--output", required=True, help="New SQLite database path")
    args = parser.parse_args(argv)
    result = materialize_to_file(args.case, args.output)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
