import pytest

from bursa import db
from bursa_eval.demo_case import materialize_to_file


def test_gold_case_can_become_persistent_fictional_demo(tmp_path):
    output = tmp_path / "sibling.db"
    result = materialize_to_file(
        "data/gold/gold-0003-sibling-split-en.yaml", str(output)
    )
    assert result["scenario_family"] == "sibling_split"
    assert result["transaction_id"]
    conn = db.connect(str(output))
    assert conn.execute("SELECT COUNT(*) FROM students").fetchone()[0] >= 2
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    conn.close()


def test_demo_materializer_never_overwrites_existing_ledger(tmp_path):
    output = tmp_path / "existing.db"
    output.write_text("keep me")
    with pytest.raises(FileExistsError):
        materialize_to_file("data/gold/gold-0001-exact-id-en.yaml", str(output))
    assert output.read_text() == "keep me"
