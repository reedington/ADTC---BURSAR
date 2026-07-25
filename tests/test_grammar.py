import re
from bursa.inference.grammar import build_grammar


def test_grammar_contains_candidate_ids_only():
    g = build_grammar("TXN-1", ["STU-1042", "STU-1188"], ["NAME_ALIAS_MATCH"])
    assert "STU-1042" in g and "STU-1188" in g
    assert "STU-9999" not in g          # non-candidate cannot appear
    assert "TXN-1" in g                 # transaction id fixed as a literal
    assert "NAME_ALIAS_MATCH" in g
    assert "root ::=" in g


def test_grammar_ids_equal_input_ids():
    ids = ["STU-1", "STU-2", "STU-3"]
    g = build_grammar("TXN-9", ids, ["X"])
    found = set(re.findall(r"STU-\d+", g))
    assert found == set(ids)
