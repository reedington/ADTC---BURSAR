from bursa.normalize import (normalize_name, canonicalize_reference, narration_tokens,
                             jaro_winkler)


def test_normalize_name_strips_titles_and_case():
    assert normalize_name("  Mr. C. N.  OKAFOR ") == "c n okafor"
    assert normalize_name("Chidi-Somto") == "chidi somto"


def test_canonicalize_reference():
    assert canonicalize_reference(" nip-728/1049 ") == "NIP7281049"
    assert canonicalize_reference(None) is None
    assert canonicalize_reference("") is None


def test_narration_tokens():
    assert narration_tokens("CHI AND SOMTO SCH FEE") == ["chi", "and", "somto", "sch", "fee"]
    assert narration_tokens(None) == []


def test_jaro_winkler_identical_and_empty():
    assert jaro_winkler("chidi", "chidi") == 1.0
    assert jaro_winkler("", "x") == 0.0


def test_jaro_winkler_close_names_above_threshold():
    assert jaro_winkler("chidi", "chidy") >= 0.88
    assert jaro_winkler("okafor", "okafo") >= 0.88


def test_jaro_winkler_distinct_names_low():
    assert jaro_winkler("chidi", "bello") < 0.6
