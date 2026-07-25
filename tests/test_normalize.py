from bursa.normalize import normalize_name, canonicalize_reference, narration_tokens


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
