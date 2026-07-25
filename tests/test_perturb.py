import random
from bursa_eval.synth.perturb import ocr_corrupt, name_variant, inject, to_pidgin


def test_ocr_corrupt_deterministic_and_same_length():
    out = ocr_corrupt(random.Random(1), "NIP0100OSS")
    assert out == ocr_corrupt(random.Random(1), "NIP0100OSS")
    assert len(out) == len("NIP0100OSS")


def test_inject_appends_payload_but_keeps_original():
    out = inject(random.Random(1), "chi sch fee")
    assert "chi sch fee" in out and len(out) > len("chi sch fee")


def test_name_variant_deterministic():
    assert name_variant(random.Random(5), "Chidi Okafor") == name_variant(random.Random(5), "Chidi Okafor")


def test_to_pidgin_mentions_names():
    out = to_pidgin(random.Random(1), ["Chi"], 5000)
    assert "Chi" in out
