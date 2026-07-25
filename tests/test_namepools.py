import random
from bursa_eval.synth.namepools import pick_name, nickname, initials


def test_pick_name_deterministic():
    assert pick_name(random.Random(1)) == pick_name(random.Random(1))


def test_pick_name_by_language():
    first, last = pick_name(random.Random(2), lang="ig")
    assert first and last


def test_nickname_and_initials():
    assert nickname(random.Random(3), "Chidi")
    assert initials("Chidi Okafor") == "C.O."
