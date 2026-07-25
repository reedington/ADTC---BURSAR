from bursa_eval.synth.seeds import stable_seed, config_hash


def test_stable_seed_is_deterministic_and_int():
    a = stable_seed(42, "sibling_split", 3)
    b = stable_seed(42, "sibling_split", 3)
    assert a == b and isinstance(a, int)
    assert stable_seed(42, "sibling_split", 4) != a


def test_config_hash_stable_regardless_of_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert len(config_hash({"a": 1})) == 64
