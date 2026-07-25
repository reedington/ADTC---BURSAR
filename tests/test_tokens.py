import os
import pytest
from bursa.inference.tokens import HeuristicTokenCounter, get_token_counter


def test_heuristic_positive_and_deterministic():
    h = HeuristicTokenCounter()
    assert h.count("hello world") == h.count("hello world")
    assert h.count("hello world") > 0


def test_heuristic_surcharges_nonascii():
    h = HeuristicTokenCounter()
    assert h.count("₦naira") > h.count("naira")


def test_get_counter_falls_back_without_asset():
    c = get_token_counter(None)
    assert c.count("x") >= 1


ASSET = os.environ.get("QWEN_TOKENIZER_JSON")


@pytest.mark.skipif(not ASSET, reason="no tokenizer.json asset")
def test_heuristic_overcounts_corpus():
    from bursa.inference.tokens import QwenTokenizer
    real = QwenTokenizer(ASSET)
    h = HeuristicTokenCounter()
    corpus = ["Payment for STU-1042 tuition second term",
              "CHI AND SOMTO SCH FEE ₦75,000", "Tobi remaining 35k",
              "school fees for my pikin", "NIP7281049 transfer Okafor"]
    for text in corpus:
        assert h.count(text) >= real.count(text), text
