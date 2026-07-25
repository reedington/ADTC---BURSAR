import math
from typing import Protocol
from bursa.inference.constants import SPECIAL_MARGIN


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Conservative upper bound: over-counts English BPE (~4 chars/token) via /3,
    surcharges non-ASCII (multi-byte -> potentially multiple tokens), plus a margin."""

    def count(self, text: str) -> int:
        nonascii = sum(1 for ch in text if ord(ch) > 127)
        return math.ceil(len(text) / 3) + nonascii + SPECIAL_MARGIN


class QwenTokenizer:
    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_json_path)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


def get_token_counter(tokenizer_path: str | None) -> TokenCounter:
    if tokenizer_path:
        try:
            return QwenTokenizer(tokenizer_path)
        except Exception:
            pass
    return HeuristicTokenCounter()
