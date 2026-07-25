import re

_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "chief", "alhaji", "mallam"}


def normalize_name(raw: str) -> str:
    tokens = re.split(r"[^a-zA-Z]+", (raw or "").lower())
    tokens = [t for t in tokens if t and t not in _TITLES]
    return " ".join(tokens)


def canonicalize_reference(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return cleaned or None


def narration_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t for t in re.split(r"[^a-zA-Z0-9]+", raw.lower()) if t]
