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


def jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(0, max(len1, len2) // 2 - 1)
    m1 = [False] * len1
    m2 = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if m2[j] or s1[i] != s2[j]:
                continue
            m1[i] = m2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(len1):
        if not m1[i]:
            continue
        while not m2[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    t //= 2
    jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)
