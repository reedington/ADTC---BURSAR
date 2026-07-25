import hashlib
import json


def stable_seed(base_seed: int, template_id: str, index: int) -> int:
    """Process-stable per-case seed (NEVER builtin hash(), which is PYTHONHASHSEED-salted)."""
    digest = hashlib.sha256(f"{base_seed}:{template_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def config_hash(config: dict) -> str:
    """sha256 over canonical (sorted-key) JSON — same stability class as stable_seed."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
