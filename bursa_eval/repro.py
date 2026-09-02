"""Small reproducibility helpers shared by command-line workflows."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip())


def require_clean(*, allow_dirty: bool) -> None:
    if not allow_dirty and git_dirty():
        raise RuntimeError(
            "working tree is dirty; commit the intended baseline or pass --allow-dirty"
        )


def environment_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def free_gib(path: str | Path) -> float:
    return shutil.disk_usage(Path(path)).free / (1024 ** 3)


def require_disk(path: str | Path, minimum_gib: float) -> None:
    available = free_gib(path)
    if available < minimum_gib:
        raise RuntimeError(
            f"disk preflight failed: {available:.2f} GiB free, "
            f"{minimum_gib:.2f} GiB required at {Path(path).resolve()}"
        )


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def immutable_run_dir(root: str | Path, label: str) -> Path:
    path = Path(root) / f"{timestamp()}-{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
