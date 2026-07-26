"""Application-owned local inference runtime.

The runtime is optional so deterministic imports, matching, review, and ledger work remain
available when a GGUF or llama-server is absent. Tests may inject a backend directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bursa.inference.backend import LlamaServerBackend
from bursa.inference.server import LlamaServer


@dataclass
class AppRuntime:
    model_path: str | None = None
    tokenizer_path: str | None = None
    actor: str = "bursar"
    backend: Any | None = None
    server: LlamaServer | None = None
    port: int = 8080
    threads: int = 4
    ctx: int = 2048
    last_error: str | None = field(default=None, init=False)

    @classmethod
    def from_environment(cls) -> "AppRuntime":
        return cls(
            model_path=os.environ.get("BURSA_MODEL_PATH") or None,
            tokenizer_path=os.environ.get("BURSA_TOKENIZER_PATH") or None,
            actor=os.environ.get("BURSA_ACTOR", "bursar").strip() or "bursar",
        )

    def start(self) -> None:
        if self.backend is not None:
            return
        if not self.model_path:
            self.last_error = "BURSA_MODEL_PATH is not configured"
            return
        if not Path(self.model_path).is_file():
            self.last_error = f"Model file does not exist: {self.model_path}"
            return
        try:
            self.server = LlamaServer(
                self.model_path, port=self.port, threads=self.threads, ctx=self.ctx
            )
            self.server.start()
            if not self.server.wait_until_ready():
                self.server.stop()
                raise RuntimeError("llama-server did not become ready within 30 seconds")
            self.backend = LlamaServerBackend(f"http://127.0.0.1:{self.port}")
            self.last_error = None
        except (OSError, RuntimeError, ValueError) as exc:
            self.server = None
            self.backend = None
            self.last_error = str(exc)

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.server = None
        if self.model_path:
            self.backend = None

    def model_available(self) -> bool:
        if self.backend is None:
            return False
        if self.server is None:
            return True
        return self.server.health()

    def health(self) -> dict:
        return {
            "available": self.model_available(),
            "configured": bool(self.model_path or self.backend),
            "model_path": self.model_path,
            "last_error": self.last_error,
            "threads": self.threads,
            "context": self.ctx,
        }
