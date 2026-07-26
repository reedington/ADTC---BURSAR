import json
import urllib.request
import urllib.error
from typing import Protocol


class BackendTransportError(Exception):
    """Transport-level failure (timeout, connection, mid-request death) — retryable once."""


class InferenceBackend(Protocol):
    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str: ...
    # Bare-model / ADTC suites: apply the model's EMBEDDED chat template (the judge-visible
    # artifact). A distinct operation from grammar-constrained generate() — never a mode flag.
    def chat(self, prompt: str) -> str: ...


class FakeBackend:
    def __init__(self, response=None, raises: Exception | None = None, chat_response=None):
        self._response = response
        self._raises = raises
        self._chat_response = chat_response

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        if self._raises is not None:
            raise self._raises
        if callable(self._response):
            return self._response(raw_prompt, grammar, n_predict)
        return self._response if self._response is not None else "{}"

    def chat(self, prompt: str) -> str:
        if callable(self._chat_response):
            return self._chat_response(prompt)
        return self._chat_response if self._chat_response is not None else ""


class LlamaServerBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        body = json.dumps({
            "prompt": raw_prompt, "grammar": grammar, "n_predict": n_predict,
            "temperature": 0, "cache_prompt": True,
        }).encode()
        req = urllib.request.Request(self.base_url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["content"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise BackendTransportError(str(exc)) from exc

    def chat(self, prompt: str) -> str:
        """POST to /v1/chat/completions so the GGUF's EMBEDDED chat template is applied
        (no self-applied Qwen template, no grammar). i5-only; exercised via the runbook."""
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}], "temperature": 0,
        }).encode()
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise BackendTransportError(str(exc)) from exc
