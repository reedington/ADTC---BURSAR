import json
import urllib.request
import urllib.error
from typing import Protocol


class BackendTransportError(Exception):
    """Transport-level failure (timeout, connection, mid-request death) — retryable once."""


class InferenceBackend(Protocol):
    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str: ...


class FakeBackend:
    def __init__(self, response=None, raises: Exception | None = None):
        self._response = response
        self._raises = raises

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        if self._raises is not None:
            raise self._raises
        if callable(self._response):
            return self._response(raw_prompt, grammar, n_predict)
        return self._response if self._response is not None else "{}"


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
