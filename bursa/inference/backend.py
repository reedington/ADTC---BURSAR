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
        self.last_timings = None

    def _request(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise BackendTransportError(str(exc)) from exc

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        return self.completion(
            raw_prompt, max_tokens=n_predict, grammar=grammar
        )

    def completion(
        self,
        raw_prompt: str,
        *,
        max_tokens: int = 512,
        grammar: str | None = None,
    ) -> str:
        body = {
            "prompt": raw_prompt,
            "n_predict": max_tokens,
            "temperature": 0,
            "cache_prompt": True,
        }
        if grammar:
            body["grammar"] = grammar
        payload = self._request("/completion", body)
        self.last_timings = payload.get("timings")
        return payload["content"]

    def chat(self, prompt: str) -> str:
        """POST to /v1/chat/completions so the GGUF's EMBEDDED chat template is applied
        (no self-applied Qwen template, no grammar). i5-only; exercised via the runbook."""
        return self.chat_completion([{"role": "user", "content": prompt}], max_tokens=512)

    def chat_completion(
        self,
        messages: list[dict],
        *,
        grammar: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        body = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if grammar:
            body["grammar"] = grammar
        payload = self._request("/v1/chat/completions", body)
        self.last_timings = payload.get("timings")
        return payload["choices"][0]["message"]["content"]

    def apply_template(self, messages: list[dict]) -> str:
        payload = self._request("/apply-template", {"messages": messages})
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise BackendTransportError("/apply-template returned no prompt")
        return prompt

    def tokenize(self, content: str) -> list[int]:
        payload = self._request("/tokenize", {"content": content})
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise BackendTransportError("/tokenize returned no token list")
        return tokens
