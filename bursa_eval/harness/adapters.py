"""Evaluation-only adapters for each GGUF's embedded native chat template."""
from __future__ import annotations

from dataclasses import dataclass


def split_qwen_prompt(raw_prompt: str) -> tuple[str, str]:
    system_start = "<|im_start|>system\n"
    user_start = "<|im_start|>user\n"
    end = "<|im_end|>"
    if not raw_prompt.startswith(system_start) or user_start not in raw_prompt:
        raise ValueError("expected Bursa's Qwen system/user evaluation envelope")
    system, remainder = raw_prompt[len(system_start):].split(
        end + "\n" + user_start, 1
    )
    user = remainder.split(end, 1)[0]
    return system, user


@dataclass
class NativeTemplateBackend:
    """Make the gold runner exercise a non-Qwen model's embedded template."""

    backend: object
    family: str
    prompt_token_limit: int = 1300

    def _messages(self, system: str, user: str) -> list[dict]:
        if self.family == "gemma3":
            return [{
                "role": "user",
                "content": f"Instructions:\n{system}\n\nTask:\n{user}",
            }]
        if self.family == "llama3":
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        raise ValueError(f"unsupported native adapter family: {self.family}")

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        system, user = split_qwen_prompt(raw_prompt)
        messages = self._messages(system, user)
        rendered = self.backend.apply_template(messages)
        token_count = len(self.backend.tokenize(rendered))
        if token_count > self.prompt_token_limit:
            raise ValueError(
                f"native prompt is {token_count} tokens; limit is {self.prompt_token_limit}"
            )
        return self.backend.chat_completion(
            messages, grammar=grammar, max_tokens=n_predict
        )

    def chat(self, prompt: str) -> str:
        # Bare and enterprise evaluation intentionally receive no Bursa system prompt.
        return self.backend.chat(prompt)


@dataclass
class QwenEvaluationBackend:
    """Keep production Qwen generation byte-for-byte while enforcing the eval token gate."""

    backend: object
    prompt_token_limit: int = 1300

    def generate(self, raw_prompt: str, grammar: str, n_predict: int) -> str:
        token_count = len(self.backend.tokenize(raw_prompt))
        if token_count > self.prompt_token_limit:
            raise ValueError(
                f"production Qwen prompt is {token_count} tokens; "
                f"limit is {self.prompt_token_limit}"
            )
        return self.backend.generate(raw_prompt, grammar, n_predict)

    def chat(self, prompt: str) -> str:
        return self.backend.chat(prompt)
