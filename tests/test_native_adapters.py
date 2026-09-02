import pytest

from bursa_eval.harness.adapters import (
    NativeTemplateBackend,
    QwenEvaluationBackend,
    split_qwen_prompt,
)


RAW = (
    "<|im_start|>system\nFollow the contract.<|im_end|>\n"
    "<|im_start|>user\nDo the work.<|im_end|>\n<|im_start|>assistant"
)


class Backend:
    def __init__(self, tokens=20):
        self.messages = None
        self.tokens = tokens

    def apply_template(self, messages):
        self.messages = messages
        return "rendered"

    def tokenize(self, content):
        return list(range(self.tokens))

    def chat_completion(self, messages, grammar=None, max_tokens=0):
        self.messages = messages
        return "{}"

    def chat(self, prompt):
        return "plain"

    def generate(self, raw_prompt, grammar, n_predict):
        self.generate_args = (raw_prompt, grammar, n_predict)
        return "production"


def test_split_qwen_prompt():
    assert split_qwen_prompt(RAW) == ("Follow the contract.", "Do the work.")


def test_gemma_folds_system_into_user_and_bare_stays_plain():
    backend = Backend()
    adapter = NativeTemplateBackend(backend, "gemma3")
    assert adapter.generate(RAW, "grammar", 100) == "{}"
    assert len(backend.messages) == 1
    assert "Follow the contract." in backend.messages[0]["content"]
    assert adapter.chat("hello") == "plain"


def test_llama_preserves_system_role_and_enforces_token_limit():
    backend = Backend(tokens=1301)
    adapter = NativeTemplateBackend(backend, "llama3")
    with pytest.raises(ValueError, match="limit"):
        adapter.generate(RAW, "grammar", 100)
    assert backend.messages[0]["role"] == "system"


def test_qwen_adapter_preserves_production_generate_call_and_checks_server_tokens():
    backend = Backend(tokens=1300)
    adapter = QwenEvaluationBackend(backend)
    assert adapter.generate(RAW, "grammar", 100) == "production"
    assert backend.generate_args == (RAW, "grammar", 100)
    with pytest.raises(ValueError, match="limit"):
        QwenEvaluationBackend(Backend(tokens=1301)).generate(
            RAW, "grammar", 100
        )
