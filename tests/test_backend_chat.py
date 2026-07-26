"""The bare-model / ADTC suites drive the model's EMBEDDED chat template via a distinct
backend.chat() method (never a mode flag on generate()). FakeBackend fakes it for CI."""
from bursa.inference.backend import FakeBackend


def test_fake_backend_chat_returns_string_response():
    b = FakeBackend(chat_response="Thank you for your payment.")
    assert b.chat("Draft a thank-you.") == "Thank you for your payment."


def test_fake_backend_chat_supports_callable_keyed_on_prompt():
    b = FakeBackend(chat_response=lambda prompt: "4" if "2+2" in prompt else "?")
    assert b.chat("what is 2+2?") == "4"
    assert b.chat("something else") == "?"


def test_fake_backend_chat_defaults_empty_and_is_independent_of_generate():
    b = FakeBackend(response="{}")          # generate configured, chat not
    assert b.chat("anything") == ""
    assert b.generate("p", "g", 8) == "{}"  # generate() path untouched
