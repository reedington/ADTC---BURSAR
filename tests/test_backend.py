import pytest
from bursa.inference.backend import FakeBackend, BackendTransportError


def test_fake_returns_canned():
    fb = FakeBackend(response='{"ok":true}')
    assert fb.generate("p", "g", 100) == '{"ok":true}'


def test_fake_can_raise_transport():
    fb = FakeBackend(raises=BackendTransportError("boom"))
    with pytest.raises(BackendTransportError):
        fb.generate("p", "g", 100)


def test_fake_callable_response():
    fb = FakeBackend(response=lambda prompt, grammar, n: f"got:{n}")
    assert fb.generate("p", "g", 42) == "got:42"
