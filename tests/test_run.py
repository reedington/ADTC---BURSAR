import pytest
from bursa.inference.run import run_inference
from bursa.inference.backend import BackendTransportError


class _FlakyBackend:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def generate(self, p, g, n):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise BackendTransportError("transient")
        return '{"ok":true}'


def test_transport_retry_succeeds_second_try():
    b = _FlakyBackend(fail_times=1)
    assert run_inference(b, "p", "g", 100) == '{"ok":true}'
    assert b.calls == 2


def test_transport_gives_up_after_one_retry():
    b = _FlakyBackend(fail_times=5)
    with pytest.raises(BackendTransportError):
        run_inference(b, "p", "g", 100)
    assert b.calls == 2   # original + exactly one retry
