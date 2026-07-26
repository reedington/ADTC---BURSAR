from bursa.inference.backend import FakeBackend
from bursa.runtime import AppRuntime


def test_runtime_without_model_stays_in_safe_unavailable_mode():
    runtime = AppRuntime()
    runtime.start()
    assert runtime.backend is None
    assert runtime.model_available() is False
    assert "not configured" in runtime.health()["last_error"]
    runtime.stop()


def test_injected_backend_is_available_and_not_replaced():
    backend = FakeBackend()
    runtime = AppRuntime(backend=backend, actor="test-bursar")
    runtime.start()
    assert runtime.backend is backend
    assert runtime.model_available() is True
    runtime.stop()
    assert runtime.backend is backend
