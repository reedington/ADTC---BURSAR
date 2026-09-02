import hashlib
import json

from bursa.calibrator import CalibrationArtifact, FEATURE_KEYS
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


def test_runtime_binds_calibration_to_exact_model_hash(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture-model")
    artifact = CalibrationArtifact(
        schema_version=1,
        features_version=1,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        feature_names=FEATURE_KEYS,
        means=(0.0,) * len(FEATURE_KEYS),
        scales=(1.0,) * len(FEATURE_KEYS),
        intercept=10,
        coefficients=(0.0,) * len(FEATURE_KEYS),
        auto_threshold=.9,
        review_threshold=.65,
        auto_enabled=True,
        fit_positive_count=25,
        fit_negative_count=25,
        seed=3407,
        source_manifest_sha256="b" * 64,
        threshold_case_count=200,
        threshold_negative_count=100,
        threshold_false_positive_count=0,
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(artifact.to_dict()))
    runtime = AppRuntime(
        model_path=str(model),
        calibration_path=str(calibration),
        backend=FakeBackend(),
    )
    runtime.start()
    assert runtime.health()["calibrated"] is True
    assert runtime.health()["model_auto_post_enabled"] is True


def test_runtime_rejects_calibration_for_another_model(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture-model")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema_version": 1,
        "features_version": 1,
        "model_sha256": "a" * 64,
        "feature_names": list(FEATURE_KEYS),
        "means": [0] * len(FEATURE_KEYS),
        "scales": [1] * len(FEATURE_KEYS),
        "intercept": 10,
        "coefficients": [0] * len(FEATURE_KEYS),
        "auto_threshold": .9,
        "review_threshold": .65,
        "auto_enabled": True,
        "fit_positive_count": 25,
        "fit_negative_count": 25,
        "seed": 3407,
        "source_manifest_sha256": "b" * 64,
        "threshold_case_count": 200,
        "threshold_negative_count": 100,
        "threshold_false_positive_count": 0,
    }))
    runtime = AppRuntime(
        model_path=str(model),
        calibration_path=str(calibration),
        backend=FakeBackend(),
    )
    runtime.start()
    assert runtime.health()["calibrated"] is False
    assert "different model" in runtime.health()["calibration_error"]


def test_runtime_health_is_fail_closed_before_start():
    health = AppRuntime().health()
    assert health["calibrated"] is False
    assert health["model_auto_post_enabled"] is False
