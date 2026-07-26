from bursa.inference.backend import FakeBackend
from bursa_eval.harness.adtc import load_adtc, score_adtc, regression_delta


def test_score_adtc_exact_match(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id":"q1","prompt":"2+2?","expected":"4"}\n'
                 '{"id":"q2","prompt":"cap of France?","expected":"Paris"}\n')
    cases = load_adtc(str(p))
    backend = FakeBackend(chat_response=lambda prompt: "4" if "2+2" in prompt else "Paris")
    res = score_adtc(cases, backend, label="proxy")
    assert res["label"] == "proxy"
    assert res["accuracy"] == 1.0
    assert res["per_id"] == {"q1": True, "q2": True}


def test_score_adtc_partial(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id":"q1","prompt":"2+2?","expected":"4"}\n'
                 '{"id":"q2","prompt":"cap of France?","expected":"Paris"}\n')
    cases = load_adtc(str(p))
    backend = FakeBackend(chat_response=lambda prompt: "4" if "2+2" in prompt else "London")
    res = score_adtc(cases, backend, label="official")
    assert res["accuracy"] == 0.5


def test_regression_delta_is_relative_only():
    pre = {"accuracy": 0.80}
    post = {"accuracy": 0.79}
    assert abs(regression_delta(pre, post) - (-0.01)) < 1e-9
