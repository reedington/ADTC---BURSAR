from fastapi.testclient import TestClient

from bursa.web import create_app


def test_dashboard_is_runnable_and_seeded(tmp_path):
    with TestClient(create_app(tmp_path / "demo.db")) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Bursa" in response.text
        assert "STU-1042" in response.text
        assert client.get("/health").json()["offline"] is True


def test_exact_demo_transaction_reconciles_through_real_core(tmp_path):
    with TestClient(create_app(tmp_path / "demo.db")) as client:
        response = client.post("/reconcile/TXN-NIPDEMO001", follow_redirects=False)
        assert response.status_code == 303
        page = client.get("/")
        assert "status-auto" in page.text


def test_csv_upload_rejects_non_csv(tmp_path):
    with TestClient(create_app(tmp_path / "demo.db")) as client:
        response = client.post(
            "/import/students",
            files={"file": ("students.txt", b"student_id,name\nS1,Ada\n", "text/plain")},
        )
        assert response.status_code == 415
