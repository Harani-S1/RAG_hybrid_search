from fastapi.testclient import TestClient

from app.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200


def test_ask():
    with TestClient(app) as client:
        response = client.post(
            "/v1/ask",
            json={
                "question": "What is machine learning?"
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "question" in data
        assert "answer" in data

        assert data["question"] == "What is machine learning?"
        assert len(data["answer"]) > 0