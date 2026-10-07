from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"


def test_missing_features():
    response = client.post(
        "/predict",
        json={
            "features": {
                "hour_of_day": 12
            }
        },
    )

    assert response.status_code == 400


def test_health_endpoint_exists():
    response = client.get("/health")

    assert response.status_code in [200, 503]