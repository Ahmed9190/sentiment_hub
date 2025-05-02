from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_predict_positive():
    response = client.post("/predict/", json={"text": "I love this movie!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_negative():
    response = client.post("/predict/", json={"text": "This was a terrible film."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_empty_text():
    response = client.post("/predict/", json={"text": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
