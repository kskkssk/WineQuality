from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_readiness():
    response = client.get("/readiness")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
