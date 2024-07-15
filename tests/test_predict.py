from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

response = client.post("/predict", json={
        "fixed_acidity": 8.0,
        "volatile_acidity": 0.5,
        "citric_acid": 0.3,
        "residual_sugar": 10.0,
        "chlorides": 0.08,
        "free_sulfur_dioxide": 30.0,
        "total_sulfur_dioxide": 120.0,
        "density": 0.996,
        "pH": 3.4,
        "sulphates": 0.6,
        "alcohol": 11.5
        }
    )
assert response.status_code == 200
assert "predicted_quality" in response.json()