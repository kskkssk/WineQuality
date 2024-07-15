import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from model_train import load_model

app = FastAPI()


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    class Config:
        orm_mode = True


class Result(BaseModel):
    predicted_quality: float

    class Config:
        orm_mode = True


@app.post('/predict', response_model=Result)
def predict(data: Wine):
    data = data.to_dict()
    model = load_model()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction


is_alive = True
is_ready = True

@app.get("/liveness")
async def liveness():
    if is_alive:
        return {"status": "OK"}
    raise HTTPException(status_code=503, detail="Service not alive")

@app.get("/readiness")
async def readiness():
    if is_ready:
        return {"status": "OK"}
    raise HTTPException(status_code=503, detail="Service not ready")