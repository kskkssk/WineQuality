import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import subprocess
import logging


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


def load_model():
    subprocess.run(["dvc", "pull"])
    try:
        with open('model.pkl', 'rb') as f:
            model = joblib.load(f)
            return model
    except Exception:
        raise HTTPException(status_code=503, detail="Model not loaded")


model = load_model()


@app.get("/")
async def root():
    return "Hello"


@app.post('/predict', response_model=Result)
def predict(data: Wine):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = data.dict()
        df = pd.DataFrame({
            'fixed acidity': [data['fixed_acidity']],
            'volatile acidity': [data['volatile_acidity']],
            'citric acid': [data['citric_acid']],
            'residual sugar': [data['residual_sugar']],
            'chlorides': [data['chlorides']],
            'free sulfur dioxide': [data['free_sulfur_dioxide']],
            'total sulfur dioxide': [data['total_sulfur_dioxide']],
            'density': [data['density']],
            'pH': [data['pH']],
            'sulphates': [data['sulphates']],
            'alcohol': [data['alcohol']],
        })

        prediction = model.predict(df)
        return Result(predicted_quality=prediction)
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


is_alive = True
is_ready = True


@app.get("/liveness")
async def liveness():
    if is_alive:
        return {"status": "OK"}
    raise HTTPException(status_code=503, detail="Service not alive")


@app.get("/readiness")
async def readiness():
    try:
        if is_ready:
            return {"status": "OK"}
        else:
            return {"status": "NOT_READY"}
    except:
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get('/healthcheck')
def health():
    if model:
        return JSONResponse(content={'status': 'ok'}, status_code=200)
    if not model:
        return JSONResponse(content={'status': 'error', 'reason': 'model'}, status_code=503)
    else:
        return JSONResponse(content={'status': 'error', 'reason': 'unknown'}, status_code=500)


if __name__ == "__main__":
    try:
        load_model()
    except Exception as e:
        print(f"Model is not loaded: {str(e)}")
    uvicorn.run(app, host='127.0.0.1', port=8082)