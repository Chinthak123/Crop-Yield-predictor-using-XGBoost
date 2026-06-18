from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import pickle
import numpy as np

app = FastAPI()
base_dir = Path(__file__).resolve().parent

model = pickle.load(open('model.pkl', 'rb')) 


class PredictionRequest(BaseModel):
    rainfall: float
    temp: float

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(base_dir / "index.html", media_type="text/html")

@app.post("/predict")
async def get_prediction(data: PredictionRequest):
    
    features = np.array([[data.rainfall, data.temp]])
    
    prediction = model.predict(features)
    
    return {"yield": float(prediction[0])}