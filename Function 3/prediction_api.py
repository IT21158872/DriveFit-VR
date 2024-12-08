from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

with open('Models/future_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# FastAPI app
app = FastAPI()

class PredictionRequest(BaseModel):
    Level: str
    Engagement_Score: float
    Engagement_Time: float
    Predicted_Time: float

def preprocess_input(data):

    level_encoded = encoder.transform([[data.Level]]).toarray()

    input_data = np.hstack([level_encoded, [[data.Engagement_Score, data.Engagement_Time, data.Predicted_Time]]])

    input_data[:, -3:] = scaler.transform(input_data[:, -3:])
    return input_data


# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):

    input_data = preprocess_input(request)

    prediction = model.predict(input_data)[0]

    label_map = {0: "Needs Improvement", 1: "Good", 2: "Excellent"}
    return {"Prediction": label_map[prediction]}
