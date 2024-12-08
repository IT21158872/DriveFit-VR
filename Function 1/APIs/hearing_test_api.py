from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# Trained model
with open('Function 1/Models/sound_localization_model.pkl', 'rb') as file:
    model = pickle.load(file)

# FastAPI app
app = FastAPI(
    title="Sound Localization Test Assessment API",
    description="API for assesing eyesight based on test parameters.",
    version="1.0",
)

# Inputs structure
class Inputs(BaseModel):
    accuracy: float
    response_time: float
    num_correct_sounds: int

# Output structure
class Output(BaseModel):
    prediction: str

# Label mapping
label_mapping = {0: "Bad", 1: "Good", 2: "Normal"}

# Home
@app.get("/")
def home():
    return {"message": "Welcome to the Sound Localization Test Assessment API!"}

# Prediction
@app.post("/predict", response_model=Output)
def predict(data: Inputs):
    try:

        input_data = pd.DataFrame([{
            "Accuracy (%)": data.accuracy,
            "ResponseTime (ms)": data.response_time,
            "NumCorrectSounds": data.num_correct_letters,
        }])
        
        prediction = model.predict(input_data)[0]
        prediction_label = label_mapping[prediction]
        
        return {"prediction": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
