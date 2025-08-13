from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI()

path = os.path.join("..", "models", "final_model.pkl")
model = joblib.load(path)

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return {"attrition_prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))