import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException

# Add the root directory to sys.path so that modules in sibling directories can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model loading utility from the MadeWithML package
from MadeWithML.utils import get_model

from pydantic import BaseModel
from typing import List

# Initialize the FastAPI application
app = FastAPI()

# Load the trained model from the specified path
model = get_model("final_model.pkl")


# Define the input data schema using Pydantic BaseModel
class InputData(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    Over18: str
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


# Define an endpoint for single prediction
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        # Make prediction using the loaded model
        prediction = model.predict(input_df)
        # Return the prediction result
        return {"attrition_prediction": prediction.tolist()}
    except Exception as e:
        # Raise HTTPException if prediction fails
        raise HTTPException(status_code=400, detail=str(e))


# Define an endpoint for batch prediction
@app.post("/predict_batch")
def predict_batch(data: List[InputData], batch_size: int = 0):
    try:
        # Convert list of InputData objects to list of dictionaries
        input_dicts = [item.model_dump() for item in data]
        predictions = []

        # If batch_size is specified, process in batches
        if batch_size > 0:
            for i in range(0, len(input_dicts), batch_size):
                batch = input_dicts[i:i + batch_size]
                batch_df = pd.DataFrame(batch)
                batch_pred = model.predict(batch_df)
                predictions.extend(batch_pred.tolist())
        else:
            # Process all data at once
            input_df = pd.DataFrame(input_dicts)
            predictions = model.predict(input_df).tolist()

        # Return the batch prediction results
        return {"predictions": predictions}
    except Exception as e:
        # Raise HTTPException if batch prediction fails
        raise HTTPException(status_code=400, detail=str(e))


# Define an endpoint to return model metrics
@app.get("/metrics")
def get_metrics():
    # Return static metrics for demonstration purposes
    metrics = {
        "model_name": "final_model.pkl",
        "accuracy": 0.87,
        "latency_example_prediction_seconds": 0.05,
        "latency_example_batch_prediction_seconds": 0.12,
        "version": "1.0.0"
    }
    return metrics
