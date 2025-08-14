from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Load the model
path = os.path.join("..", "models", "final_model.pkl")
model = joblib.load(path)

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


@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)
        return {"attrition_prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))