# employee-attrition-predictor
Predict employee attrition using machine learning. Includes data exploration, model selection (Logistic Regression, KNN, Decision Tree, Random Forest), hyperparameter tuning, and deployment-ready code with FastAPI and Docker.

## Features
- Exploratory data analysis on HR datasets
- Preprocessing pipeline for numerical and categorical features
- Model training with hyperparameter tuning using GridSearchCV
- Logging of training process and error handling
- REST API for single and batch predictions via FastAPI
- Dockerfile for containerization
- Unit tests for model validation

## Installation Guidelines
### 1. Clone the repository
git clone https://github.com/your-org/employee-attrition-predictor.git
cd employee-attrition-predictor

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the API locally:
uvicorn app.main:app --reload

### 4. Or use Docker:
docker build -t attrition-api .
docker run -p 8000:8000 attrition-api

## Examples
### 1. Single prediction:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"Age": 35, "BusinessTravel": "Travel_Rarely", ...}'

### 2. Batch predition:
curl -X POST http://localhost:8000/predict_batch -H "Content-Type: application/json" -d '[{...}, {...}]'

## Project Struture
.

├── app/

│   ├── main.py              # FastAPI endpoints

│   ├── preprocess.py        # Preprocessing pipeline

│   ├── train.py             # Model training

│   ├── test_model.py        # Unit test

│   ├── utils.py             # CSV/model helpers

│   └── logging_config.py    # Logging setup

├── models/

│   └── final_model.pkl      # Trained model

├── SourceFiles/

│   └── source.csv           # HR dataset

├── requirements.txt

├── Dockerfile

└── README.md

## Model Info
- Model: Logistic Regression (L1 penalty)
- Hyperparameters: C=[0.1, 1, 10], max_iter=[100, 500]
- Scoring: F1-score
- Accuracy: 87%
- Latency: 0.05s (single), 0.12s (batch)

## Contribution
- Kaggle
- Cursus Cloud for AI
- Cursus Python
- Cursus Machine Learning
