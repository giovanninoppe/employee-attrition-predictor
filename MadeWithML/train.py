import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import joblib
import os

from preprocess import build_preprocessor
import utils as utils

df = utils.csv_to_df("source.csv")
X = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])
y = df['Attrition'].map({'Yes': 1, 'No': 0})

numerical = X.select_dtypes(include=[np.number]).columns
categorical = X.select_dtypes(exclude=[np.number]).columns

preprocessor = build_preprocessor(numerical, categorical)

pipeline = Pipeline([('preprocessor', preprocessor),
                     ('classifier', LogisticRegression(
                         solver='liblinear', penalty='l1'
                     ))])

param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__max_iter": [100, 500]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(f1_score, pos_label=1))
grid.fit(X, y)

path = os.path.join("..", "models", "final_model.pkl")

joblib.dump(grid.best_estimator_, path)
