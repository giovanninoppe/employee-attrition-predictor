import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Add the root directory to sys.path to allow importing modules from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logging configuration and initialize logger
import logging.config
from logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("employee_attrition")

# Import preprocessing pipeline builder and utility functions
from preprocess import build_preprocessor
import utils as utils


def train_model(sourcefile: str):
    """
        Train a logistic regression model using the provided source CSV file.
        Includes preprocessing, hyperparameter tuning, and model saving.
    """

    # Load the dataset from CSV into a DataFrame
    df = utils.csv_to_df(sourcefile)

    # Separate features (X) and target variable (y)
    # Drop irrelevant columns from features
    X = df.drop(columns=['Attrition', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])
    # Convert target variable 'Attrition' from Yes/No to 1/0
    y = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Identify numerical and categorical columns
    numerical = X.select_dtypes(include=[np.number]).columns
    categorical = X.select_dtypes(exclude=[np.number]).columns

    # Build preprocessing pipeline for numerical and categorical features
    preprocessor = build_preprocessor(numerical, categorical)

    # Create a pipeline with preprocessing and logistic regression classifier
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier', LogisticRegression(
                             solver='liblinear', penalty='l1'
                         ))])

    # Define hyperparameter grid for tuning
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__max_iter": [100, 500]
    }

    # Log the start of training
    logger.info('Training model...')

    # Perform grid search with cross-validation using F1 score as evaluation metric
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(f1_score, pos_label=1))
    grid.fit(X, y)

    # Get the path to save the trained model
    path = utils.get_path_to_model("final_model.pkl")

    # Save the best model from grid search to disk
    joblib.dump(grid.best_estimator_, path)

    # Log the location of the saved model
    logger.info('Model saved in %s', path)


# Execute training when script is run directly
if __name__ == '__main__':
    train_model("source.csv")
