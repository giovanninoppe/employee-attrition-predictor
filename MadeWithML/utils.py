import os
import sys

import joblib
import pandas as pd

# Add the root directory to sys.path to allow importing modules from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging using the centralized logging configuration
import logging.config
from logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("employee_attrition")

def csv_to_df(csv_file):
    """
        Load a CSV file from the 'SourceFiles' directory into a pandas DataFrame.

        Parameters:
        - csv_file (str): Name of the CSV file to load.

        Returns:
        - pd.DataFrame: Loaded data as a DataFrame.

        Raises:
        - FileNotFoundError: If the specified file does not exist.
    """
    try:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        MODEL_PATH = os.path.join(BASE_DIR, 'SourceFiles', csv_file)

        data = pd.read_csv(MODEL_PATH)
        logger.info(f"Successfully loaded {csv_file} with shape {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File {MODEL_PATH} not found")
        raise


def get_model(name_model: str):
    """
    Load a trained model from the 'models' directory using joblib.

    Parameters:
    - name_model (str): Filename of the model to load.

    Returns:
    - Any: Loaded model object.

    Raises:
    - FileNotFoundError: If the model file does not exist.
    - Exception: For any other errors during model loading.
    """

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', name_model)

    try:
        path = os.path.join(MODEL_PATH)
        model = joblib.load(path)
        logger.info(f"Successfully loaded model from {path}")
        return model
    except FileNotFoundError:
        logger.error(f"File {MODEL_PATH} not found")
        raise
    except Exception as e:
        logger.error(e)
        raise


def get_path_to_model(name_model: str):
    """
        Construct the full path to a model file in the 'models' directory.

        Parameters:
        - name_model (str): Filename of the model.

        Returns:
        - str: Full path to the model file.

        Raises:
        - Exception: If path construction fails.
    """
    try:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', name_model)
        logger.info(f"Constructed model path: {MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        logger.error(f"Error constructing model path for {name_model}: {e}")
        raise


# Example usage for testing when run as a standalone script
if __name__ == "__main__":
    df_test = csv_to_df("source.csv")