import pandas as pd
import logging
import os

def csv_to_df(csv_file):
    try:
        # Ga één map omhoog en open SourceFiles
        path = os.path.join("..", "SourceFiles", csv_file)
        data = pd.read_csv(path)
        logging.info(f"Successfully loaded {csv_file} with shape {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"File {path} not found")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df_test = csv_to_df("source.csv")