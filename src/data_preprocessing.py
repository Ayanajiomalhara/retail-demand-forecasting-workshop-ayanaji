import pandas as pd
from src.config.config import DATA_RAW_PATH


def preprocess_run():
    # Load raw sales data
    df = pd.read_csv(DATA_RAW_PATH)

    # Convert date column into real date format
    df["date"] = pd.to_datetime(df["date"])

    # Remove empty rows
    df = df.dropna()

    return df
