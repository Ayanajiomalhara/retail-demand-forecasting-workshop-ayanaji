import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


def add_features(df):
    # Create day_of_week from date
    # Monday = 0, Tuesday = 1, ..., Sunday = 6
    df["day_of_week"] = df["date"].dt.dayofweek

    # Create weekend feature
    weekend_days = [5, 6]  # 5 = Saturday, 6 = Sunday
    df["is_weekend"] = df["day_of_week"].isin(weekend_days).astype(int)

    return df


def prepare_features(df, target="sales_qty", test_size=0.2, random_state=42):
    # Separate target column
    y = df[target]

    # Remove target and date from input features
    X = df.drop(columns=[target, "date"])

    # Convert store_id text into numbers
    store_le = LabelEncoder()
    X["store_id"] = store_le.fit_transform(X["store_id"])

    # Convert item_id text into numbers
    item_le = LabelEncoder()
    X["item_id"] = item_le.fit_transform(X["item_id"])

    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Save encoders for future prediction
    joblib.dump(store_le, "models/store_le.pkl")
    joblib.dump(item_le, "models/item_le.pkl")

    # Split data into train and test
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
