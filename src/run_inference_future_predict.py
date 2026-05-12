from src.feature_engineering import add_features
from src.config.config import INFERENCE_OUTPUT_PATH
import pandas as pd
import joblib
import os


def run_inference_future_predict(
    model_path: str,
    input_path: str,
    output_path: str = INFERENCE_OUTPUT_PATH
):
    # Load future data
    df_future = pd.read_csv(input_path)

    # Convert date column into real date format
    df_future["date"] = pd.to_datetime(df_future["date"])

    # Remove empty rows
    df_future = df_future.dropna()

    # Add same features used during training
    df_future = add_features(df_future)

    # Load saved label encoders
    store_le_path = "models/store_le.pkl"
    item_le_path = "models/item_le.pkl"

    if os.path.exists(store_le_path) and os.path.exists(item_le_path):
        store_le = joblib.load(store_le_path)
        item_le = joblib.load(item_le_path)

        df_future["store_id"] = store_le.transform(df_future["store_id"])
        df_future["item_id"] = item_le.transform(df_future["item_id"])
    else:
        raise FileNotFoundError("Label encoders not found. Run training first.")

    # Remove date column before prediction
    cols_to_drop = ["date"]

    if "sales_qty" in df_future.columns:
        cols_to_drop.append("sales_qty")

    X_future = df_future.drop(columns=cols_to_drop)

    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)

    # Predict future sales
    df_future["sales_qty_pred"] = model.predict(X_future)

    # Create output folder if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save predictions
    df_future.to_csv(output_path, index=False)

    print(f"Future predictions saved to {output_path}")
