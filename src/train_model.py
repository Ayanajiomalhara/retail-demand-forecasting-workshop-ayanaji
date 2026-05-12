from sklearn.ensemble import RandomForestRegressor
import os
import joblib
from src.config.config import MODEL_PATH


def train_model(X_train, y_train, model_path=None):
    # If no model path is given, use the default path from config.py
    if model_path is None:
        model_path = MODEL_PATH

    # Create the machine learning model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Create models folder if it does not exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save trained model
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    return model
