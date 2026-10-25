"""
Train a Random Forest model for AQI prediction.
Inputs: pm2_5, pm10, co, no2, so2, o3, temperature, humidity
Output: AQI value (continuous)

Run once:  python train_model.py
Produces: model/aqi_model.pkl

If the model file already exists, training is skipped.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "air_quality.csv")

FEATURE_COLS = ["pm2_5", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity"]
TARGET_COL = "aqi"


def train():
    # ------------------------------------------------------------------
    # Skip if model already exists
    # ------------------------------------------------------------------
    if os.path.isfile(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH} -- skipping training.")
        print("     Delete the file and re-run if you want to retrain.")
        return

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if os.path.isfile(DATA_PATH):
        print(f"[DATA] Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"[WARN] Dataset not found at {DATA_PATH}. Generating synthetic data...")
        df = _generate_synthetic_data()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"       Saved synthetic dataset to {DATA_PATH}")

    # Validate columns
    for col in FEATURE_COLS + [TARGET_COL]:
        if col not in df.columns:
            print(f"[ERROR] Missing column '{col}' in dataset. Aborting.")
            sys.exit(1)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print(f"   Samples: {len(df)}")

    # ------------------------------------------------------------------
    # Train / Evaluate
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n[OK] Model trained successfully!")
    print(f"   MAE : {mae:.2f}")
    print(f"   R²  : {r2:.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[SAVED] Model saved to {MODEL_PATH}")


def _generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate realistic synthetic environmental data as a fallback."""
    np.random.seed(42)

    pm2_5 = np.random.uniform(5, 400, n_samples)
    pm10 = pm2_5 * np.random.uniform(1.0, 2.5, n_samples)
    co = np.random.uniform(0.1, 15, n_samples)
    no2 = np.random.uniform(5, 200, n_samples)
    so2 = np.random.uniform(2, 150, n_samples)
    o3 = np.random.uniform(10, 250, n_samples)
    temperature = np.random.uniform(-5, 45, n_samples)
    humidity = np.random.uniform(10, 100, n_samples)

    aqi = (
        0.30 * pm2_5
        + 0.15 * (pm10 / 2.0)
        + 0.10 * (co * 20)
        + 0.12 * no2
        + 0.08 * so2
        + 0.10 * o3
        + 0.05 * np.abs(temperature - 25)
        + 0.10 * np.abs(humidity - 50)
    )
    aqi += np.random.normal(0, 8, n_samples)
    aqi = np.clip(aqi, 0, 500)

    return pd.DataFrame({
        "pm2_5": np.round(pm2_5, 1),
        "pm10": np.round(pm10, 1),
        "co": np.round(co, 2),
        "no2": np.round(no2, 1),
        "so2": np.round(so2, 1),
        "o3": np.round(o3, 1),
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "aqi": np.round(aqi, 1),
    })


if __name__ == "__main__":
    train()
