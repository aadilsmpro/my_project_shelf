"""
train_model.py
----------------
Model training utility for the Tennis Kinematic EPV Engine.
Allows training an XGBoost classifier using either synthetic fallback data
or a custom tracking dataset supplied via CSV.

Usage:
    - Processed Data:           python train_model.py
    - Custom CSV Dataset:       python train_model.py --data path/to/dataset.csv
    - Force Synthetic Data:     python train_model.py --data ""
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def train_and_save_model(
    data_path: str = "processed_epv_data.csv",
    output_path: str = "epv_xgboost_model.pkl",
):
    """Trains an XGBoost model and serializes the binary file to disk."""
    if data_path and data_path.strip():
        print(f"[INFO] Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

        # Validate required features
        required_cols = [
            "recovery_deficit_m",
            "exposed_area_m2",
            "deficit_x_exposed",
            "target",
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"[ERROR] Missing required columns in dataset: {missing}\n"
                f"Expected format must include: {required_cols}"
            )

        X = df[["recovery_deficit_m", "exposed_area_m2", "deficit_x_exposed"]]
        y = df["target"]
    else:
        print(
            "[INFO] No valid dataset path provided. Generating synthetic baseline data..."
        )
        np.random.seed(42)
        n_samples = 1000

        df = pd.DataFrame(
            {
                "recovery_deficit_m": np.random.uniform(0, 8, n_samples),
                "exposed_area_m2": np.random.uniform(0, 30, n_samples),
                "deficit_x_exposed": np.random.uniform(0, 240, n_samples),
            }
        )

        y = (
            (
                df["recovery_deficit_m"] * 0.4
                + df["exposed_area_m2"] * 0.05
                + np.random.normal(0, 0.5, n_samples)
            )
            > 2.0
        ).astype(int)

        X = df

    # Initialize XGBoost Classifier
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )

    print("[INFO] Fitting XGBoost Classifier...")
    model.fit(X, y)

    # Save artifact to disk
    joblib.dump(model, output_path)
    print(f"[SUCCESS] Model artifact successfully saved to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for Kinematic EPV Engine."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="processed_epv_data.csv",  # Changed default from None to processed_epv_data.csv
        help="Path to CSV dataset (default: processed_epv_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="epv_xgboost_model.pkl",
        help="Path to output saved model artifact file.",
    )

    args, unknown = parser.parse_known_args()
    train_and_save_model(data_path=args.data, output_path=args.output)
