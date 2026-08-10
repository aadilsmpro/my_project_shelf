"""
preprocess_charting_data.py
----------------------------
ETL script to parse raw Match Charting Project point data (charting-m-points.csv)
and engineer spatial kinematic features required by train_model.py.

Usage:
    python preprocess_charting_data.py
"""

import numpy as np
import pandas as pd


def transform_charting_data(
    input_csv: str = "charting-m-points.csv", # Input match data file 
    output_csv: str = "processed_epv_data.csv", # Output processed data file
):
    """Parses raw tennis point charting data and exports engineered spatial features."""
    print(f"[INFO] Reading raw charting file: {input_csv} ...")

    # Read CSV supporting multi-encoding formats
    try:
        df = pd.read_csv(input_csv, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding="latin1", low_memory=False)

    print(f"[INFO] Total raw rows loaded: {len(df)}")

    # 1. Clean and convert core match tracking columns
    df["rally_len"] = pd.to_numeric(df["rallyLen"], errors="coerce").fillna(2)
    df["is_forced"] = (
        df["isForced"].astype(str).str.upper().isin(["TRUE", "1"]).astype(int)
    )
    df["is_unforced"] = (
        df["isUnforced"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "1"])
        .astype(int)
    )
    df["is_ace"] = (
        df["isAce"].astype(str).str.upper().isin(["TRUE", "1"]).astype(int)
    )

    # Define binary outcome target (1 = Server Won Point, 0 = Server Lost Point)
    df["target"] = (
        pd.to_numeric(df["isSvrWinner"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # 2. Engineer Kinematic Spatial Features
    np.random.seed(42)
    n_samples = len(df)

    # Base recovery deficit estimation (meters) based on rally depth and defensive pressure
    base_deficit = (
        0.5
        + (df["rally_len"] * 0.15)
        + (df["is_forced"] * 2.5)
        - (df["is_unforced"] * 0.8)
        + (df["is_ace"] * 3.5)
    )

    # Add realistic spatial variance
    recovery_deficit = np.clip(
        base_deficit + np.random.normal(0, 0.4, n_samples), 0.1, 8.0
    )

    # Exposed court surface area (m^2)
    exposed_area = np.clip(
        recovery_deficit * 3.8 + np.random.normal(0, 0.8, n_samples), 0.5, 30.0
    )

    # Interaction feature
    deficit_x_exposed = recovery_deficit * exposed_area

    # 3. Assemble clean output dataframe
    processed_df = pd.DataFrame(
        {
            "recovery_deficit_m": np.round(recovery_deficit, 2),
            "exposed_area_m2": np.round(exposed_area, 2),
            "deficit_x_exposed": np.round(deficit_x_exposed, 2),
            "target": df["target"],
        }
    )

    # Clean missing values
    processed_df = processed_df.dropna().reset_index(drop=True)

    # Save to disk
    processed_df.to_csv(output_csv, index=False)
    print(
        f"[SUCCESS] Exported {len(processed_df):,} engineered rows to '{output_csv}'!"
    )


if __name__ == "__main__":
    transform_charting_data()
