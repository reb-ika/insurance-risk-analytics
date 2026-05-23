
"""
data_loader.py
--------------
Modular data loading and cleaning functions for insurance risk analytics.
"""

import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw insurance data from pipe-separated text file."""
    df = pd.read_csv(filepath, sep="|", low_memory=False)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean insurance dataset:
    - Drop columns with >90% missing values
    - Compute derived metrics (LossRatio, Margin)
    - Parse dates
    """
    # Drop columns with >90% missing
    threshold = 0.9
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >90% missing: {cols_to_drop}")

    # Compute derived metrics
    df['LossRatio'] = np.where(
        df['TotalPremium'] > 0,
        df['TotalClaims'] / df['TotalPremium'], 0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

    # Parse dates
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(
            df['TransactionMonth'], errors='coerce')

    print(f"Clean dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = load_raw_data("data/MachineLearningRating_v3.txt")
    df_clean = clean_data(df)
    print(df_clean.describe())
