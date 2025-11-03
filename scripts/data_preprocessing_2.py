"""
data_preprocessing_2.py
-----------------------------------
Loads the original Bank Marketing dataset (semicolon-delimited),
applies feature engineering, encoding, and scaling,
and saves a processed CSV ready for modeling (v2).

Output: data/bank-full-processed-v2.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------
# CONFIG
# -------------------------
# Resolve paths relative to repo root (parent of scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "bank-full.csv"
OUTPUT_PATH = BASE_DIR / "data" / "bank-full-processed-v2.csv"
TARGET_COL = "y"


# -------------------------
# FUNCTIONS
# -------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load raw dataset."""
    df = pd.read_csv(path, sep=';')
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path.name}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering, encoding, and scaling."""
    # Ensure binary target 1/0
    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip().str.lower() == 'yes').astype(int)
    else:
        raise ValueError(f"Target column '{TARGET_COL}' not found in input data")

    # Feature Engineering
    # 1) was_contacted_before from pdays (1 if pdays != 999 else 0), then drop pdays
    if 'pdays' in df.columns:
        df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
        df = df.drop(columns=['pdays'])

    # 2) previous -> previous_contacts, drop original
    if 'previous' in df.columns:
        df['previous_contacts'] = df['previous']
        df = df.drop(columns=['previous'])

    # 3) balance_log using log1p; clip negatives to 0 before log
    if 'balance' in df.columns:
        balance_nonneg = np.clip(df['balance'].astype(float), a_min=0, a_max=None)
        df['balance_log'] = np.log1p(balance_nonneg)

    # 4) campaign_capped (cap original campaign at 5)
    if 'campaign' in df.columns:
        df['campaign_capped'] = df['campaign'].clip(upper=5)

    # 5) age_group: 5 quantile bins (labels 0..4)
    if 'age' in df.columns:
        try:
            df['age_group'] = pd.qcut(df['age'], q=5, labels=False, duplicates='drop')
        except Exception:
            # Fallback if not enough unique values
            df['age_group'] = pd.cut(df['age'], bins=5, labels=False, include_lowest=True)

    # 6) Ordinal encode education with predefined mapping
    #    unknown as mid-level
    if 'education' in df.columns:
        edu_map = {
            'primary': 1,
            'secondary': 2,
            'unknown': 2,   # treat as mid-level
            'tertiary': 3
        }
        df['education_ord'] = df['education'].astype(str).str.strip().str.lower().map(edu_map).fillna(2).astype(int)
        df = df.drop(columns=['education'])

    # 7) Interactions: job_marital and month_contact
    if 'job' in df.columns and 'marital' in df.columns:
        df['job_marital'] = df['job'].astype(str) + '_' + df['marital'].astype(str)
    if 'month' in df.columns and 'contact' in df.columns:
        df['month_contact'] = df['month'].astype(str) + '_' + df['contact'].astype(str)

    # Separate target from features
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # Encoding & Scaling
    # One-hot encode categoricals
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        X_cat = pd.DataFrame(
            encoder.fit_transform(X[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X.index,
        )
    else:
        X_cat = pd.DataFrame(index=X.index)

    # Scale numeric columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_num = pd.DataFrame(
            scaler.fit_transform(X[num_cols]),
            columns=num_cols,
            index=X.index,
        )
    else:
        X_num = pd.DataFrame(index=X.index)

    # Combine features and append target as last column
    X_processed = pd.concat([X_num, X_cat], axis=1)
    processed_df = pd.concat([X_processed, y], axis=1)
    print(f"Processed dataset shape (v2): {processed_df.shape}")
    return processed_df


def save_data(df: pd.DataFrame, output_path: Path):
    """Save processed dataset."""
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")


def main():
    df_raw = load_data(INPUT_PATH)
    df_processed = preprocess(df_raw)
    save_data(df_processed, OUTPUT_PATH)


if __name__ == "__main__":
    main()
