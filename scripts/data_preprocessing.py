"""
data_preprocessing.py
-----------------------------------
Loads the raw or cleaned Bank Marketing dataset, 
applies preprocessing transformations, and outputs a processed CSV 
ready for modeling.

Steps:
1. Load raw/cleaned CSV
2. Drop irrelevant or leaky columns
3. Encode categorical variables
4. Scale numeric variables
5. Save fully processed dataset to /data/processed_bank_marketing.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------
# CONFIG
# -------------------------
INPUT_PATH = Path("data") / "bank-full.csv"
OUTPUT_PATH = Path("data") / "bank-full-processed.csv"
TARGET_COL = "y"

EXCLUDE_COLS = [
    "duration",  # Call duration - only known after the call
    "id", "contact_date", "dataset_split",  # identifiers
    "poutcome",  # Previous campaign outcome - future information
    "pdays",  # Days since last contact - future information  
    "previous"  # Number of previous contacts - could be future info
]
RANDOM_STATE = 42


# -------------------------
# FUNCTIONS
# -------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path.name}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    exclude_cols = [col for col in EXCLUDE_COLS if col in df.columns]
    df = df.drop(columns=exclude_cols, errors="ignore")

    # Ensure target is last column and encode as 1/0
    y = df[TARGET_COL]
    y_encoded = (y == 'yes').astype(int)  # yes=1, no=0
    X = df.drop(columns=[TARGET_COL])

    # Separate types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # One-hot encode categorical vars
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    X_encoded = pd.DataFrame(
        encoder.fit_transform(X[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index,
    )

    # Scale numeric vars
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X[num_cols]),
        columns=num_cols,
        index=X.index,
    )

    # Combine processed features and encoded target
    X_processed = pd.concat([X_scaled, X_encoded], axis=1)
    processed_df = pd.concat([X_processed, y_encoded], axis=1)

    print(f"Processed dataset shape: {processed_df.shape}")
    return processed_df


def save_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")


def main():
    df = load_data(INPUT_PATH)
    processed_df = preprocess(df)
    save_data(processed_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
# loads data, generates and saves report
