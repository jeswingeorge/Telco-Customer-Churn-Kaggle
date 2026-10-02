"""Load the raw Telco CSV, clean it, and write the cleaned parquet file.

Run from the repo root:  uv run python -m churn.dataset
"""

import pandas as pd

from churn.config import CLEAN_DATA_FILE, RAW_DATA_FILE


def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded from Kaggle."""
    return pd.read_csv(RAW_DATA_FILE)  # blank strings become NaN


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed cleaning steps that don't learn anything from the data (so no leakage)."""
    df = df.copy()
    ## Convert TotalCharges to numeric, filling the 11 NaNs with 0
    ## 11 blanks, all tenure == 0 (not yet billed) -> 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    ## Churn "Yes"/"No" -> 1/0 (sklearn metrics and XGBoost expect 1 = positive class)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # SeniorCitizen is the only binary column stored as 0/1; make it Yes/No like the others,
    # so it is treated as categorical (never scaled as a number) and the app shows a Yes/No choice.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def save(df: pd.DataFrame) -> None:
    """Write the cleaned table. Parquet keeps the dtypes."""
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_DATA_FILE, engine="pyarrow", index=False)


def main() -> None:
    df = clean(load_raw())
    save(df)
    print(f"Saved {df.shape[0]:,} rows x {df.shape[1]} cols to {CLEAN_DATA_FILE}")


if __name__ == "__main__":  # runs only when executed as a script, not when imported
    main()
