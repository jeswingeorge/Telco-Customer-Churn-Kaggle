"""Load the saved pipeline and score customers using the saved threshold.

Run from the repo root:
  uv run python -m churn.modeling.predict                                  # scores the raw Kaggle CSV
  uv run python -m churn.modeling.predict --input new.csv --output scored.csv

The Streamlit app imports load_model / predict / predict_one from here, so the app and this
script score customers exactly the same way. There is no feature logic in this file: the
saved pipeline (collapse -> features -> preprocessor -> model) does all of it. This file only
checks the input is something the model can score, then applies the saved threshold.
"""

import argparse
import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn import config
from churn.features import collapse_no_internet

MODEL_PATH = config.MODELS_DIR / "churn_model.joblib"
METADATA_PATH = config.MODELS_DIR / "metadata.json"
TENURE_MIN, TENURE_MAX = 0, 72  # the training range (decision 2026-10-02); bins stop at 72
HIGH_RISK, LOW_RISK = "High risk", "Low risk"


@lru_cache(maxsize=1)  # load from disk once per process, not once per prediction
def load_model(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    """Return (fitted pipeline, metadata dict). Run `python -m churn.modeling.train` first."""
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"{model_path.name} / {metadata_path.name} not found in {model_path.parent}. "
            "Train the model first: uv run python -m churn.modeling.train"
        )
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trained_with = metadata.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        # joblib files aren't guaranteed to work across scikit-learn versions
        warnings.warn(
            f"Model saved with scikit-learn {trained_with}, running {sklearn.__version__}.",
            stacklevel=2,
        )
    return model, metadata


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Accept the raw Kaggle format too: SeniorCitizen 0/1 -> No/Yes (as in dataset.clean)."""
    X = X.copy()
    if "SeniorCitizen" in X.columns and pd.api.types.is_numeric_dtype(X["SeniorCitizen"]):
        X["SeniorCitizen"] = X["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return X


def validate(X: pd.DataFrame, model, metadata: dict) -> pd.Series:
    """One reason string per row; "" means the row can be scored.

    Why: OneHotEncoder(handle_unknown="ignore") would silently score a typo like "Fibre" as
    if the customer had none of the known levels, and pd.cut gives NaN for tenure > 72.
    So rows the model wasn't trained for are flagged instead of getting a confident score.
    """
    missing = [c for c in metadata["input_columns"] if c not in X.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    problems = pd.Series([[] for _ in range(len(X))], index=X.index)

    def flag(mask, message):
        for i in mask[mask].index:
            problems[i].append(message)

    for col in config.NUM_COLS:
        values = pd.to_numeric(X[col], errors="coerce")
        flag(values.isna(), f"{col} is missing or not a number")
    tenure = pd.to_numeric(X["tenure"], errors="coerce")
    flag(tenure.notna() & ~tenure.between(TENURE_MIN, TENURE_MAX),
         f"tenure outside {TENURE_MIN}-{TENURE_MAX} months (the training range)")
    charges = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
    flag(charges < 0, "MonthlyCharges is negative")

    # Allowed levels = the categories the fitted encoder learned (after the same collapse
    # step the pipeline applies), so this check can never disagree with the model.
    encoder = model.named_steps["prep"].named_transformers_["cat"]
    known = dict(zip(encoder.feature_names_in_, encoder.categories_))
    collapsed = collapse_no_internet(X)
    for col, levels in known.items():
        if col not in metadata["input_columns"]:
            continue  # engineered (tenure_group): built by the pipeline, not supplied
        flag(~collapsed[col].isin(levels), f"unknown {col} value")

    return problems.map("; ".join)


def predict(X: pd.DataFrame, model=None, metadata: dict | None = None) -> pd.DataFrame:
    """Score a table of customers (raw columns). Invalid rows are flagged, not scored.

    Returns a DataFrame with the same index:
      churn_probability  P(churn) from the calibrated, unweighted model (NaN if not scored)
      churn_flag         probability >= the saved threshold (contact this customer)
      risk_label         "High risk" / "Low risk" (or "Not scored")
      problem            why a row wasn't scored ("" if it was)
    """
    if model is None or metadata is None:
        model, metadata = load_model()
    X = _prepare(X)
    problem = validate(X, model, metadata)
    ok = problem == ""

    proba = pd.Series(np.nan, index=X.index)
    if ok.any():
        proba[ok] = model.predict_proba(X.loc[ok, metadata["input_columns"]])[:, 1]
    flagged = proba >= metadata["threshold"]  # NaN >= t is False

    return pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_flag": flagged.where(ok),  # None for rows that weren't scored
        "risk_label": np.where(~ok, "Not scored", np.where(flagged, HIGH_RISK, LOW_RISK)),
        "problem": problem,
    }, index=X.index)


def predict_one(customer: dict, model=None, metadata: dict | None = None) -> dict:
    """Score one customer given as {column: value} (the Streamlit form)."""
    result = predict(pd.DataFrame([customer]), model, metadata).iloc[0]
    if result["problem"]:
        raise ValueError(result["problem"])
    return {
        "churn_probability": float(result["churn_probability"]),
        "churn_flag": bool(result["churn_flag"]),
        "risk_label": result["risk_label"],
    }


def main(input_path: Path, output_path: Path | None) -> None:
    model, metadata = load_model()
    df = pd.read_csv(input_path)
    scored = predict(df, model, metadata)
    if config.ID_COL in df.columns:  # keep the customer ID so the scores can be joined back
        scored.insert(0, config.ID_COL, df[config.ID_COL])

    n_ok = int((scored["problem"] == "").sum())
    n_high = int((scored["risk_label"] == HIGH_RISK).sum())
    print(f"Model: {metadata['model']}, threshold {metadata['threshold']:.3f}")
    print(f"Scored {n_ok:,} of {len(df):,} rows; {n_high:,} high risk "
          f"({n_high / max(n_ok, 1):.1%} of scored); {len(df) - n_ok:,} not scored")
    print(scored.head(10).to_string())
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":  # runs only when executed as a script, not when imported
    parser = argparse.ArgumentParser(description="Score customers with the saved churn model.")
    parser.add_argument("--input", type=Path, default=config.RAW_DATA_FILE,
                        help="CSV with the raw input columns (default: the raw Kaggle file)")
    parser.add_argument("--output", type=Path, default=None, help="where to save the scores")
    args = parser.parse_args()
    main(args.input, args.output)
