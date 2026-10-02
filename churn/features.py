"""Engineered features and the preprocessing ColumnTransformer."""

import pandas as pd

from churn.config import NO_INTERNET_COLS

NO_INTERNET = "No internet service"


def collapse_no_internet(X: pd.DataFrame) -> pd.DataFrame:
    """Replace "No internet service" with "No" in the add-on service columns.

    Why: that level is identical to InternetService == "No", so after one-hot encoding the
    same fact appears as six identical dummies. That is perfect multicollinearity for
    Logistic Regression and splits importance six ways in trees/SHAP. InternetService keeps
    the information, so nothing is lost.

    Stateless (learns nothing from the data), so it can't leak. Used inside the Pipeline via
    FunctionTransformer, so training and the Streamlit app apply the same step. It's a
    module-level function rather than a lambda so joblib can save it with the pipeline.
    """
    X = X.copy()
    cols = [c for c in NO_INTERNET_COLS if c in X.columns]
    X[cols] = X[cols].replace(NO_INTERNET, "No")
    return X
