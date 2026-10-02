"""Engineered features and the preprocessing ColumnTransformer."""

import numpy as np
import pandas as pd

from churn.config import ADDON_COLS, NO_INTERNET_COLS, TENURE_BINS, TENURE_LABELS

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


def add_num_services(X: pd.DataFrame) -> pd.DataFrame:
    """Count of add-on services the customer has (0-5).

    Why: churn falls from 55% (internet customers, 0 add-ons) to 5% (all 5).
    Caveat: an exact sum of the 5 binaries, so LR gets the count OR the binaries (decided by CV).
    No-internet customers are also 0; InternetService separates them in the model.
    """
    X = X.copy()
    X["num_services"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
    return X


def add_tenure_group(X: pd.DataFrame) -> pd.DataFrame:
    """Band tenure into 0-6 / 7-12 / 13-24 / 25-48 / 49-72 months.

    Why: tenure is ~linear in log-odds except a steep first-6-months kink; bands let
    Logistic Regression model that kink. Fixed edges from config → stateless, no leakage.
    """
    X = X.copy()
    X["tenure_group"] = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    X["tenure_group"] = X["tenure_group"].astype(str)
    return X


def add_has_family(X: pd.DataFrame) -> pd.DataFrame:
    """'Yes' if the customer has a Partner or Dependents, else 'No'.

    Why: both lower churn. A CV candidate only: EDA suggests merging them loses information.
    """
    X = X.copy()
    X["has_family"] = np.where((X["Partner"] == "Yes") | (X["Dependents"] == "Yes"), "Yes", "No")
    return X


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered columns. The single entry point the Pipeline will call.

    Always adds all three; build_preprocessor's column lists decide which ones a model uses,
    so each can be switched on/off and compared with CV in the baseline notebook.
    """
    X = add_tenure_group(X)
    X = add_num_services(X)
    X = add_has_family(X)
    return X

