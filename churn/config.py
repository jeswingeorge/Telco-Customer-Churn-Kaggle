"""Project-wide paths and constants. Import these instead of hard-coding paths."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Final cleaned table written by churn.dataset; the input for modelling (all columns kept).
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "telco_churn_clean.parquet"

RANDOM_STATE = 42
TARGET = "Churn"
ID_COL = "customerID"

# Columns left out of the model, with the reason (evidence in notebooks/2_bi_multivariate_analysis.ipynb).
# Kept in the cleaned data; the pipeline simply never selects them, and the Streamlit app doesn't ask for them.
DROP_COLS = {
    "customerID": "Identifier, unique per row; no predictive value.",
    "TotalCharges": "~ tenure x MonthlyCharges; collinear with them; confirmed with CV 2026-10-10: adding it lowered LR ROC-AUC.",
    "gender": "No link to churn: Cramer's V = 0.000, p = 0.47, both genders on the 26.5% baseline.",
    "PhoneService": "Fully contained in MultipleLines ('No phone service' level); U(PhoneService | MultipleLines) = 1.0.",
    "StreamingMovies": "Adds no churn signal beyond StreamingTV (U(Churn) 0.0011 -> 0.0017 among internet customers).",
}

# Add-on service columns whose "No internet service" level is exactly InternetService == "No".
# features.collapse_no_internet maps that level to "No" so the information lives only in InternetService.
NO_INTERNET_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV"]

# ---- Engineered features (built by churn.features.add_features) ----
# Tenure bands, the same as the EDA. Fixed edges (not quantiles), so the step learns nothing → no leakage.
# Tenure is limited to 0-72 months (dataset range, spec decision 2026-10-02); the app enforces it.
TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]

# num_services counts "Yes" across these add-on columns.
ADDON_COLS = NO_INTERNET_COLS

ENGINEERED_FEATURES = ["tenure_group", "num_services", "has_family"]

# ---- Model feature lists (used by churn.features.build_preprocessor) ----
# Everything not listed here (DROP_COLS, the target) is dropped by remainder="drop".
# Feature decisions (phase 5, CV 2026-10-10): has_family, tenure_group, num_services left out; see 05_modelling.md.
NUM_COLS = ["tenure", "MonthlyCharges"]
CAT_COLS = ['SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'Contract', 'PaperlessBilling', 'PaymentMethod']



### Modelling constants 
N_SPLITS = 5
SCORING_METRICS = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

