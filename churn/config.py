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

RANDOM_STATE = 42
TARGET = "Churn"
ID_COL = "customerID"

# Columns left out of the model, with the reason (evidence in notebooks/2_bi_multivariate_analysis.ipynb).
# Kept in the cleaned data; the pipeline simply never selects them, and the Streamlit app doesn't ask for them.
DROP_COLS = {
    "customerID": "Identifier, unique per row; no predictive value.",
    "TotalCharges": "~ tenure x MonthlyCharges; collinear with them (to confirm with CV in the baseline phase).",
    "gender": "No link to churn: Cramer's V = 0.000, p = 0.47, both genders on the 26.5% baseline.",
    "PhoneService": "Fully contained in MultipleLines ('No phone service' level); U(PhoneService | MultipleLines) = 1.0.",
    "StreamingMovies": "Adds no churn signal beyond StreamingTV (U(Churn) 0.0011 -> 0.0017 among internet customers).",
}

# Add-on service columns whose "No internet service" level is exactly InternetService == "No".
# features.collapse_no_internet maps that level to "No" so the information lives only in InternetService.
NO_INTERNET_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV"]
