# Telco Customer Churn Prediction

Predict which telecom customers are likely to churn, so retention offers can be targeted.
Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, ~26.5% churn).

## Context
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

## Content
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

**The data set includes information about:**

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## Progress
| Phase | Status |
|---|---|
| 1. Scaffolding (CCDS v2 layout, uv + Python 3.12, `churn` package) | ✅ Done |
| 2. Data cleaning (`notebooks/1_data-explore.ipynb`, `churn/dataset.py`) | 🟡 In progress: univariate analysis done, `TotalCharges` fixed |
| 3. EDA · 4. Feature engineering · 5. Modelling · 6. SHAP | ⬜ Not started |
| 7. Streamlit app · 8. Docker · 9. Cloud Run | ⬜ Not started |

## Data findings so far
- **Imbalanced target:** 73% stayed, 27% churned, so accuracy is misleading and the project reports Precision, Recall, F1, ROC-AUC and PR-AUC.
- **Cleaned data format:** saved to `data/interim/telco_customer_churn_interim.parquet`, which keeps column dtypes (unlike CSV/Excel).
- **`TotalCharges` quirk:** stored as text; 11 rows are blank. All 11 have `tenure == 0` (new customers not yet billed, none churned), so the blanks are filled with **0** rather than a mean/median or dropped.
- **`tenure`** is U-shaped: many brand-new customers and a spike at 72 months, which is the dataset's cap, not real behaviour.
- **`TotalCharges`** is right-skewed and roughly equals `tenure × MonthlyCharges` (collinear; to handle in feature engineering).
- **Demographics:** gender is ~50/50, ~16% are senior citizens, ~52% have a partner, ~30% have dependents.

## Results
_To be filled in after model selection._

## Live app
_To be filled in after deploying to Cloud Run._

## Project organization
```
├── data/{raw,interim,processed,external}   <- git-ignored; raw CSV goes in data/raw/
├── models/          <- trained pipeline + metadata
├── notebooks/       <- 1_data-explore.ipynb (cleaning + univariate); later N.0-jg-<topic>.ipynb
├── references/      <- data dictionary and other reference material
├── reports/figures/ <- generated plots
├── churn/           <- source package (config, dataset, features, evaluate, plots, modeling/)
├── app/             <- Streamlit app
├── tests/
└── docs/
```

## How to run
```bash
uv sync                      # Python 3.12 env + editable install of the churn package
uv run jupyter lab           # open notebooks/1_data-explore.ipynb
```
The raw CSV is not committed: download it from Kaggle into `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
_More commands to be added as the pipeline is built._
