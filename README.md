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
| 2. Data cleaning (`notebooks/1_data-explore.ipynb`, `churn/dataset.py`) | ✅ Done: univariate analysis, cleaning script, [data dictionary](references/data_dictionary.md) |
| 3. EDA (`notebooks/2_bi_multivariate_analysis.ipynb`) | ✅ Done: feature decisions and 7 business insights (figures saved after modelling) |
| 4. Feature engineering (`churn/features.py`, `churn/config.py`) | 🟡 Started: drop list in `config.DROP_COLS`, `collapse_no_internet` transform |
| 5. Modelling · 6. SHAP | ⬜ Not started |
| 7. Streamlit app · 8. Docker · 9. Cloud Run | ⬜ Not started |

## Data findings so far
- **Imbalanced target:** 73% stayed, 27% churned, so accuracy is misleading and the project reports Precision, Recall, F1, ROC-AUC and PR-AUC.
- **Cleaned data:** `uv run python -m churn.dataset` fixes `TotalCharges`, maps `Churn` to 1/0 and `SeniorCitizen` to Yes/No, and saves `data/processed/telco_churn_clean.parquet`. Parquet keeps column dtypes (unlike CSV/Excel). All columns are kept; the model pipeline chooses the features.
- **`TotalCharges` quirk:** stored as text; 11 rows are blank. All 11 have `tenure == 0` (new customers not yet billed, none churned), so the blanks are filled with **0** rather than a mean/median or dropped.
- **`tenure`** is U-shaped: many brand-new customers and a spike at 72 months, which is the dataset's cap, not real behaviour.
- **`TotalCharges`** is right-skewed and roughly equals `tenure × MonthlyCharges` (collinear). Decision: `TotalCharges` is dropped from the model features; `tenure` and `MonthlyCharges` carry nearly all of its information (to be confirmed by CV with and without it).
- **Demographics:** gender is ~50/50, ~16% are senior citizens, ~52% have a partner, ~30% have dependents.

## EDA findings
Churn rate by group, against the 26.5% baseline. Associations, not proof of cause. Feature strength measured with Cramér's V (symmetric effect size) and Theil's U (how much knowing the feature reduces uncertainty about churn); p-values only used as a gate, since with 7,043 rows almost everything is "significant".

| Driver | Highest-risk group | Lowest-risk group |
|---|---|---|
| Contract (strongest, U = 0.17) | Month-to-month 42.7% | Two-year 2.8% |
| Tenure | First 6 months 52.9% | 4+ years 9.5% |
| Payment method | Electronic check 45.3% | Automatic methods 15–17% |
| Internet service | Fibre optic 41.9% | No internet 7.4% |
| Add-on services (internet customers) | 0 add-ons 54.9% | All 5 add-ons 5.3% |
| Online security / tech support | Without 42% | With 15% |
| Senior citizen | Senior 41.7% | Non-senior 23.6% |
| Paperless billing | Yes 33.6% | No 16.3% (holds within every payment method) |
| Partner / Dependents | No partner 33.0%, no dependents 31.3% | With partner 19.7%, with dependents 15.5% |

**Business insights** (full write-up at the end of the EDA notebook):
1. **Contract** is the biggest driver: month-to-month customers churn 15× more than two-year customers. Moving them to a one-year contract is the main lever.
2. **The first six months** are the danger zone (52.9% churn); 55% of churners leave in their first year, so onboarding matters most.
3. **Fibre** customers churn at 41.9% vs 19.0% for DSL, at every tenure: a price-for-value or service-quality question.
4. **Electronic-check** payers churn at 45.3%, even within month-to-month contracts; nudging them to autopay is cheap.
5. **Each add-on service** lowers churn: 55% with none, 5% with all five; security and tech support matter most.
6. **One segment** (month-to-month + fibre + electronic check) is 18.6% of customers but **42% of all churners** (60.4% churn rate): the first target list.
7. **Seniors and single customers** churn more, largely because of their contract and payment choices.

**Feature decisions** (reasons kept in `churn/config.py → DROP_COLS`; each to be confirmed with cross-validation in the baseline phase):
- Dropped `TotalCharges`: ≈ `tenure × MonthlyCharges`, so it is collinear with them.
- Dropped `gender`: no relationship with churn (Cramér's V = 0.000, p = 0.47).
- Dropped `PhoneService`: fully contained in `MultipleLines` (its "No phone service" level).
- Dropped `StreamingMovies`: adds no churn signal beyond `StreamingTV`. Their apparent overlap (V = 0.77) was inflated by the shared "No internet service" level; among internet customers it is 0.43.
- **"No internet service"** in five add-on columns is identical to `InternetService == "No"`, which would create six identical one-hot columns. It is collapsed to "No" inside the model pipeline (`churn.features.collapse_no_internet`), so the app applies the same step.

## Results
_To be filled in after model selection._

## Live app
_To be filled in after deploying to Cloud Run._

## Project organization
```
├── data/{raw,interim,processed,external}   <- git-ignored; raw CSV goes in data/raw/
├── models/          <- trained pipeline + metadata
├── notebooks/       <- 1_data-explore.ipynb (cleaning + univariate), 2_bi_multivariate_analysis.ipynb (EDA); later N.0-jg-<topic>.ipynb
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
uv run python -m churn.dataset   # raw CSV -> data/processed/telco_churn_clean.parquet
uv run jupyter lab           # open the notebooks
```
The raw CSV is not committed: download it from Kaggle into `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
_More commands to be added as the pipeline is built._
