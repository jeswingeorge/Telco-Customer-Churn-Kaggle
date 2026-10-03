# Telco Customer Churn Prediction

Predict which telecom customers are likely to churn, so retention offers can be targeted.
Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, ~26.5% churn).

**🚀 Live app:** [https://churn-app-1012735950104.asia-south1.run.app](https://churn-app-1012735950104.asia-south1.run.app) (Streamlit on Google Cloud Run; the first visit may take a few seconds to wake up)

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
| 4. Feature engineering (`notebooks/3_feature_engg.ipynb`, `churn/features.py`) | ✅ Done: 3 engineered features and the preprocessing `ColumnTransformer` |
| 5. Modelling (`notebooks/4_baselines.ipynb`, `notebooks/5_tuning_final.ipynb`, `churn/modeling/`) | ✅ Done: baselines, feature ablation, Optuna tuning, threshold, model choice, saved model |
| 6. SHAP explainability | ⏸ Deferred |
| 7. Streamlit app (`app/streamlit_app.py`) | ✅ Done: single-customer scoring with reasons, batch CSV scoring, model card |
| 8. Docker (`Dockerfile`) | ✅ Done: slim Python 3.12 image, runtime deps only, non-root user |
| 9. Cloud Run | ✅ Done: [live app](https://churn-app-1012735950104.asia-south1.run.app) (Cloud Build + Artifact Registry, scale-to-zero) |
| Tests (pytest) | ⏸ Deferred |

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

## Feature engineering
All preprocessing lives in `churn/features.py` and runs **inside the sklearn Pipeline**, so anything that learns from the data (encoders, scalers) is fitted only on training folds (no leakage), and the Streamlit app applies exactly the same steps.

| Feature | Rule | Why |
|---|---|---|
| `tenure_group` | bands 0–6 / 7–12 / 13–24 / 25–48 / 49–72 months (fixed edges) | captures the steep first-6-months kink (52.9% → 35.9% churn) that raw `tenure` misses in Logistic Regression |
| `num_services` | count of the 5 add-on services | churn falls from 54.9% (internet customers, 0 add-ons) to 5.3% (all 5); **not used** in the final model (see below) |
| `has_family` | has a partner or dependents | both lower churn; **not used** in the final model (see below) |

- **Tenure is limited to 0–72 months**, the dataset's range; the app enforces it rather than extrapolating beyond the training data.
- **A pitfall in the count:** customers without internet also have 0 add-ons but churn at only 7.4%, which hides the 54.9% of internet customers with none. The `InternetService` column separates the two groups in the model.
- **Preprocessing per model family** (`build_preprocessor`): Logistic Regression gets standardised numerics and one-hot categoricals with one column per Yes/No feature (`drop="if_binary"`, avoids the dummy-variable trap); tree models get raw numerics and full one-hot columns. Excluded columns (`config.DROP_COLS`) are dropped by the transformer.
- **Final feature set (decided by a cross-validated ablation):** `tenure`, `MonthlyCharges` and 14 categoricals including `tenure_group` (29 model columns for Logistic Regression). Removing `tenure_group` was the only change that clearly hurt (−0.003 ROC-AUC). `num_services` and `has_family` added nothing over the columns they are built from, so they were removed. Adding `TotalCharges` or the other dropped columns back gave no gain, which confirms the EDA decisions.

## Results
Full details: [reports/model_comparison.md](reports/model_comparison.md) and `notebooks/5_tuning_final.ipynb`.

**Process (no leakage):** stratified 80/20 split; every preprocessing step sits inside an sklearn `Pipeline` fitted on training folds only; models compared with 5-fold stratified cross-validation; the test set was used **once**, at the end.

| Model (tuned with Optuna, CV on train) | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Baseline: always "no churn" | 0.500 | 0.265 | — |
| Decision tree | 0.829 | 0.625 | 0.141 |
| XGBoost | 0.850 | 0.671 | 0.133 |
| **Logistic Regression (elastic-net), chosen** | **0.848** | **0.665** | **0.134** |

**Why Logistic Regression, although XGBoost scored slightly higher?**

The gap is tiny: XGBoost scored 0.850 ROC-AUC and Logistic Regression 0.848. Comparing them on the same 15 cross-validation splits, XGBoost was ahead by only **0.0013** on average. In business terms that means roughly **12 fewer wasted offers per 2,000 customers contacted**, which is almost nothing. My rule was to pick the more complex model only if it won by at least 0.01.

For that small gap, Logistic Regression gives a lot back:
- **You can explain it in one sentence.** Example: *"a customer on a month-to-month contract has about 3.7× the odds of churning compared with one on a two-year contract."* That number comes straight from the model's coefficients. XGBoost is hundreds of decision trees added together; to explain even one prediction you need extra tools such as SHAP.
- **Its probabilities can be trusted.** When it says "30% chance of churn", about 30 in 100 such customers really do churn (it is *calibrated*). That matters because the decision threshold below is chosen on these probabilities.
- **It is simpler to run:** 14× faster to train, smaller, and easier to maintain.

Why it loses almost nothing: the EDA showed that churn risk mostly *adds up* factor by factor (contract + tenure + internet type + payment method...). That is exactly the pattern a linear model captures. XGBoost's strength, finding complex interactions, has little extra to find here.

**Why flag customers at 30.7% churn probability instead of 50%?**

The model gives each customer a probability of churning. The **threshold** is the cut-off above which we call a customer "high risk" and send a retention offer. 50% is only the default; the right cut-off depends on what each kind of mistake costs:
- **Missing a churner** (no offer, they leave): we lose a customer paying about **$74/month**.
- **An unnecessary offer** (they would have stayed anyway): we lose only the cost of the offer, e.g. a small discount.

Missing a churner is much more expensive, so it pays to flag more customers. Here is what the two thresholds do on the 1,409 test customers (374 of whom actually churned):

| Threshold | Customers flagged | Churners caught | Churners missed | Offers to customers who'd have stayed |
|---|---|---|---|---|
| 0.50 (default) | 288 | 195 (52%) | 179 | 93 |
| **0.307 (chosen)** | 523 | **286 (77%)** | **88** | 237 |

Lowering the threshold catches **91 more churners** at the cost of **144 more offers** to customers who would have stayed. As long as an offer is much cheaper than losing a $74/month customer, that is a good trade.

How 0.307 was picked: the goal was to **catch at least 75% of churners** (recall ≥ 75%), and among the thresholds that do, to choose the one with the fewest wasted offers (best precision). It was chosen on cross-validation predictions from the training data, never on the test set, so the test result above is an honest check.

Alternatives I tested and rejected:
- **Class weighting** (making churners count more during training) gave the same trade-off as simply moving the threshold, but distorted the probabilities.
- **Optimising F2** (a score that favours recall) flagged half of all customers, and the extra people it flagged churned no more often than average.

**Test set (1,409 customers, evaluated once):**

| ROC-AUC | PR-AUC | Precision | Recall | Customers flagged |
|---|---|---|---|---|
| 0.846 | 0.651 | 0.547 | 0.765 | 37% |

The test results are in line with cross-validation (ROC-AUC 0.848), so the estimate holds. The model catches 286 of 374 churners; 237 offers go to customers who would have stayed.

**What drives churn (model odds ratios):** month-to-month contract 2.0 vs two-year 0.55 (≈3.7× the odds), first 6 months of tenure 1.7, fibre optic 1.6, electronic check 1.4; online security (0.66) and tech support (0.70) go with staying. These match the EDA findings.

## Streamlit app
`uv run streamlit run app/streamlit_app.py`, then open http://localhost:8501.
- **Single customer:** fill in the sidebar; the churn probability and risk label (threshold 0.307) update live. A "Why this score?" table shows the top features pushing this customer's log-odds up or down (logistic-regression coefficient × scaled/one-hot value; associations, not causes).
- **Batch scoring:** upload a CSV (the raw Kaggle file works) and download it with `churn_probability`, `risk_label` and a `problem` column. Rows the model wasn't trained for (tenure outside 0-72, unknown categories) are flagged "Not scored" instead of getting a misleading score.
- **About the model:** model card and CV vs test metrics, read from `models/metadata.json`.

The app scores only through `churn.modeling.predict` and the saved pipeline, so it can't drift from training.

## Live app
**[https://churn-app-1012735950104.asia-south1.run.app](https://churn-app-1012735950104.asia-south1.run.app)**

Hosted on Google Cloud Run with scale-to-zero (`min-instances 0`): it costs nothing while idle, so the **first visit after a quiet period takes a few seconds** (cold start) while a container starts. After that it responds immediately.

## Project organization
```
├── data/{raw,interim,processed,external}   <- git-ignored; raw CSV goes in data/raw/
├── models/          <- churn_model.joblib (full pipeline) + metadata.json (threshold, metrics, versions)
├── notebooks/       <- 1_data-explore (cleaning + univariate), 2_bi_multivariate_analysis (EDA), 3_feature_engg, 4_baselines (baselines + feature ablation), 5_tuning_final (tuning, threshold, model choice, final fit)
├── references/      <- data dictionary and other reference material
├── reports/         <- model_comparison.md + figures/
├── churn/           <- source package: config, dataset, features, stats, evaluate, modeling/{train,predict}
├── app/             <- Streamlit app
├── tests/
└── docs/
```

## How to run
```bash
uv sync                      # Python 3.12 env + editable install of the churn package
uv run python -m churn.dataset   # raw CSV -> data/processed/telco_churn_clean.parquet
uv run python -m churn.modeling.train     # tune, select, fit, test once; saves models/ + reports/ (~3-4 min)
uv run python -m churn.modeling.predict   # score customers with the saved model (--input/--output CSV)
uv run streamlit run app/streamlit_app.py   # the app at http://localhost:8501
uv run jupyter lab           # open the notebooks
```
The raw CSV is not committed: download it from Kaggle into `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
### With Docker
```bash
docker build -t churn-app .                  # build the image (Python 3.12 + runtime deps + churn/ + app/ + models/)
docker run --rm -p 8080:8080 churn-app       # then open http://localhost:8080
```
The image installs only the runtime dependencies from `uv.lock` (`uv sync --frozen --no-dev`), so it uses the same library versions the model was trained with. Training-only libraries (xgboost, shap, optuna, jupyter) live in the dev group and stay out of the image, which halves it (4.2 GB → 2.0 GB; 456 MB compressed). It runs as a non-root user and listens on `$PORT` (default 8080), as Cloud Run expects.

### Deploying to Google Cloud Run
One-time setup: a GCP project with billing, the Google Cloud SDK, then:
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create churn-repo --repository-format=docker --location=asia-south1
```
Build and deploy:
```bash
gcloud builds submit                            # Cloud Build runs cloudbuild.yaml -> churn-app:v1 in Artifact Registry
gcloud run deploy churn-app \
  --image asia-south1-docker.pkg.dev/<project-id>/churn-repo/churn-app:v1 \
  --region asia-south1 --allow-unauthenticated --memory 1Gi \
  --min-instances 0 --max-instances 2 --session-affinity --timeout 3600
```
- The image is built in the cloud from the same `Dockerfile`; `.gcloudignore` uploads only the files it needs (~20 files).
- `--min-instances 0` scales to zero (no idle cost, short cold start); `--max-instances 2` caps the cost.
- `--session-affinity` keeps each user on one container, because Streamlit keeps the session (and uploaded CSVs) in memory; `--timeout 3600` lets its WebSocket connection stay open.
- New version: `gcloud builds submit --substitutions=_TAG=v2`, then deploy `:v2`. Each deploy is a new revision, so you can roll back.
