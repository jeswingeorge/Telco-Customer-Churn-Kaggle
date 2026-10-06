# Telco Customer Churn Prediction

Predict which telecom customers are likely to leave, so a retention team can target offers at the right people, and explain *why* they leave.

Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data): 7,043 customers, 21 columns, **26.5% churn**. Column details are in the [data dictionary](references/data_dictionary.md).

## Live app
_To be added after deploying to Cloud Run._

## Results
_To be added after model selection: CV and test metrics, the chosen threshold and why._

## Key findings
Churn rate by group, against the 26.5% baseline (associations, not proof of cause). Feature strength is measured with Cramér's V and Theil's U rather than p-values, since with 7,043 rows almost everything is "significant".

| Driver | Highest-risk group | Lowest-risk group |
|---|---|---|
| Contract (strongest, U = 0.17) | Month-to-month 42.7% | Two-year 2.8% |
| Tenure | First 6 months 52.9% | 4+ years 9.5% |
| Payment method | Electronic check 45.3% | Automatic methods 15–17% |
| Internet service | Fibre optic 41.9% | No internet 7.4% |
| Add-on services (internet customers) | 0 add-ons 54.9% | All 5 add-ons 5.3% |
| Online security / tech support | Without 42% | With 15% |
| Senior citizen | Senior 41.7% | Non-senior 23.6% |
| Paperless billing | Yes 33.6% | No 16.3% |
| Partner / Dependents | No partner 33.0%, no dependents 31.3% | With partner 19.7%, with dependents 15.5% |

**Business insights:**
1. **Contract** is the biggest driver: month-to-month customers churn 15× more than two-year customers. Moving them to a one-year contract is the main lever.
2. **The first six months** are the danger zone (52.9% churn); 55% of churners leave in their first year, so onboarding matters most.
3. **Fibre** customers churn at 41.9% vs 19.0% for DSL, at every tenure: a price-for-value or service-quality question.
4. **Electronic-check** payers churn at 45.3%, even within month-to-month contracts; nudging them to autopay is cheap.
5. **Each add-on service** lowers churn: 55% with none, 5% with all five; security and tech support matter most.
6. **One segment** (month-to-month + fibre + electronic check) is 18.6% of customers but **42% of all churners**: the first target list.
7. **Seniors and single customers** churn more, largely because of their contract and payment choices.

## Approach (CRISP-DM)
| Stage | What was done | Where | Status |
|---|---|---|---|
| Business understanding | Framed as binary classification; a missed churner costs more than an unneeded offer | this README | ✅ |
| Data cleaning | Fixed `TotalCharges` (11 blanks = new customers → 0), encoded the target, saved typed parquet | [1_data-explore](notebooks/1_data-explore.ipynb), `churn/dataset.py` | ✅ |
| EDA | Churn drivers, column drop decisions, 7 business insights | [2_bi_multivariate_analysis](notebooks/2_bi_multivariate_analysis.ipynb) | ✅ |
| Feature engineering | 3 engineered features + per-model preprocessing inside an sklearn Pipeline | [3_feature_engg](notebooks/3_feature_engg.ipynb), `churn/features.py` | ✅ |
| Modelling | Logistic Regression → KNN/SVM → Decision Tree → Random Forest → XGBoost/LightGBM/CatBoost, 5-fold CV | | ⬜ |
| Evaluation | Business-driven threshold, one-time test evaluation, SHAP explanations | | ⬜ |
| Deployment | Drift checks, Streamlit app, Docker, GCP Cloud Run | | ⬜ |

## Key modelling decisions
- **Columns left out** (reasons in `churn/config.py`):
  - `TotalCharges`: ≈ tenure × MonthlyCharges, so collinear.
  - `gender`: no link to churn (Cramér's V = 0.000).
  - `PhoneService`: contained in `MultipleLines`.
  - `StreamingMovies`: adds nothing beyond `StreamingTV`.
  - `customerID`: an identifier.
- **Engineered features:**
  - `tenure_group` (0–6 / 7–12 / 13–24 / 25–48 / 49–72 months) captures the steep first-6-months drop.
  - `num_services` counts the 5 add-on services.
  - `has_family` is a candidate, kept only if cross-validation supports it.
- **"No internet service"** in the add-on columns duplicates `InternetService == "No"`, so it is collapsed to "No" inside the Pipeline.
- **Tenure is limited to 0–72 months** (the training range); the app enforces it rather than extrapolating.

## How class imbalance is handled
Only 26.5% of customers churn. A model that always predicts "stays" is 73.5% accurate and catches no churners, so accuracy is not used to judge models.
- **Metrics:** PR-AUC, ROC-AUC, recall, precision and F1.
- **Class weights** (`class_weight="balanced"`, `scale_pos_weight` for XGBoost) make each churner count more during training.
- **A tuned decision threshold** instead of the default 0.5, chosen from the business cost of a missed churner vs an unneeded offer.

**Why no SMOTE or undersampling:**
- The imbalance is moderate, not extreme.
- SMOTE creates synthetic customers by blending one-hot encoded rows, which produces unrealistic profiles.
- Resampling done outside the cross-validation folds leaks information.
- Resampling distorts the predicted probabilities that the cost-based threshold and the app's "% risk" rely on.
- Class weights plus a threshold usually reach the same recall more simply.

A SMOTE comparison is planned for the second iteration to back this up with evidence.

## How data leakage is prevented
- **Split first:** stratified 80/20 train/test before anything learns from the data.
- **Everything that learns sits inside the sklearn Pipeline** (scaling, encoding, engineered features), so it is fitted on training folds only, and the app applies exactly the same steps.
- **The threshold is chosen on out-of-fold training predictions**, never on the test set.
- **The test set is used once**, at the end.
- **Checks:** a review of which features are known before churn, a scan for any single feature that predicts too well, train-vs-CV and CV-vs-test gaps.

## Monitoring
_To be added: input checks (ranges, unseen categories), PSI-based data and prediction drift against a training reference profile, and retrain triggers._

## How to run
```bash
uv sync                          # Python 3.12 env + editable install of the churn package
uv run python -m churn.dataset   # raw CSV -> data/processed/telco_churn_clean.parquet
uv run jupyter lab               # open the notebooks
```
The raw CSV is not committed: download it from Kaggle into `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Project structure
```
├── data/{raw,interim,processed,external}   <- git-ignored; raw CSV goes in data/raw/
├── models/          <- trained pipeline + metadata
├── notebooks/       <- 1_data-explore, 2_bi_multivariate_analysis, 3_feature_engg, ...
├── references/      <- data dictionary
├── reports/figures/ <- generated plots
├── churn/           <- source package (config, dataset, features, stats, evaluate, plots, modeling/)
├── app/             <- Streamlit app
└── tests/
```
