# Spec: Telco Customer Churn Prediction (Portfolio / Interview Project)

## Context
You need an end-to-end churn project to show interviewers: clean structure (Cookiecutter Data Science v2), careful data work, a comparison of several models judged on metrics that suit imbalanced data (Precision, Recall, ROC-AUC, not plain accuracy), and a deployed app (Streamlit → Docker → GCP Cloud Run). The finished repo should show *judgement*: no leakage, a reasoned choice of threshold, results you can explain, and a working public URL.

**Status (2026-09-30):** Phase 1 (scaffolding) is done and committed. Phase 2 (data cleaning) is **in progress**: notebook `notebooks/1_data-explore.ipynb` has a univariate pass over all 21 columns, converts `TotalCharges` to numeric, fills its 11 blanks with 0, and writes `data/interim/telco_customer_churn_interim.parquet` (via `pyarrow`). `churn/dataset.py` is still a stub and `references/data_dictionary.md` is not written yet. `docker` and `gcloud` are **not** installed yet.

**Tutor skill:** `.claude/skills/churn-tutor/SKILL.md` (user-invoked only) runs Socratic tutoring + interview quizzes phase by phase.

**Decisions made:** uv + Python 3.12 · notebooks narrate, a reusable `churn` package does the work · extras: hyperparameter tuning (Optuna) + SHAP explainability · cleaned data saved as parquet (`.parquet`, via `pyarrow`; briefly Excel on 2026-09-30, switched back 2026-10-01 because parquet keeps dtypes) · `TotalCharges` is excluded from the model features (decided 2026-10-02, see section 3).

---

## 1. Project structure (CCDS v2)
Created by hand following the CCDS v2 layout. The `ccds` generator couldn't be used: Windows Application Control blocks executables run from uv's cache, so `uvx` fails on this machine. `churn/` is installed in editable mode through the `uv_build` backend (`[tool.uv.build-backend] module-name = "churn"`, `module-root = ""`), so notebooks can `import churn`.

```
Telco-Customer-Churn-Kaggle/
├── README.md               # problem, results table, screenshots, live URL, how to run
├── pyproject.toml / uv.lock  # requires-python = ">=3.12"
├── .python-version         # pins uv to 3.12 (xgboost/shap wheels lag on 3.14)
├── Makefile                # data / train / app / docker targets (thin wrappers on `uv run`)
├── Dockerfile  .dockerignore
├── data/{raw,interim,processed,external}/   # raw CSV moves to data/raw/ (git-ignored)
├── models/                 # final churn_model.joblib + metadata.json (committed; small)
├── notebooks/
│   ├── 1_data-explore.ipynb    # current name for the cleaning/univariate notebook (convention: 1.0-jg-data-cleaning)
│   ├── 2.0-jg-eda.ipynb
│   ├── 3.0-jg-baseline-models.ipynb
│   ├── 4.0-jg-tuning-and-selection.ipynb
│   └── 5.0-jg-explainability-shap.ipynb
├── reports/figures/ + reports/model_comparison.md
├── references/data_dictionary.md
├── .claude/specs/project_spec.md   # this file
├── churn/                  # source package
│   ├── config.py           # paths, RANDOM_STATE=42, TARGET, feature lists
│   ├── dataset.py          # load raw → clean → data/interim/, data/processed/
│   ├── features.py         # engineered features + ColumnTransformer builder
│   ├── modeling/train.py   # CV, Optuna tuning, final fit, save artifact
│   ├── modeling/predict.py # load artifact, predict_proba, apply threshold
│   ├── evaluate.py         # metric functions, threshold selection, curves
│   └── plots.py            # shared EDA/eval plotting helpers
├── app/streamlit_app.py    # UI (imports churn.modeling.predict)
└── tests/                  # small pytest suite (cleaning, pipeline shape, predict)
```
`.gitignore` ignores `/data/`, `archive.zip`, `.venv/`, caches, notebook checkpoints and `.env`. `models/churn_model.joblib` stays tracked so the Docker build is self-contained. Empty folders hold a `.gitkeep`.

**Dependencies (in `pyproject.toml`):** runtime: pandas, numpy, scikit-learn, xgboost, joblib, shap, streamlit, matplotlib, seaborn. Dev group: jupyterlab, ipykernel, optuna, pytest, ruff.

## 2. Data cleaning (`churn/dataset.py`, notebook 1.0, `references/data_dictionary.md`)
Known quirks to handle and *document in the notebook*:
- `TotalCharges` is text; 11 rows are `" "`. All have `tenure == 0` (new customers) → set to 0, not drop, with a reason written down.
- `SeniorCitizen` is 0/1 while other binary columns are Yes/No → make them consistent.
- `"No internet service"` / `"No phone service"` values in 7 service columns → keep as their own category (they carry information) and note the redundancy with `InternetService`/`PhoneService`.
- Drop `customerID` (an identifier, no predictive value); target `Churn` Yes/No → 1/0.
- Checks: duplicates, dtypes, value ranges, and class balance (~26.5% churn → imbalanced, which is why accuracy is not used).
- Output: `data/interim/telco_customer_churn_interim.parquet` (decided 2026-10-01). Parquet stores column dtypes (`TotalCharges` stays float, categoricals stay strings), is smaller and faster than Excel, and `pd.read_parquet` needs no re-casting. `dataset.py` writes the same file with `df.to_parquet(..., engine="pyarrow", index=False)`. Trade-off: it can't be opened by hand in Excel.

**Progress so far (notebook `1_data-explore.ipynb`):**
- ✅ `TotalCharges` → numeric (`pd.to_numeric(errors="coerce")`) exposes 11 NaNs. Inspected them: all `tenure == 0`, none churned, none senior, all have dependents, 10/11 on two-year contracts → new customers not yet billed → filled with 0.
- ✅ Univariate look at every column. Notes recorded: `customerID` is unique per row (7,043); gender ≈ 50/50; ~16% senior; ~52% have a partner; ~30% have dependents; `tenure` is U-shaped with a pile-up at the 72-month cap; `TotalCharges` is right-skewed (≈ tenure × MonthlyCharges); churn is 73/27.
- ⬜ Still to do: harmonise `SeniorCitizen` to Yes/No, drop `customerID`, map `Churn` → 1/0, check duplicates and value ranges, move the logic into `churn/dataset.py` (reading paths from `churn.config`, not `../data/...`), write `references/data_dictionary.md`.

## 3. EDA (notebook 2.0, figures saved to `reports/figures/`)
- Target balance; churn rate by each categorical column (contract, payment method, internet type, tech support…).
- Numeric distributions split by churn (tenure, MonthlyCharges, TotalCharges); tenure cohorts.
- Correlation / Cramér's V; note that TotalCharges ≈ tenure × MonthlyCharges (collinear).
- **Decision (2026-10-02): drop `TotalCharges` from the model features; keep `tenure` and `MonthlyCharges`.** Evidence from notebook `2_bi_multivariate_analysis.ipynb`: the MonthlyCharges vs TotalCharges scatter is a wedge bounded by about 72 × MonthlyCharges, and their correlation is 0.65 (spread comes from tenure). `TotalCharges / tenure` (for tenure > 0) tracks `MonthlyCharges` almost linearly, so the two kept columns carry nearly all of its information. This removes collinearity (unstable Logistic Regression coefficients, split importances in tree models). Caveat: the small gap between average and current monthly charge (price changes over a customer's life) is lost. The planned `avg_monthly_charge` feature (`TotalCharges / tenure`) is removed from the spec: the EDA showed it is almost identical to `MonthlyCharges`. To validate in notebook 3.0: compare CV PR-AUC / ROC-AUC with and without `TotalCharges`. Open question: whether the column is dropped in `dataset.py` or left out of the feature lists in `churn/config.py`; either way the Streamlit app must not ask for it.
- End with 5–7 written **business insights** (e.g. month-to-month + fiber + electronic check = high risk). These are what interviewers remember.

## 4. Feature engineering (`churn/features.py`)
- Engineered: `tenure_group` bins, `num_services` (count of add-on services), `has_family` (Partner or Dependents).
- `build_preprocessor(model_type)` returns a `ColumnTransformer`: OneHotEncoder(handle_unknown="ignore") for categoricals; StandardScaler on numerics for LR only (trees don't need scaling).
- **Every step is inside an sklearn `Pipeline`**, fitted only on training folds, so nothing leaks from the test data.

## 5. Modeling & evaluation (`churn/modeling/train.py`, `churn/evaluate.py`, notebooks 3.0–4.0)
- **Split:** stratified 80/20 train/test (`random_state=42`); the test set is used exactly once, at the end.
- **Baselines (notebook 3.0):** DummyClassifier (the floor), Logistic Regression, Decision Tree, XGBoost; stratified 5-fold CV on train.
- **Imbalance:** `class_weight="balanced"` (LR, DT) and `scale_pos_weight = neg/pos` (XGB); compare with and without. No SMOTE (keeps it simple and easy to defend).
- **Metrics reported for every model:** Precision, Recall, F1, ROC-AUC, **PR-AUC** (more informative when positives are rare), plus a confusion matrix. Accuracy shown only for reference.
- **Tuning (notebook 4.0):** Optuna, 50–100 trials per model, objective = mean CV ROC-AUC; search spaces: LR (C, penalty), DT (max_depth, min_samples_leaf, ccp_alpha), XGB (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda).
- **Threshold selection:** use out-of-fold predictions on train to pick the threshold that maximises F2, or meets a set minimum recall (e.g. ≥0.75) with the best precision. Explain the business reasoning: missing a churner costs more than an unnecessary retention offer.
- **Selection rule:** highest CV ROC-AUC, with PR-AUC / recall at the chosen threshold as the tie-breaker. If LR is within ~0.01 AUC of XGB, say so openly and discuss interpretability vs performance.
- **Final:** refit the winning pipeline on the full train set → evaluate once on test → save `models/churn_model.joblib` (pipeline) + `models/metadata.json` (model name, threshold, test metrics, feature list, training date, library versions). Write `reports/model_comparison.md` with the CV + test table and the ROC/PR curves.

## 6. Explainability (notebook 5.0)
- SHAP (`TreeExplainer` for XGB/DT, `LinearExplainer` for LR) on the transformed test set: summary/beeswarm plot, top-feature bar chart, and 2–3 waterfall plots for individual customers.
- Compare the SHAP results with the EDA insights (they should agree).

## 7. Streamlit app (`app/streamlit_app.py`)
- Loads the artifact + metadata once (`@st.cache_resource`).
- **Single customer:** sidebar form with all raw input fields → churn probability, a risk label based on the saved threshold, and a SHAP waterfall showing the top reasons.
- **Batch:** upload a CSV → table of scores, downloadable CSV.
- **About tab:** metrics table and model card taken from `metadata.json`.
- The feature engineering lives in the saved pipeline (or in `churn.features`, imported), so the app and training transform data the same way.

## 8. Docker (`Dockerfile`)
- `python:3.12-slim`; install only the runtime deps with `uv sync --frozen --no-dev`; copy `churn/`, `app/`, `models/`.
- Run as a non-root user; `CMD streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true`.
- `.dockerignore`: data/, notebooks/, .venv, reports/, tests/.
- Local check: `docker build -t churn-app . && docker run -p 8080:8080 churn-app`.

## 9. GCP Cloud Run deployment
Prerequisites: install Docker Desktop + Google Cloud SDK; have a GCP project with billing enabled.
1. `gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com`
2. `gcloud artifacts repositories create churn-repo --repository-format=docker --location=<region>`
3. `gcloud builds submit --tag <region>-docker.pkg.dev/<project>/churn-repo/churn-app:v1`
4. `gcloud run deploy churn-app --image <...>:v1 --region <region> --allow-unauthenticated --memory 1Gi --min-instances 0 --max-instances 2`
- Put the live URL and a screenshot in the README. min-instances 0 keeps it within the free tier; mention the cold start in the README.

## 10. Implementation order
1. ✅ Scaffold CCDS, `git init`, uv env on 3.12, move raw CSV. (Data dictionary moves to Phase 2.)
2. 🟡 `dataset.py` + notebook 1.0 (univariate pass + TotalCharges fix done in the notebook; `dataset.py` + data dictionary pending) → 3. EDA notebook 2.0 → 4. `features.py`
5. Baselines (3.0) → 6. Optuna + threshold + final model (4.0, `train.py`) → 7. SHAP (5.0)
8. Streamlit app → 9. Docker → 10. Cloud Run → 11. README + tests polish.

## 11. Verification
- `uv run pytest`: the cleaning output has no nulls and a numeric TotalCharges (in the interim data; it is not a model feature); the pipeline fits/predicts on a sample; `predict.py` returns probabilities in [0,1] and respects the threshold.
- `uv run python -m churn.dataset && uv run python -m churn.modeling.train` rebuilds the artifact from the raw CSV and gives the same metrics (fixed seed).
- Notebooks run top to bottom (`jupyter nbconvert --execute`).
- Sanity target: tuned models should reach test ROC-AUC ≈ 0.83–0.85 on this dataset; much higher suggests leakage.
- `uv run streamlit run app/streamlit_app.py` works locally → the same in `docker run` → the Cloud Run URL gives the same prediction for the same test customer.

## Interview talking points to capture in the README
Why not accuracy · how leakage was prevented · how the threshold was chosen · LR vs XGB trade-off · top churn drivers (from SHAP) · what I would do next (cost-sensitive threshold using customer value, monitoring for drift, retraining).
