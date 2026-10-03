# Spec: Telco Customer Churn Prediction (Portfolio / Interview Project)

## Context
You need an end-to-end churn project to show interviewers: clean structure (Cookiecutter Data Science v2), careful data work, a comparison of several models judged on metrics that suit imbalanced data (Precision, Recall, ROC-AUC, not plain accuracy), and a deployed app (Streamlit → Docker → GCP Cloud Run). The finished repo should show *judgement*: no leakage, a reasoned choice of threshold, results you can explain, and a working public URL.

**Status (2026-10-03):** Phases 1–5 are **done** (scaffolding, cleaning, EDA, feature engineering, modelling). Notebooks: `1_data-explore.ipynb` (univariate + `TotalCharges` fix), `2_bi_multivariate_analysis.ipynb` (EDA, 7 business insights), `3_feature_engg.ipynb`, `4_baselines.ipynb` (the spec's 3.0: baselines + feature ablation), `5_tuning_final.ipynb` (the spec's 4.0: Optuna, threshold, model choice, final fit). Package: `churn.dataset` (raw CSV → `data/processed/telco_churn_clean.parquet`), `churn.features`, `churn.evaluate`, `churn.modeling.train` (tunes, selects, fits, saves) and `churn.modeling.predict` (loads the artifact, validates, scores). **Final model:** unweighted elastic-net Logistic Regression, threshold 0.307; test ROC-AUC 0.846, PR-AUC 0.651, recall 0.765, precision 0.547. `models/churn_model.joblib` + `models/metadata.json` and `reports/model_comparison.md` are generated. **Deferred by the user (2026-10-03):** the pytest suite (section 11) and Phase 7 SHAP (section 6). Streamlit app (section 7), Docker (section 8) and Cloud Run (section 9) are **done** (2026-10-03); live at https://churn-app-1012735950104.asia-south1.run.app; images are in Artifact Registry (`churn-repo/churn-app`, console: Artifact Registry → churn-repo). README polish started (2026-10-03): live link at the top; the model-choice and threshold explanations were rewritten in plain language with a worked example (test set at 0.5 vs 0.307: 195 vs 286 churners caught, 93 vs 237 unnecessary offers). **Next:** README screenshot of the live app, the user's same-customer check on the live URL, `git push`; then the deferred tests/SHAP if the user wants them.

**Tutor skill:** `.claude/skills/churn-tutor/SKILL.md` (user-invoked only) runs Socratic tutoring phase by phase (no interview quizzes unless asked).

**Decisions made:** uv + Python 3.12 · notebooks narrate, a reusable `churn` package does the work · extras: hyperparameter tuning (Optuna) + SHAP explainability (SHAP deferred on 2026-10-03; the app goes first) · cleaned data saved as parquet (`.parquet`, via `pyarrow`; briefly Excel on 2026-09-30, switched back 2026-10-01 because parquet keeps dtypes) · `TotalCharges` is excluded from the model features (decided 2026-10-02, see section 3).

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
│   ├── dataset.py          # load raw → clean → data/processed/telco_churn_clean.parquet
│   ├── features.py         # engineered features + ColumnTransformer builder
│   ├── modeling/train.py   # CV, Optuna tuning, final fit, save artifact
│   ├── modeling/predict.py # load artifact, predict_proba, apply threshold
│   ├── evaluate.py         # metric functions, threshold selection, curves
│   └── plots.py            # shared EDA/eval plotting helpers
├── app/streamlit_app.py    # UI (imports churn.modeling.predict)
└── tests/                  # small pytest suite (cleaning, pipeline shape, predict)
```
`.gitignore` ignores `/data/`, `archive.zip`, `.venv/`, caches, notebook checkpoints and `.env`. `models/churn_model.joblib` stays tracked so the Docker build is self-contained. Empty folders hold a `.gitkeep`.

**Dependencies (in `pyproject.toml`):** runtime (what the app/Docker image needs): pandas, numpy, scikit-learn, scipy, joblib, pyarrow, streamlit, matplotlib. Dev group (training/analysis, local only): jupyterlab, ipykernel, optuna, pytest, ruff, xgboost, shap, seaborn. **Decision (2026-10-03):** xgboost, shap and seaborn moved to dev so the Docker image skips them (xgboost pulls ~345 MB of NVIDIA GPU libs on Linux; image 4.17 GB → 1.98 GB, compressed 1.32 GB → 456 MB). `uv sync` still installs them locally; `train.py` and the notebooks are unaffected.

## 2. Data cleaning (`churn/dataset.py`, notebook 1.0, `references/data_dictionary.md`)
Known quirks to handle and *document in the notebook*:
- `TotalCharges` is text; 11 rows are `" "`. All have `tenure == 0` (new customers) → set to 0, not drop, with a reason written down.
- `SeniorCitizen` is 0/1 while other binary columns are Yes/No → **Decision (2026-10-02):** converted to Yes/No in `dataset.clean()` so it is categorical like the others (never scaled as a number) and the app shows a Yes/No choice.
- `"No phone service"` in `MultipleLines` → kept as its own level; it makes `PhoneService` redundant, so `PhoneService` is dropped.
- **Decision (2026-10-02):** `"No internet service"` in the add-on columns (`config.NO_INTERNET_COLS`) is identical to `InternetService == "No"` → collapsed to `"No"` by `features.collapse_no_internet`, a stateless `FunctionTransformer` step inside the Pipeline (so the app applies it too). Avoids six identical one-hot dummies.
- Drop `customerID` (an identifier, no predictive value); target `Churn` Yes/No → 1/0.
- Checks: duplicates, dtypes, value ranges, and class balance (~26.5% churn → imbalanced, which is why accuracy is not used).
- Output: notebook 1 writes `data/interim/telco_customer_churn_interim.parquet` (decided 2026-10-01; used by the EDA notebook); `dataset.py` writes the modelling input `data/processed/telco_churn_clean.parquet`. Parquet stores column dtypes (`TotalCharges` stays float, categoricals stay strings), is smaller and faster than Excel, and `pd.read_parquet` needs no re-casting. Both use `df.to_parquet(..., engine="pyarrow", index=False)`. Trade-off: it can't be opened by hand in Excel.

**Progress so far (notebook `1_data-explore.ipynb`):**
- ✅ `TotalCharges` → numeric (`pd.to_numeric(errors="coerce")`) exposes 11 NaNs. Inspected them: all `tenure == 0`, none churned, none senior, all have dependents, 10/11 on two-year contracts → new customers not yet billed → filled with 0.
- ✅ Univariate look at every column. Notes recorded: `customerID` is unique per row (7,043); gender ≈ 50/50; ~16% senior; ~52% have a partner; ~30% have dependents; `tenure` is U-shaped with a pile-up at the 72-month cap; `TotalCharges` is right-skewed (≈ tenure × MonthlyCharges); churn is 73/27.
- ✅ Duplicates checked: none; 22 rows are identical once `customerID` is removed (different customers, same profile) → kept.
- ✅ `references/data_dictionary.md` written (all 21 columns: type, values, meaning, quirks, model use).
- ✅ `churn/dataset.py` (2026-10-02; written by the user from the skeleton): `load_raw()` → `clean()` → `save()`, paths from `churn.config`. `clean()`: TotalCharges → numeric, 11 blanks → 0; `Churn` → 1/0 (sklearn metrics default to `pos_label=1`, XGBoost rejects string labels); `SeniorCitizen` → Yes/No. Output: 7,043 rows × 21 columns, no nulls, at `config.CLEAN_DATA_FILE` (*processed* = final modelling input; notebook 1's interim file stays as is). Lesson recorded: with pandas Copy-on-Write, `df[col].fillna(..., inplace=True)` silently does nothing; assign back instead. **Columns are not dropped in the parquet**: `customerID` labels batch predictions and the drops must stay testable with CV; the pipeline selects features from `config`.

## 3. EDA (notebook 2.0, figures saved to `reports/figures/`)
- Target balance; churn rate by each categorical column (contract, payment method, internet type, tech support…).
- Numeric distributions split by churn (tenure, MonthlyCharges, TotalCharges); tenure cohorts.
- Correlation / Cramér's V; note that TotalCharges ≈ tenure × MonthlyCharges (collinear).
- **Decision (2026-10-02): drop `TotalCharges` from the model features; keep `tenure` and `MonthlyCharges`.** Evidence from notebook `2_bi_multivariate_analysis.ipynb`: the MonthlyCharges vs TotalCharges scatter is a wedge bounded by about 72 × MonthlyCharges, and their correlation is 0.65 (spread comes from tenure). `TotalCharges / tenure` (for tenure > 0) tracks `MonthlyCharges` almost linearly, so the two kept columns carry nearly all of its information. This removes collinearity (unstable Logistic Regression coefficients, split importances in tree models). Caveat: the small gap between average and current monthly charge (price changes over a customer's life) is lost. The planned `avg_monthly_charge` feature (`TotalCharges / tenure`) is removed from the spec: the EDA showed it is almost identical to `MonthlyCharges`. To validate in notebook 3.0: compare CV PR-AUC / ROC-AUC with and without `TotalCharges`. Resolved: it stays in the cleaned data and is listed in `config.DROP_COLS`, so the pipeline never selects it and the app does not ask for it.
- **Decisions (2026-10-02): also drop `gender`** (V = 0.000, p = 0.47), **`PhoneService`** (fully contained in `MultipleLines`), **`StreamingMovies`** (no churn signal beyond `StreamingTV` among internet customers; V 0.77 overall was inflated by the shared "No internet service" level, 0.43 among internet customers). All drops with reasons live in `config.DROP_COLS`; confirm with CV in notebook 3.0.
- ✅ 7 written **business insights** at the end of the notebook (contract, first 6 months, fibre, electronic check, add-on count, the month-to-month + fibre + e-check segment = 18.6% of customers but 42% of churners, seniors/family). These are what interviewers remember.
- ✅ Tenure log-odds check: close to linear (R² ≈ 0.92 over 12 bins) except a steep first-6-months kink; `log1p` fits worse (R² ≈ 0.86); the tenure effect lives in month-to-month customers. → LR: raw `tenure` + test a `tenure_group` / new-customer (≤ 6 months) flag with CV; trees: raw `tenure`.
- ⏸ Figures saved to `reports/figures/` after modelling (user's decision).

## 4. Feature engineering (`churn/features.py`)
- Engineered: `tenure_group` bins, `num_services` (count of add-on services), `has_family` (Partner or Dependents).
- EDA evidence (2026-10-02): `num_services` is strong (internet customers: 55% churn with 0 add-ons → 5% with all 5), but it is an exact sum of the add-on binaries, so LR gets either the count or the binaries, not both; trees can take both. `has_family` looks unhelpful (Partner and Dependents each lower churn within the other's groups), so validate with CV before adding. `tenure` churn falls steeply in the first year and then flattens, so check the log-odds shape to choose raw / bins / `log1p` for LR.
- **Decision (2026-10-02): `tenure` is limited to 0–72 months**, the range of the dataset (72 is its maximum). `tenure_group` uses fixed edges `[-1, 6, 12, 24, 48, 72]` → `0-6 / 7-12 / 13-24 / 25-48 / 49-72` (0–6 → 7–12 is the biggest drop, 52.9% → 35.9%). A value above 72 would get NaN from `pd.cut`, so the app enforces the limit (section 7) instead of the bins being open-ended. Trade-off: the model is only valid for customers inside the training range.
- ✅ **Built (2026-10-03)** in `notebooks/3_feature_engg.ipynb`, then moved to `churn/features.py`:
  - `add_features(X)` chains `add_tenure_group`, `add_num_services` (counts `"Yes"` over `config.ADDON_COLS`, which is `NO_INTERNET_COLS`) and `add_has_family`. Stateless (fixed constants from `config`), so it is a `FunctionTransformer` step after `collapse_no_internet` and can't leak. It **always adds all three**; the column lists decide which ones a model uses. Output checked against the notebook versions (identical, no nulls, input not modified).
  - Finding: `num_services == 0` churns at only 24.8% overall because 1,526 of its 2,404 customers have no internet (7.4% churn); internet customers with 0 add-ons churn at 54.9%. The `InternetService` dummies separate the two groups in LR. Counting `"Yes"` gives the same result before or after the "No internet service" collapse.
- ✅ `build_preprocessor(model_type, num_cols=None, cat_cols=None)` returns an unfitted `ColumnTransformer` with `remainder="drop"` (so `DROP_COLS` and the target never reach the model) and `set_output(transform="pandas")` (keeps column names for coefficients and SHAP). Defaults come from `config.NUM_COLS` and `config.CAT_COLS`; the baseline notebook passes other lists for the CV comparisons. Phase 4 started with `num_services` and `has_family` included (31 LR / 41 tree columns); the Phase 5 ablation removed both (section 5), so the defaults are now `NUM_COLS = [tenure, MonthlyCharges]` and 14 categoricals.
  - `"lr"`: `StandardScaler` on numerics; `OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")` → **29 columns** (31 before the Phase 5 ablation).
  - `"tree"` (DT, XGBoost): numerics `"passthrough"`; `OneHotEncoder` without `drop` → **38 columns** (41 before).
  - **Decision (2026-10-03): `drop="if_binary"` for LR.** It avoids the dummy-variable trap for the 10 Yes/No columns (one `_Yes` column each, clean coefficients and SHAP) and keeps every level of the multi-level columns (`MultipleLines`, which has a "No phone service" level, plus `InternetService`, `Contract`, `PaymentMethod` and `tenure_group`); L2 regularisation handles the remaining redundancy. `drop="first"` was considered: it makes the alphabetically first level the reference, and with `handle_unknown="ignore"` an unseen value is encoded exactly like that reference level (only a warning), so it was not chosen.
  - Categories are learned in `fit` (`categories="auto"`), not listed by hand: no upkeep, and inside the Pipeline they are learned from training folds only.
- **Every step is inside an sklearn `Pipeline`**, fitted only on training folds, so nothing leaks from the test data.

## 5. Modeling & evaluation (`churn/modeling/train.py`, `churn/evaluate.py`, notebooks 3.0–4.0)
- **Split:** stratified 80/20 train/test (`random_state=42`); the test set is used exactly once, at the end.
- **Baselines (notebook 3.0):** DummyClassifier (the floor), Logistic Regression, Decision Tree, XGBoost; stratified 5-fold CV on train.
- **Imbalance:** `class_weight="balanced"` (LR, DT) and `scale_pos_weight = neg/pos` (XGB); compare with and without. No SMOTE (keeps it simple and easy to defend).
- **Metrics reported for every model:** Precision, Recall, F1, ROC-AUC, **PR-AUC** (more informative when positives are rare), plus a confusion matrix. Accuracy shown only for reference.
- **Tuning (notebook 4.0):** Optuna, 50–100 trials per model, objective = mean CV ROC-AUC; search spaces: LR (C, l1_ratio with `solver="saga"`; sklearn 1.9 deprecates `penalty`, and `l1_ratio` 0→1 covers L2 → elastic-net → L1), DT (max_depth, min_samples_leaf, ccp_alpha), XGB (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda).
- **Threshold selection:** use out-of-fold predictions on train to pick the threshold that maximises F2, or meets a set minimum recall (e.g. ≥0.75) with the best precision. Explain the business reasoning: missing a churner costs more than an unnecessary retention offer.
- **Selection rule:** highest CV ROC-AUC, with PR-AUC / recall at the chosen threshold as the tie-breaker. If LR is within ~0.01 AUC of XGB, say so openly and discuss interpretability vs performance.
- ✅ **Baselines done (2026-10-03)** in `notebooks/4_baselines.ipynb` (the spec's 3.0). Notebook-level `build_pipeline(model, model_type, num_cols, cat_cols)` = `FunctionTransformer(collapse_no_internet)` → `FunctionTransformer(add_features)` → `build_preprocessor` → model; `StratifiedKFold(5, shuffle=True, random_state=42)`; `cross_validate` with ROC-AUC, PR-AUC (`average_precision`), precision, recall, F1, accuracy + train ROC-AUC. CV results (Phase 4 feature set): Dummy 0.500 / 0.265 PR-AUC (= churn rate) / 0.735 accuracy; **LR 0.847 / 0.664** (train 0.852); LR balanced 0.847 / 0.662 with recall 0.54 → 0.80; DT unlimited 0.659 (train 1.000, overfits); DT max_depth=5 balanced 0.827; XGB default 0.821 (train 0.987, overfits), XGB balanced 0.821. Takeaways: balancing moves threshold metrics, not ranking (AUC); untuned trees overfit, so the LR vs XGB verdict waits for tuning; the signal is mostly additive (EDA), which favours LR.
  - Caveat: `class_weight="balanced"` makes LR's probabilities uncalibrated (a 0.6 score ≠ 60% churn). Matters if the app shows a probability; decide in steps 7–8 (balanced vs unweighted + tuned threshold).
- ✅ **Feature-set ablation (2026-10-03)**, CV ROC-AUC / PR-AUC changes vs the Phase 4 set, LR (XGB balanced as a second opinion); rule: < ~0.003 is noise, ties go to the simpler set:
  - `-tenure_group`: LR −0.0031 / −0.0042 (the only clear change) → **keep** (captures the first-6-months kink).
  - `+TotalCharges`: −0.0000 / −0.0000 → **stays dropped** (confirms the EDA decision).
  - `+dropped` (TotalCharges, gender, PhoneService, StreamingMovies): −0.0001 / +0.0010 → **all stay dropped**.
  - `-has_family`: +0.0004 / −0.0000 → **removed**.
  - `num_services` vs the 5 add-on binaries: binaries only = identical to both for LR (exact sum); count only −0.0007 / −0.0021; XGB best with binaries only (+0.0023 / +0.0075) → **`num_services` removed**; the binaries are also more actionable in SHAP.
  - **Decision (2026-10-03): final features** `NUM_COLS = [tenure, MonthlyCharges]`, `CAT_COLS` = 14 (the Phase 4 list minus `has_family`) → LR 29 / tree 38 columns. Final set CV: LR 0.8478 / 0.6635, XGB balanced 0.8237 / 0.6245. `add_features` still builds all three columns; the unused two are dropped by `remainder="drop"`.
- ✅ **Tuning, threshold and model choice (2026-10-03)** in `notebooks/5_tuning_final.ipynb` (steps 6–8):
  - Optuna (TPE, seed 42; 50/50/100 trials), tuned CV ROC-AUC: LR 0.8483 (baseline 0.8478), DT 0.8327, **XGB 0.8507** (baseline 0.8237; best params are strongly regularised: depth 2, lr 0.016, 750 trees, subsample/colsample ≈ 0.6, reg_lambda 7.9). LR best: `C ≈ 0.069`, `l1_ratio ≈ 0.23` (8 of 29 coefficients exactly 0).
  - Threshold (OOF, `cross_val_predict`): balanced and unweighted give the same precision/recall at their own thresholds; balancing only shifts the threshold (0.31 → 0.55) and worsens Brier (0.134 → 0.163) → **train unweighted, handle imbalance with the threshold**. Rule: **best precision with recall ≥ 0.75** (LR: threshold ≈ 0.307, precision 0.551, flags 36% of customers). F2 rejected: it flags 50% for ~193 extra churners at a marginal precision of ~26% (= base rate).
  - Selection (15 fresh paired folds, `RepeatedStratifiedKFold(5, 3, random_state=7)`): XGB − LR = +0.0013 ROC-AUC (11/15 folds), +0.0068 PR-AUC (14/15), +0.006 precision at recall ≥ 0.75 (≈12 fewer wasted offers per 2,000 contacts). Consistent but far below the practical bar (≥0.01 AUC or ≥0.02 precision).
  - **Decision (2026-10-03): final model = unweighted elastic-net LR** (`solver="saga"`), threshold ≈ 0.307; reported openly that XGB was within 0.0013 AUC. Reasons: interpretability (coefficients = log-odds effects), simplicity, 14× faster fit, cheap exact SHAP.
  - Note: `saga` shuffles data, so set `random_state` on the final `LogisticRegression` for a reproducible artifact.
- ✅ **Final (2026-10-03):** refit the winning pipeline on the full train set → evaluate once on test → save `models/churn_model.joblib` (pipeline) + `models/metadata.json` (model name, threshold, test metrics, feature list, training date, library versions). Write `reports/model_comparison.md` with the CV + test table and the ROC/PR curves.
  - Test (1,409 customers, evaluated once): ROC-AUC 0.846, PR-AUC 0.651, precision 0.547, recall 0.765, 37% flagged (CV/OOF: 0.848 / 0.662 / 0.551 / 0.751). Confusion matrix: TN 798, FP 237, FN 88, TP 286. Strongest effects (odds ratios): month-to-month contract 2.01 vs two-year 0.55, `tenure_group` 0–6 1.74, fibre optic 1.59, online security 0.66, tech support 0.70.
  - Built as code: `uv run python -m churn.modeling.train` (≈3–4 min; `--n-trials N` for a quick run) reproduces the notebook's parameters, threshold and test metrics. `churn/evaluate.py` holds the metric/threshold/plot helpers.
  - `churn/modeling/predict.py`: `load_model()` (cached; warns on a scikit-learn version mismatch), `validate()`, `predict(df)`, `predict_one(dict)`. Rows with tenure outside 0–72, missing/non-numeric numbers or category levels the fitted encoder never saw are returned as "Not scored" with a reason. Accepts the raw Kaggle format (SeniorCitizen 0/1) and the cleaned one. CLI: `uv run python -m churn.modeling.predict [--input csv] [--output csv]`.

## 6. Explainability (notebook 5.0) — deferred (user's decision, 2026-10-03)
- SHAP (`TreeExplainer` for XGB/DT, `LinearExplainer` for LR) on the transformed test set: summary/beeswarm plot, top-feature bar chart, and 2–3 waterfall plots for individual customers.
- Compare the SHAP results with the EDA insights (they should agree).

## 7. Streamlit app (`app/streamlit_app.py`): ✅ done (2026-10-03)
- Loads the artifact + metadata once (`@st.cache_resource`).
- Uses `churn.modeling.predict` (`load_model`, `predict_one`, `predict`) for all scoring. The dropdown options come from the fitted `OneHotEncoder`'s categories, so the form can only offer levels the model was trained on.
- **Single customer:** sidebar inputs for all 15 raw fields → churn probability, a risk label based on the saved threshold, and the top reasons. **No `st.form`** (decision 2026-10-03): the score updates live on every change, and the 5 add-on dropdowns are disabled (forced to "No") when `InternetService == "No"`.
- **Top reasons (decision 2026-10-03, SHAP still deferred):** for the LR, each feature's push on the log-odds = coefficient × transformed value (`model[:-1].transform`, so no feature logic is duplicated), with one-hot columns summed back to their feature. The top 5 by |push| are shown, hiding features with |push| < 0.01 (elastic-net zeroed some coefficients). Checked: intercept + contributions reproduces `predict_proba` exactly. The helper `top_reasons()` lives in the app, not in `churn/`.
- **`tenure` is limited to 0–72 months** (the dataset's range; decision 2026-10-02, see section 4): the form input has `min_value=0, max_value=72`.
- **Batch:** upload a CSV (raw Kaggle format works) → summary metrics, table sorted by risk, downloadable scored CSV, plus a one-row template CSV. Rows with `tenure` outside 0–72, bad numbers or unknown categories are flagged "Not scored" with a `problem` reason; a missing column shows an error.
- **About tab:** metrics table and model card taken from `metadata.json`.
- The feature engineering lives in the saved pipeline (or in `churn.features`, imported), so the app and training transform data the same way.
- Verified with `streamlit.testing.v1.AppTest` (renders without exceptions, re-scores when inputs change, add-ons disabled with no internet), batch scoring on the raw CSV (7,043 scored, bad rows flagged) and a headless server health check (`/_stcore/health` → `ok`).

## 8. Docker (`Dockerfile`): ✅ done (2026-10-03)
- Written 2026-10-03 (`Dockerfile` + `.dockerignore`, commented line by line). uv 0.12.10 is copied from `ghcr.io/astral-sh/uv`; dependencies are installed in a separate layer (`--no-install-project`) before the code is copied, for build caching. Claude's test build: health check `ok`, runs as `appuser`, model loads; the user then built and ran it and checked the app at http://localhost:8080. Image 1.98 GB on disk (456 MB compressed) after moving xgboost/shap/seaborn to dev (section 1).
- `python:3.12-slim`; install only the runtime deps with `uv sync --frozen --no-dev`; copy `churn/`, `app/`, `models/`. The app must not import `churn.modeling.train` (Optuna is a dev dependency).
- Run as a non-root user; `CMD streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true`.
- `.dockerignore`: .venv, data/, notebooks/, reports/, references/, tests/, docs/, archive.zip, interview_prep.md, .git, .claude, caches.
- Local check: `docker build -t churn-app .` then `docker run --rm -p 8080:8080 churn-app` → http://localhost:8080.

## 9. GCP Cloud Run deployment: ✅ done (2026-10-03)
**Live:** https://churn-app-1012735950104.asia-south1.run.app (revision `churn-app-00001-6q4`). Project `project-ddf6c9f3-f5d6-49e3-bb0` (number 1012735950104), region `asia-south1` (Mumbai, set as `run/region`), billing enabled.
1. `gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com`
2. `gcloud artifacts repositories create churn-repo --repository-format=docker --location=asia-south1`
3. One-time IAM fix: new projects no longer give the Compute Engine default service account (which Cloud Build runs as) Editor, so the first build failed with `storage.objects.get` denied. Fix: `gcloud projects add-iam-policy-binding <project> --member="serviceAccount:<project-number>-compute@developer.gserviceaccount.com" --role="roles/cloudbuild.builds.builder"` (read source, write logs, push to Artifact Registry).
4. `gcloud builds submit` builds with **`cloudbuild.yaml`** (user's choice over a long `--tag`; one `docker build` step + `images:` push; substitutions `_REGION=asia-south1`, `_TAG=v1`, `$PROJECT_ID` built in; timeout 1200s). New version: `gcloud builds submit --substitutions=_TAG=v2`. Uploads are filtered by **`.gcloudignore`** (gcloud ignores `.dockerignore`): only the ~20 files the Dockerfile needs.
5. `gcloud run deploy churn-app --image asia-south1-docker.pkg.dev/<project>/churn-repo/churn-app:v1 --region asia-south1 --allow-unauthenticated --memory 1Gi --min-instances 0 --max-instances 2 --session-affinity --timeout 3600`
   - `--session-affinity`: Streamlit keeps each session in one container's memory; without it a CSV upload can hit the other instance and fail.
   - `--timeout 3600`: Streamlit holds a WebSocket open; the 300 s default would force reconnects.
   - min-instances 0 = scale to zero (no cost when idle, cold start of a few seconds); max 2 caps cost. Startup CPU boost is on by default.
- Verified: `/_stcore/health` returns `ok` on the live URL; settings confirmed with `gcloud run services describe`. Still to do: a README screenshot, and the user's same-customer check (about 80% High risk, see section 11).

## 10. Implementation order
1. ✅ Scaffold CCDS, `git init`, uv env on 3.12, move raw CSV. (Data dictionary moves to Phase 2.)
2. ✅ `dataset.py` + notebook 1.0 + data dictionary → 3. ✅ EDA notebook 2.0 (insights written; figures saved after modelling) → 4. ✅ `features.py` (`collapse_no_internet`, `add_features`, `build_preprocessor`; built and checked in `3_feature_engg.ipynb`)
5. ✅ Baselines + feature ablation (`4_baselines.ipynb`) → 6. ✅ Optuna + threshold + final model (`5_tuning_final.ipynb`, `train.py`, `evaluate.py`, `predict.py`) → 7. SHAP (5.0): ⏸ deferred
8. ✅ Streamlit app (`app/streamlit_app.py`) → 9. ✅ Docker → 10. ✅ Cloud Run → 11. **Next:** README + tests polish (the pytest suite is deferred to here).

## 11. Verification
- `uv run pytest` (deferred, not written yet): the cleaning output has no nulls and a numeric TotalCharges (in the interim data; it is not a model feature); the pipeline fits/predicts on a sample; `predict.py` returns probabilities in [0,1] and respects the threshold.
- `uv run python -m churn.dataset && uv run python -m churn.modeling.train` rebuilds the artifact from the raw CSV and gives the same metrics (fixed seed).
- Notebooks run top to bottom (`jupyter nbconvert --execute`).
- Sanity target: tuned models should reach test ROC-AUC ≈ 0.83–0.85 on this dataset; much higher suggests leakage.
- `uv run streamlit run app/streamlit_app.py` works locally → the same in `docker run` → the Cloud Run URL gives the same prediction for the same test customer (tenure 2, MonthlyCharges 90, Month-to-month, Fiber optic, Electronic check, PaperlessBilling Yes, all else No → P(churn) ≈ 0.80, High risk).

## Interview talking points to capture in the README
Why not accuracy · how leakage was prevented · how the threshold was chosen · LR vs XGB trade-off · top churn drivers (from SHAP) · what I would do next (cost-sensitive threshold using customer value, monitoring for drift, retraining).
