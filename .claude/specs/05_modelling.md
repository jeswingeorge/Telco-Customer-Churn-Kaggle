# Phase 5: Modelling

**CRISP-DM:** Modeling · **Status:** 🔄 Next · **Where:** `notebooks/4_baseline_models.ipynb`, then `churn/modeling/train.py`

## Goal
Compare models from simple to complex with 5-fold cross-validation on the training set, settle the open feature questions from phase 4, and pick 1–2 finalists for phase 6. This happens **without touching the test set**.

## Inputs → Outputs
- **In:** `config.CLEAN_DATA_FILE`; `churn.features` (`collapse_no_internet`, `add_features`, `build_preprocessor`)
- **Out:**
  - A comparison table (CV mean ± std per model), saved as `reports/model_comparison.md`
  - The final feature lists (update `config.NUM_COLS` / `CAT_COLS` if they change)
  - 1–2 tuned finalist Pipelines
  - `churn/modeling/train.py` (written by the user from a skeleton)

## How to read the scores
- **ROC-AUC:** the chance that a random churner gets a higher score than a random non-churner. A random model gets 0.5.
- **PR-AUC (`average_precision`):** precision averaged over all recall levels. A random model gets 0.265 (the churn rate). It is more informative than ROC-AUC when positives are the minority.
- Precision, recall and F1 in CV use the default 0.5 threshold, so **models are ranked on ROC-AUC / PR-AUC** (threshold-free). The real threshold is chosen in phase 6.

## Steps (the model ladder: simple → complex)
Format: **what's new / why** · preprocessor · what to record.

- [ ] **1. Setup.**
  - Load the parquet. `X` = all columns except `Churn`; `y` = `Churn`.
  - Stratified 80/20 split; `skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)`.
  - Write a small helper `make_pipeline(model, model_type, num_cols=None, cat_cols=None)` that returns `Pipeline([("collapse", FunctionTransformer(collapse_no_internet)), ("features", FunctionTransformer(add_features)), ("prep", build_preprocessor(...)), ("model", model)])`.
  - Write a helper that runs `cross_validate(pipe, X_train, y_train, cv=skf, scoring=["roc_auc", "average_precision", "precision", "recall", "f1"], return_train_score=True)` and returns one row of means ± std.
  - Why a Pipeline: the scaler and encoder are refitted inside every fold, so no validation data leaks into training.
- [x] **Leakage checks L1, L2, L3, L5** (see below) before the first model. 2026-10-10: all passed.
- [ ] **2. Logistic Regression** · `"lr"` · the interpretable reference model.
  - Default settings, then `class_weight="balanced"`.
  - **Feature-set CV comparisons happen here** (the 4 questions from phase 4): with/without `TotalCharges`, `has_family`, `tenure_group`; `num_services` vs the 5 add-on binaries. Record one row per variant, then decide and record the decisions below.
  - Read the coefficients as odds ratios (`np.exp(coef)`): "a two-year contract multiplies the odds of churn by X". Check that they agree with the EDA.
- [ ] **3. KNN and SVM** · `"lr"` (they need scaled features).
  - KNN: "similar customers behave alike". SVM: maximum-margin boundary; needs `probability=True` for `predict_proba` (slower).
  - Expect them to be no better than LR on tabular one-hot data. Being able to say *why* (distance becomes less meaningful across many one-hot columns; no native probabilities) is the interview value.
- [ ] **4. Decision Tree** · `"tree"`.
  - Default settings → record the train vs CV gap (it overfits: train AUC ≈ 1.0).
  - Then limit `max_depth` / `min_samples_leaf`. Plot a shallow tree (`plot_tree`): readable rules for the business.
- [ ] **5. Random Forest** · `"tree"`.
  - Bagging: many de-correlated trees averaged → lower variance than one tree. `class_weight="balanced"`.
- [ ] **6. XGBoost** · `"tree"`.
  - Boosting: trees built one after another, each fixing the previous errors. `scale_pos_weight = neg/pos` (≈ 2.77).
- [ ] **7. LightGBM and CatBoost** · `"tree"`.
  - `uv add lightgbm catboost` (runtime deps, since the final model may be one of them and the Docker image needs it).
  - Cycle 1 uses the same one-hot input for a fair comparison (CatBoost's native categorical handling can be tried in cycle 2).
- [ ] **8. Imbalance handling.**
  - For each model, record CV scores with and without class weighting (`class_weight="balanced"` / `scale_pos_weight` / `auto_class_weights="Balanced"` for CatBoost).
  - **No resampling (SMOTE/undersampling) in cycle 1.** The imbalance is moderate (26.5%); SMOTE blends one-hot rows into unrealistic customers; resampling distorts predicted probabilities (needed for the cost-based threshold); class weights + threshold usually reach the same recall more simply. The experiment is in cycle 2.
- [ ] **9. Tuning (cycle 1).**
  - `GridSearchCV` for LR (`C`, `penalty`), KNN (`n_neighbors`, `weights`), SVM (`C`, `gamma`), DT (`max_depth`, `min_samples_leaf`, `ccp_alpha`). Small grids are easy to explain.
  - `RandomizedSearchCV` (`n_iter` ≈ 30–50) for RF, XGB, LightGBM, CatBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`).
  - `scoring="roc_auc"`, `cv=skf`, on the **whole Pipeline** (parameters are named `model__C`, etc.). Record the best parameters, the best CV score and the search time (needed for the cycle 2 Optuna comparison).
- [ ] **Leakage checks L4, L6, L7** for every model.
- [ ] **10. Comparison table and selection.**
  - Columns: model · ROC-AUC · PR-AUC · precision · recall · F1 (each mean ± std) · train−CV gap · fit time.
  - **Rule:** highest CV ROC-AUC; PR-AUC as the tie-breaker. If LR is within ~0.01 of the best boosted model, say so openly and weigh interpretability vs performance.
  - Choose 1–2 finalists and save the table as `reports/model_comparison.md`.
- [ ] **11. Move into the package.** Once the notebook works, write `churn/modeling/train.py` from a skeleton: `main()` loads data → splits → builds the chosen Pipeline → fits → saves (the threshold and saving are finished in phase 6). Run with `uv run python -m churn.modeling.train`; `uv run ruff check churn/` must pass.

## Leakage checklist (part 1: L1–L7)
Format: **Action.** How · Record · Pass if · Why.

- [x] **L1. Target-leakage review of features.** How: a table of each model feature → "known before churn happens?" Record: the table in the notebook. Pass if: every row is "yes" (no post-churn fields; `customerID` and `DROP_COLS` excluded). Why: a feature recorded *because* the customer churned gives a perfect but useless model.
  - 2026-10-10 ✅ All 18 model features (15 raw + 3 engineered) are known at scoring time; the engineered ones use fixed rules and no target. Caveat noted: tenure is frozen at exit for churners (snapshot/censoring), not leakage.
- [x] **L2. Split before anything else.** How: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)` in the first cell after loading. Record: the shapes and the churn rate in train and test (both ≈ 26.5%). Pass if: no `fit`, `fit_transform` or statistics are computed on the full data before this cell. Why: the test set must stay unseen.
  - 2026-10-10 ✅ Train 5634 × 20, test 1409 × 20; churn rate 0.265 in both; no fit or statistics before the split.
- [x] **L3. Duplicates across the split.** How: count the identical feature profiles (without `customerID`) that appear in both train and test. Record: the count (out of the 22 known profiles). Pass if: the count is noted; they are kept because they are different customers. Why: shows awareness of near-duplicate leakage.
  - 2026-10-10 ✅ 12 of the 22 duplicate profiles span train and test (< 1% of the 1409 test rows). Kept: they are genuine, different customers.
- [ ] **L4. Everything learnable sits inside the Pipeline.** How: `cross_validate`, `GridSearchCV` and `RandomizedSearchCV` always receive the whole Pipeline and raw `X_train`. Pass if: no scaler or encoder is fitted outside the Pipeline anywhere in the notebook. Why: otherwise the scaler sees the validation folds.
- [x] **L5. Single-feature AUC scan.** How: for each raw model feature, the CV ROC-AUC of a tiny Pipeline (one-hot or scaler + LR) on that one column. Record: a table sorted by AUC. Pass if: no single feature has AUC > 0.90 (otherwise investigate it as target leakage). Why: one feature that predicts almost perfectly is the classic leakage sign. Bonus: the ranking should match the EDA (Contract, tenure at the top).
  - 2026-10-10 ✅ Max is Contract 0.739 (< 0.90), then tenure 0.735 and tenure_group 0.721, which matches the EDA. The lowest is MultipleLines at 0.525. num_services is weak alone (0.56) because 0 mixes no-internet customers with internet customers who have no add-ons; it is not a feature-selection verdict.
- [ ] **L6. Train vs CV gap.** How: `return_train_score=True`. Record: a train−CV ROC-AUC column in the comparison table. Pass if: gap < ~0.05 after tuning (the default Decision Tree will fail this on purpose). Why: separates overfitting from real skill.
- [ ] **L7. Tuning inside CV only.** How: search objects get the Pipeline and `X_train`. Pass if: `X_test` is not used anywhere in notebook 4. Why: tuning on the test set makes the final test score optimistic.

## Decisions (fill in as you go)
| Date | Decision | Why (CV evidence) |
|---|---|---|
| | `TotalCharges`: in / out | |
| | `has_family`: in / out | |
| | `tenure_group`: in / out | |
| | LR: `num_services` or the 5 binaries | |
| | Class weighting: on / off per model | |
| | Finalist(s) | |

## Done when
- Every model has a row in the comparison table, the 4 feature questions are decided, L1–L7 are ticked, and 1–2 finalists are chosen with a written reason.
