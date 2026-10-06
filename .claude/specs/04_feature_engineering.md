# Phase 4: Feature Engineering

**CRISP-DM:** Data Preparation · **Status:** ✅ Done (2026-10-03) · **Where:** `notebooks/3_feature_engg.ipynb`, `churn/features.py`, `churn/config.py`

## Goal
Build the engineered features and the preprocessing so they run inside an sklearn Pipeline, identically in training and in the app.

## Inputs → Outputs
- **In:** `config.CLEAN_DATA_FILE`
- **Out:** in `churn/features.py`:
  - `collapse_no_internet`
  - `add_features`
  - `build_preprocessor`

  Plus the column lists in `config.py` (`NUM_COLS`, `CAT_COLS`, `TENURE_BINS`, `ADDON_COLS`).

## Steps
- [x] Built and checked each feature in the notebook, then moved them into `churn/features.py` (written by the user from TODO skeletons). Output is identical to the notebook versions, has no nulls, and doesn't modify the input.
- [x] `collapse_no_internet(X)`: "No internet service" → "No" in `config.NO_INTERNET_COLS`. Stateless; a `FunctionTransformer` step. A module-level function, so joblib can save it.
- [x] `add_features(X)` always adds all three features below. The column lists decide which ones a model uses.
  - `tenure_group`: fixed bins `[-1, 6, 12, 24, 48, 72]` → `0-6 / 7-12 / 13-24 / 25-48 / 49-72` (0–6 → 7–12 is the biggest drop, 52.9% → 35.9%).
  - `num_services`: count of `"Yes"` over `config.ADDON_COLS` (5 columns).
  - `has_family`: "Yes" if Partner or Dependents.
- [x] `build_preprocessor(model_type, num_cols=None, cat_cols=None)` returns an unfitted `ColumnTransformer`:
  - `remainder="drop"`, so `DROP_COLS` and the target never reach the model.
  - `set_output(transform="pandas")` keeps column names for coefficients and SHAP.
  - `"lr"`: `StandardScaler` + `OneHotEncoder(handle_unknown="ignore", drop="if_binary")` → **31 columns**.
  - `"tree"`: numerics passthrough + `OneHotEncoder` without `drop` → **41 columns**.
  - Defaults: `config.NUM_COLS` (`tenure`, `MonthlyCharges`, `num_services`) and `config.CAT_COLS` (15 columns).
- [x] **Leakage review:** every step is either stateless (fixed bins and rules from `config`) or learns only in `fit()` inside the Pipeline (scaler, encoder categories with `categories="auto"`). Nothing is computed on the full dataset.

## Decisions
| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | `tenure` limited to 0–72 months | The dataset's range. Values above 72 would get NaN from `pd.cut`, so the app enforces the limit (form `max_value=72`, batch rows flagged, not scored). The model is only valid inside the training range. |
| 2026-10-03 | `drop="if_binary"` for LR | Avoids the dummy-variable trap for the 10 Yes/No columns (one `_Yes` column each) and keeps every level of the multi-level columns; L2 regularisation handles the rest. `drop="first"` was rejected: with `handle_unknown="ignore"`, an unseen value would be encoded exactly like the reference level. |
| 2026-10-03 | Trees get full one-hot (no drop) | Trees don't suffer from collinearity; every level is available to split on. |
| 2026-10-03 | Categories learned in `fit` | No manual lists to maintain; learned from training folds only. |

## Finding
- `num_services == 0` churns at only 24.8% overall, because 1,526 of its 2,404 customers have no internet (7.4% churn). Internet customers with 0 add-ons churn at 54.9%. The `InternetService` dummies separate the two groups.

## To validate in phase 5 (with CV)
- [ ] With vs without `TotalCharges`
- [ ] With vs without `has_family`
- [ ] With vs without `tenure_group`
- [ ] For LR: `num_services` vs the 5 add-on binaries (an exact sum, so use one or the other)
