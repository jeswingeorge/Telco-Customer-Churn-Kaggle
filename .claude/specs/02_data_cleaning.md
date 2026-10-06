# Phase 2: Data Cleaning

**CRISP-DM:** Data Understanding + Data Preparation · **Status:** ✅ Done (2026-10-02) · **Where:** `notebooks/1_data-explore.ipynb`, `churn/dataset.py`, `references/data_dictionary.md`

## Goal
Turn the raw CSV into a clean, correctly typed table that every later step can trust.

## Inputs → Outputs
- **In:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 × 21)
- **Out:**
  - `data/interim/telco_customer_churn_interim.parquet` (written by notebook 1, used by the EDA notebook)
  - `data/processed/telco_churn_clean.parquet` = `config.CLEAN_DATA_FILE` (written by `uv run python -m churn.dataset`; **the modelling input**, 7,043 × 21, no nulls)
  - `references/data_dictionary.md` (all 21 columns: type, values, meaning, quirks, model use)

## Steps
- [x] Univariate look at every column. Notes:
  - `customerID` is unique per row.
  - Gender is ≈ 50/50; ~16% are senior; ~52% have a partner; ~30% have dependents.
  - `tenure` is U-shaped, with a pile-up at the 72-month cap.
  - `TotalCharges` is right-skewed (≈ tenure × MonthlyCharges).
- [x] **Class imbalance check:** churn is 26.5% / 73.5%. That is moderate, not severe. It sets the metric choice (no accuracy) and the handling in phase 5 (class weights + threshold, no resampling).
- [x] `TotalCharges` → numeric with `pd.to_numeric(errors="coerce")`, which turns 11 blank strings into NaN. All 11 rows have `tenure == 0` (none churned, none senior, all have dependents, 10/11 on two-year contracts): new customers not yet billed → **filled with 0**, not dropped or imputed.
- [x] Duplicates: none. 22 rows become identical once `customerID` is removed (different customers with the same profile) → kept.
- [x] `churn/dataset.py` (written by the user from a skeleton): `load_raw()` → `clean()` → `save()`, with paths from `churn.config`.
  - `clean()`: TotalCharges fix, `Churn` → 1/0, `SeniorCitizen` → Yes/No.
- [x] Data dictionary written.

## Decisions
| Date | Decision | Why |
|---|---|---|
| 2026-10-01 | Save as **parquet** (`engine="pyarrow"`, `index=False`) | Keeps dtypes (no re-casting on load), smaller and faster than CSV/Excel. Trade-off: can't be opened in Excel. (Excel was tried briefly on 2026-09-30.) |
| 2026-10-02 | `Churn` → 1/0 | sklearn metrics default to `pos_label=1`; XGBoost rejects string labels. |
| 2026-10-02 | `SeniorCitizen` 0/1 → Yes/No | Categorical like the other binary columns (never scaled as a number); the app shows a Yes/No choice. |
| 2026-10-02 | **No columns dropped in the parquet** | `customerID` labels batch predictions; drops stay testable with CV; the Pipeline selects features from `config`. |
| 2026-10-02 | Drop `customerID` from the *model* | Identifier, no predictive value (listed in `config.DROP_COLS`). |

## Lessons
- pandas v3 Copy-on-Write: `df[col].fillna(..., inplace=True)` silently does nothing. Assign back instead.

## Leftover
- [ ] The comment on `load_raw()`'s `read_csv` line ("blank strings become NaN") is inaccurate: `to_numeric` does that. A one-line fix for the user.
