# Phase 3: Exploratory Data Analysis

**CRISP-DM:** Data Understanding · **Status:** ✅ Done (2026-10-02) · **Where:** `notebooks/2_bi_multivariate_analysis.ipynb`, `churn/stats.py`

## Goal
Find what drives churn, decide which columns the model should use, and write business insights.

## Inputs → Outputs
- **In:** `data/interim/telco_customer_churn_interim.parquet`
- **Out:** feature drop decisions (`config.DROP_COLS`), the tenure shape conclusion, 7 business insights (end of the notebook, short version in the README)

## Steps
- [x] Churn rate by every categorical column vs the 26.5% baseline.
- [x] Numeric distributions split by churn (tenure, MonthlyCharges, TotalCharges); tenure cohorts.
- [x] Association strength with **Cramér's V** (symmetric) and **Theil's U** (how much a feature reduces uncertainty about churn). Helpers are in `churn/stats.py`; `cramer_matrix` / `theil_matrix` only draw heatmaps. P-values are only a gate: with 7,043 rows almost everything is "significant".
- [x] Tenure log-odds check: close to linear (R² ≈ 0.92 over 12 bins) except a steep first-6-months kink. `log1p` fits worse (R² ≈ 0.86). The tenure effect lives in month-to-month customers. → LR: raw `tenure` + test `tenure_group` with CV; trees: raw `tenure`.
- [x] 7 business insights written (see below).
- [ ] Save figures to `reports/figures/` (moved to phase 6, after modelling; the user's decision).

## Decisions (all with reasons in `config.DROP_COLS`, to be confirmed with CV in phase 5)
| Date | Decision | Evidence |
|---|---|---|
| 2026-10-02 | Drop `TotalCharges`, keep `tenure` + `MonthlyCharges` | It is ≈ tenure × MonthlyCharges: the scatter is a wedge bounded by 72 × MonthlyCharges; `TotalCharges / tenure` tracks `MonthlyCharges` almost linearly. Removes collinearity (unstable LR coefficients, split tree importances). Caveat: loses the small gap from price changes. The `avg_monthly_charge` feature was dropped from the plan (≈ `MonthlyCharges`). |
| 2026-10-02 | Drop `gender` | Cramér's V = 0.000, p = 0.47; both genders sit on the baseline. |
| 2026-10-02 | Drop `PhoneService` | Fully contained in `MultipleLines` ("No phone service" level). |
| 2026-10-02 | Drop `StreamingMovies` | No churn signal beyond `StreamingTV` among internet customers (V 0.77 overall was inflated by the shared "No internet service" level; 0.43 among internet customers). |
| 2026-10-02 | Collapse "No internet service" → "No" in the 5 add-on columns | Identical to `InternetService == "No"`; done inside the Pipeline (phase 4). |
| 2026-10-02 | Keep all 4 `PaymentMethod` levels | EDA finding (see the notebook); electronic check is the high-risk level (45.3%), the automatic methods sit at 15–17%. |

## Business insights (summary)
1. **Contract** is the biggest driver: month-to-month 42.7% vs two-year 2.8%.
2. **The first 6 months:** 52.9% churn; 55% of churners leave in year 1.
3. **Fibre:** 41.9% vs DSL 19.0%, at every tenure.
4. **Electronic check:** 45.3%, even within month-to-month.
5. **Add-ons:** 55% churn with none → 5% with all five (internet customers); security and tech support matter most.
6. **Segment:** month-to-month + fibre + electronic check = 18.6% of customers, **42% of churners** (60.4% churn rate).
7. **Seniors and single customers** churn more, largely through their contract and payment choices.

## Findings handed to phase 4
- Add-on count is strong, but it is an exact sum of the 5 binaries → for LR use one or the other.
- `has_family` probably loses information (Partner and Dependents each lower churn within the other's groups) → CV candidate only.
