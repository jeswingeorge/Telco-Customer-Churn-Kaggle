# Phase 7: Drift Monitoring

**CRISP-DM:** Deployment (monitoring) · **Status:** ⬜ Not started · **Where:** `notebooks/6_drift_monitoring.ipynb`, then `churn/monitoring.py`

## Goal
A deployed model gets worse when customers change. We build checks that warn us *before* the model's decisions go wrong, and we prove they work. This dataset has no dates, so drift can't be measured historically. Instead we build the checks and demonstrate them on the test set (no drift) and a deliberately shifted copy (drift).

## Types of drift
- **Data drift:** the inputs change (e.g. many more fibre customers, a price rise).
- **Prediction drift:** the model's scores change (e.g. the average churn probability jumps).
- **Concept drift:** the same inputs now lead to a different outcome (e.g. a competitor's offer makes loyal two-year customers leave). Only real outcome labels can reveal this.

## Inputs → Outputs
- **In:** `models/reference_profile.json` (from phase 6), the saved Pipeline + threshold, `X_test`
- **Out:** `churn/monitoring.py` (`build_reference_profile`, `psi`, `drift_report`), drift demo tables for the README, and retrain triggers written down

## Checklist (D1–D12)
Format: **Action.** How · Record · Pass if · Why.

**Build the baseline**
- [ ] **D1. Save a reference profile at training time** (done at the end of phase 6). How: from `X_train` → `models/reference_profile.json`, containing:
  - decile bin edges and the share of rows per bin for each numeric feature
  - category shares for each categorical
  - the training churn rate
  - the OOF score deciles and the % flagged at the threshold
  
  Why: drift is always measured *against* the training data.

**Input checks (every batch, before scoring)**
- [ ] **D2. Schema check.** How: compare the batch's columns and dtypes with the raw columns the app needs. Pass if: all present with the right types; otherwise reject the batch with a clear message. Why: a renamed column would otherwise crash the app or be silently dropped.
- [ ] **D3. Range check.** How: flag and don't score rows with `tenure` outside 0–72 (the phase 4 decision); warn when `MonthlyCharges` is outside the training min–max. Record: the count of flagged rows. Why: the model is only valid inside the range it was trained on.
- [ ] **D4. Unseen-category check.** How: compare each categorical column with the categories in the reference profile; count unknown values per column. Pass if: zero unknowns; otherwise warn and show them. Why: `OneHotEncoder(handle_unknown="ignore")` silently turns an unseen value into all-zero columns, so the model would score garbage **without any error**.

**Data and prediction drift**
- [ ] **D5. PSI per feature** (Population Stability Index). How: `psi(expected, actual) = Σ (actual% − expected%) × ln(actual% / expected%)` over the reference bins or categories, with a small epsilon so empty bins don't divide by zero. Rule: **< 0.1 stable · 0.1–0.25 watch · > 0.25 investigate**. Record: a table sorted by PSI. Why: one number per feature that measures *how much* the distribution moved (an effect size).
- [ ] **D6. Statistical tests as a second opinion.** How: `scipy.stats.ks_2samp` for numerics, `chi2_contingency` for categoricals. Note: with large batches almost any tiny change becomes "significant", so **PSI drives the decision**. This is the same reasoning as using Cramér's V instead of p-values in the EDA.
- [ ] **D7. Prediction drift.** How: compare the batch's mean predicted probability and % flagged with the reference OOF values, plus PSI on the score deciles. Pass if: score PSI < 0.1. Why: a shift in scores is the earliest sign that the model's output (and the team's workload) is changing.

**Concept drift and the response plan**
- [ ] **D8. Write the label-based check** (for when real outcomes arrive). Each month: actual churn rate vs mean predicted probability, and recall/precision at the frozen threshold vs the test values from T9. Documented as a procedure, since there are no future labels here.
- [ ] **D9. Write retrain triggers** in this spec and the README:
  - PSI > 0.25 on any top-5 SHAP feature
  - score PSI > 0.25
  - recall more than 0.05 below the test value
  - more than 1% of rows with unseen categories
  
  Response: investigate → retrain on recent data → rerun phases 5–6 → redeploy.

**Demonstration and build**
- [ ] **D10. No-drift demo.** How: run D2–D7 on `X_test`. Pass if: all PSI < 0.1 and no warnings. Why: shows the checks don't raise false alarms on data from the same distribution.
- [ ] **D11. Drift demo.** How: build a shifted copy of `X_test`: `MonthlyCharges × 1.2`, 30% of DSL customers switched to fibre, and one new `PaymentMethod` value. Pass if: D3, D4, D5 and D7 flag exactly those changes and nothing else. Record: before/after tables (good README material).
- [ ] **D12. Move into the package.** Once the notebook works, write `churn/monitoring.py` from a skeleton: `build_reference_profile(X, scores, threshold)`, `psi(expected, actual)` and `drift_report(batch, profile)`. `train.py` calls `build_reference_profile`. The Streamlit batch tab (phase 8) calls `drift_report` and shows the warnings above the scores.

## Decisions (fill in as you go)
| Date | Decision | Why |
|---|---|---|
| | PSI thresholds kept at 0.1 / 0.25? | |
| | Retrain triggers | |

## Notes
- Evidently (a monitoring library) is a cycle-2 option. A hand-written PSI is easier to explain in an interview.

## Done when
- D1–D12 are ticked, both demos behave as expected, and `drift_report` runs on any batch CSV.
