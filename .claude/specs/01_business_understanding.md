# Phase 1: Business Understanding

**CRISP-DM:** Business Understanding · **Status:** ✅ Done (one open item for phase 6) · **Where:** this file, README

## Goal
Turn "customers are leaving" into a prediction task with a clear way to judge success.

## Problem framing
- [x] **Business objective:** find customers likely to churn *before* they leave, so the retention team can target offers (discounts, contract upgrades, tech support) at the right people.
- [x] **ML task:** binary classification. Target `Churn` (1 = left within the last month). One row = one customer (7,043 rows, Kaggle `blastchar/telco-customer-churn`, IBM sample data).
- [x] **Who uses it:** a retention analyst scores one customer (form) or a list of customers (CSV upload) in the Streamlit app.

## Why accuracy is the wrong metric
- [x] About 26.5% of customers churn. A model that always predicts "stays" is 73.5% accurate and finds no churners. So we use Precision, Recall, F1, ROC-AUC and PR-AUC (see `00_overview.md`).

## Cost of errors
- [x] **False negative (missed churner):** the customer leaves and we lose their future revenue. **Expensive.**
- [x] **False positive (unneeded offer):** we give a discount to someone who would have stayed. **Cheaper.**
- [x] So recall matters more than precision, but not without limit: the team has a budget and limited capacity. The decision threshold balances the two (phase 6).

## Success criteria
- [x] Model: test ROC-AUC ≈ 0.83–0.85 is realistic for this dataset (much higher means leakage). Recall at the chosen threshold must fit the team's capacity.
- [x] Business: explain *who* churns and *why* (EDA insights + SHAP), and give a ranked target list.
- [x] Delivery: a working public app URL.

## Open questions
- [ ] **Cost assumptions for the threshold (needed in phase 6, step T1):**
  - cost_FP: the cost of a retention offer (e.g. a discount in $)
  - cost_FN: lost revenue ≈ average `MonthlyCharges` × months a saved customer stays × the offer success rate
  - capacity: the % of customers the team can contact per month
  
  These are stated as assumptions (the dataset has no cost data).
