# Phase 6: Evaluation, Threshold & Final Model

**CRISP-DM:** Evaluation · **Status:** ⬜ Not started · **Where:** `notebooks/5_evaluation_threshold.ipynb`, `churn/evaluate.py`, `churn/modeling/train.py`

## Goal
Judge the finalists in business terms, decide the decision threshold, evaluate the winner **once** on the test set, explain it, and save everything the app and monitoring need.

## Inputs → Outputs
- **In:** the 1–2 finalist Pipelines and the tuned parameters from phase 5; the cost assumptions from phase 1.
- **Out:**
  - `models/churn_model.joblib`: the fitted Pipeline
  - `models/metadata.json`: model name, threshold, threshold rule, cost assumptions, OOF + test metrics, CV-vs-test gap, feature lists, training date, library versions
  - `models/reference_profile.json`: drift baseline (phase 7, step D1)
  - Figures in `reports/figures/` (including the EDA figures delayed from phase 3); the test results added to `reports/model_comparison.md`

## Evaluation methods (what each one tells us)
- [ ] **Threshold-free:** ROC curve + ROC-AUC, PR curve + PR-AUC (the PR curve shows the precision we pay for each extra bit of recall).
- [ ] **At a threshold:** confusion matrix, precision, recall, F1, F2.
- [ ] **Probability quality:** calibration curve + Brier score (step T2). The app shows "% risk", so the probabilities must mean what they say.
- [ ] **Business view:** a cumulative gains / lift chart ("contacting the riskiest 20% of customers reaches X% of churners") and the cost curve from T4.
- [ ] **Stability:** CV fold std from phase 5 (a model with a high mean but a large std is less trustworthy).
- [ ] **Explainability:**
  - LR coefficients as odds ratios.
  - `permutation_importance` on the finalist.
  - SHAP: `TreeExplainer` for tree models, `LinearExplainer` for LR. On the transformed test set, after T9: beeswarm, top-feature bar chart, 2–3 waterfall plots for individual customers.
  - Check that SHAP agrees with the EDA insights (contract, tenure, fibre, electronic check, add-ons). If it disagrees, find out why.

## Threshold decision checklist (T1–T10)
Format: **Action.** How · Record · Pass if · Why.

- [ ] **T1. Write down the business assumptions** (the open item from phase 1). How: a markdown cell with cost_FP (the retention offer, e.g. a discount in $), cost_FN (lost revenue ≈ average `MonthlyCharges` × months a saved customer stays × the offer success rate) and capacity (the % of customers the team can contact per month). Record: the 3 numbers and where each comes from, stated as assumptions. Why: a threshold is a business decision; the model only supplies probabilities.
- [ ] **T2. Check that the probabilities are trustworthy.** How: `CalibrationDisplay.from_predictions(y_train, oof)` and `brier_score_loss` on the OOF scores from T3. Pass if: the curve is close to the diagonal. If not, wrap the model in `CalibratedClassifierCV` (inside the Pipeline) and redo T3. Why: the cost formula and the app's "% risk" assume calibrated probabilities. Class weighting tends to push scores upward, so check.
- [ ] **T3. Get out-of-fold probabilities on train.** How: `oof = cross_val_predict(final_pipeline, X_train, y_train, cv=skf, method="predict_proba")[:, 1]`. Record: a histogram of scores split by churn. Why: every customer is scored by a model that never saw them, so the scores behave like unseen data, without touching the test set.
- [ ] **T4. Sweep thresholds.** How: for `t` in `np.arange(0.05, 0.96, 0.01)`, compute precision, recall, F1, F2 (`fbeta_score(beta=2)`), the % flagged and expected cost = FN × cost_FN + FP × cost_FP. Record: a DataFrame and one plot (precision, recall and cost vs threshold).
- [ ] **T5. Apply the 6 candidate rules** to the sweep. Record: a table of rule → threshold → precision / recall / % flagged / cost.
  1. 0.5: reference only (the sklearn default, not a business choice)
  2. Max F1: precision and recall weighted equally
  3. Max F2: recall weighted twice as much as precision
  4. Recall ≥ 0.75, then the best precision: "catch at least 3 in 4 churners"
  5. Min expected cost: uses the T1 costs directly
  6. Capacity: the threshold = the score at the top-N% percentile, so exactly N% are flagged
- [ ] **T6. Choose one rule with a business reason.** Default proposal: **(5) min cost** if the T1 assumptions are defensible, otherwise **(3) max F2**. Then check that its % flagged ≤ capacity; if not, use rule (6). Record: the decision with a date and a one-line reason in the table below.
- [ ] **T7. Sensitivity check.** How: recompute rule 5 with cost ratios FN:FP = 3:1, 5:1 and 10:1. Record: how far the threshold moves and how much the cost changes. Pass if: the chosen threshold sits in a flat (stable) part of the cost curve. Interview point: with calibrated probabilities, the cost-optimal threshold ≈ cost_FP / (cost_FP + cost_FN). For example, FP $100 and FN $500 → ≈ 0.17.
- [ ] **T8. Freeze the threshold** before the test set is scored (= leakage check L8).
- [ ] **T9. Apply it once to the test set.** How: refit the final Pipeline on all of `X_train`, `predict_proba(X_test)`, apply the threshold → confusion matrix and all metrics. Record: test results next to the OOF results at the same threshold (= leakage check L9).
- [ ] **T10. Save it.** How: `metadata.json` gets `threshold`, `threshold_rule`, `cost_assumptions`, `oof_metrics_at_threshold`, `test_metrics`. The app and `predict.py` read the threshold from here and never hard-code it.

## Leakage checklist (part 2: L8–L10)
- [ ] **L8. Threshold chosen without the test set.** Pass if: T1–T8 used only OOF predictions and the threshold was frozen before `X_test` was scored.
- [ ] **L9. CV vs test gap.** How: compare the CV mean ROC-AUC / PR-AUC with the single test score. Pass if: |CV − test| ≤ ~0.02–0.03. A test score far *below* CV means overfitting during model selection; far *above* means luck or leakage. Record: both numbers in `metadata.json`.
- [ ] **L10. Sanity ceiling and fit check.** How: treat a test ROC-AUC > ~0.90 as an alarm. Add a pytest that fits the Pipeline on a train sample and asserts that the scaler's `mean_` equals that sample's means and that the encoder's categories come from that sample only. Pass if: both hold.

## Final steps
- [ ] Finish `churn/modeling/train.py` (tune → OOF → threshold → refit → save) and write `churn/modeling/predict.py` (load the artifact + metadata, `predict_proba`, apply the threshold). The user writes both from skeletons.
- [ ] Write `models/reference_profile.json` (see D1 in phase 7).
- [ ] Save figures to `reports/figures/`; add the test results and ROC/PR curves to `reports/model_comparison.md`; fill in the README "Results" section.

## Decisions (fill in as you go)
| Date | Decision | Why |
|---|---|---|
| | Final model | |
| | Calibration needed? | |
| | Threshold rule + value | |

## Done when
- T1–T10 and L8–L10 are ticked, the artifact, metadata and reference profile are saved, and `uv run python -m churn.modeling.train` rebuilds the same metrics from scratch (fixed seed).
