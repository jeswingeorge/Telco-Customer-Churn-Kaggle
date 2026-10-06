# Phase 11: Cycle 2 Improvements

**CRISP-DM:** back to Modeling → Evaluation → Deployment (the second loop) · **Status:** ⬜ Not started (after cycle 1 is deployed) · **Where:** `notebooks/7_cycle2_experiments.ipynb`

## Goal
Test improvements with evidence, and keep only those that clearly help. Each experiment is compared with the cycle 1 model using the **same CV folds and metric**, and the test set is still not used for choosing.

## Experiments
- [ ] **1. Optuna vs cycle 1 search.** Tune the top 1–2 models with Optuna (50–100 trials, objective = mean CV ROC-AUC, the same `skf`). Record: the best score and search time vs the cycle 1 GridSearch/RandomizedSearch. Question to answer: is the gain worth the extra complexity?
- [ ] **2. SMOTE experiment.** Use an `imblearn` Pipeline (`uv add imbalanced-learn`) so SMOTE runs **inside each training fold only**. Compare it with class weights on PR-AUC, recall at the chosen threshold, and calibration. Purpose: back up (or overturn) the cycle 1 decision not to resample.
- [ ] **3. Optional:** CatBoost with native categorical features (no one-hot); Evidently for drift reports.

## Rules
- Redeploy only if an experiment is clearly better (beyond the CV std) **and** still explainable. Otherwise, keep cycle 1 and report the result anyway: "Optuna gained +0.00X ROC-AUC" is a good interview answer either way.
- If the model changes, rerun phase 6 (new threshold and metadata) and phase 7 (new reference profile).

## Done when
- [ ] A README section "Cycle 2: what changed and did it help?" with a before/after table.
