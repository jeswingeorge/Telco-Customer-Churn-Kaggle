"""Metrics (precision, recall, F1, ROC-AUC, PR-AUC), threshold selection, and curves."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn import config


def run_cv(pipe, name, X, y):
    """Run 5-fold CV on a pipeline; return one summary row for the comparison table."""
    # Define metrics relevant for imbalanced data
    scoring_metrics = config.SCORING_METRICS
    
    ## Run cross-validation
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    cv = pd.DataFrame(cross_validate(pipe, X, y, cv=skf, scoring=scoring_metrics, n_jobs=-1, return_train_score=True))

    ### build the summary as a dictionary for model
    row = {
        "model": name,
        "roc_auc": cv['test_roc_auc'].mean(),          # mean of the "test_roc_auc" column
        "roc_auc_std": cv['test_roc_auc'].std(),      # std of the same column
        "pr_auc": cv['test_pr_auc'].mean(),           # mean of "test_pr_auc"
        "pr_auc_std": cv['test_pr_auc'].std(),
        "precision": cv['test_precision'].mean(),        # mean of "test_precision"
        "recall": cv['test_recall'].mean(),           # mean of "test_recall"
        "f1": cv['test_f1'].mean(),               # mean of "test_f1"
        "gap_roc_auc": cv['train_roc_auc'].mean() - cv['test_roc_auc'].mean(),      # mean of train_roc_auc minus mean of test_roc_auc
        "fit_time": cv['fit_time'].mean(),         # mean of "fit_time"
    }
    return row
