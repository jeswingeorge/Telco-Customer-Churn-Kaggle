"""Metrics (precision, recall, F1, ROC-AUC, PR-AUC), threshold selection, and curves.

Pure functions on (y_true, proba): no model fitting happens here, so they work the same on
out-of-fold predictions (to choose the threshold) and on the test set (to report it once).
Prototyped in notebooks/5_tuning_final.ipynb, steps 7-9.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

MIN_RECALL = 0.75  # business target: catch at least 3 of every 4 churners


def pick_thresholds(y_true, proba, min_recall: float = MIN_RECALL, beta: float = 2) -> dict:
    """Candidate thresholds from (out-of-fold) probabilities.

    "recall75": the threshold with the best precision among those with recall >= min_recall
                (the rule we use: easy to explain, avoids F2's near-random extra contacts).
    "f2":       the threshold that maximises F-beta (beta=2 weights recall 4x precision).
    Never call this on test predictions: the threshold must be chosen without the test set.
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]  # the last (P=1, R=0) point has no threshold
    fbeta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    ok = rec >= min_recall  # thresholds that meet the recall floor
    return {
        "recall75": float(thr[ok][np.argmax(prec[ok])]),
        "f2": float(thr[np.argmax(fbeta)]),
    }


def threshold_metrics(y_true, proba, threshold: float) -> dict:
    """Ranking metrics (threshold-free) + metrics at the chosen threshold, rounded to 4 dp."""
    y_hat = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, y_hat),
        "recall": recall_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "f2": fbeta_score(y_true, y_hat, beta=2),
        "accuracy": accuracy_score(y_true, y_hat),  # reference only (26.5% churn)
        "brier": brier_score_loss(y_true, proba),  # calibration + ranking, lower is better
        "flagged_share": y_hat.mean(),  # share of customers the retention team would contact
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def plot_test_curves(y_true, proba, threshold: float, name: str = "model", path: Path | None = None):
    """Confusion matrix at the threshold + ROC and precision-recall curves; optionally saved."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int),
        display_labels=["Stay", "Churn"], ax=axes[0], colorbar=False,
    )
    axes[0].set_title(f"Confusion matrix (threshold {threshold:.3f})")
    RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=axes[1], plot_chance_level=True)
    axes[1].set_title("ROC curve (test)")
    PrecisionRecallDisplay.from_predictions(
        y_true, proba, name=name, ax=axes[2], plot_chance_level=True
    )
    axes[2].set_title("Precision-recall curve (test)")
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def df_to_md(df: pd.DataFrame) -> str:
    """DataFrame -> markdown table (pandas' to_markdown needs the extra `tabulate` package)."""
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
