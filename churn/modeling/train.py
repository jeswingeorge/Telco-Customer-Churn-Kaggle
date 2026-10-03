"""Cross-validation, Optuna tuning, final fit, and saving the model artifact.

Run from the repo root:  uv run python -m churn.modeling.train
Quick smoke test:         uv run python -m churn.modeling.train --n-trials 5

The script version of notebooks/5_tuning_final.ipynb (steps 6-9), top to bottom:
  1. stratified 80/20 split (same seed -> same split as the notebooks)
  2. Optuna: tune LR, DT and XGBoost on mean 5-fold CV ROC-AUC (train only)
  3. compare the tuned models on fresh folds + out-of-fold (OOF) threshold metrics
  4. pick the model with the selection rule (a small AUC gain doesn't beat LR's simplicity)
  5. refit on the full train set, evaluate ONCE on test
  6. save models/churn_model.joblib + models/metadata.json + reports/model_comparison.md
Overwrites the files in models/ and reports/ each time it runs.
"""

import argparse
import json
import platform
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from churn import config
from churn.evaluate import (
    df_to_md,
    pick_thresholds,
    plot_test_curves,
    threshold_metrics,
)
from churn.features import add_features, build_preprocessor, collapse_no_internet

RS = config.RANDOM_STATE
MODEL_PATH = config.MODELS_DIR / "churn_model.joblib"
METADATA_PATH = config.MODELS_DIR / "metadata.json"
REPORT_PATH = config.REPORTS_DIR / "model_comparison.md"
FIGURE_PATH = config.FIGURES_DIR / "final_model_test_curves.png"

N_TRIALS = {"lr": 50, "dt": 50, "xgb": 100}  # XGB has 7 hyperparameters, so more trials
MODEL_TYPES = {"lr": "lr", "dt": "tree", "xgb": "tree"}  # which preprocessor each model gets
MODEL_NAMES = {
    "lr": "LogisticRegression (elastic-net, unweighted)",
    "dt": "DecisionTreeClassifier (unweighted)",
    "xgb": "XGBClassifier (unweighted)",
}
# Selection rule (step 8): a more complex model must beat LR by a practically meaningful margin.
MIN_AUC_GAIN = 0.01  # spec: within ~0.01 ROC-AUC counts as a tie
MIN_PRECISION_GAIN = 0.02  # at recall >= 0.75: ~40 fewer wasted offers per 2,000 contacts


def load_split():
    """Stratified 80/20 split of the cleaned data. The test set is only used in main() step 5."""
    df = pd.read_parquet(config.CLEAN_DATA_FILE)
    X, y = df.drop(columns=config.TARGET), df[config.TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)


def build_pipeline(model, model_type: str) -> Pipeline:
    """Raw columns in -> probability out. Everything is fitted inside CV folds (no leakage).

    The steps are module-level functions in churn.features, so joblib can save the pipeline
    and the app can load it with no separate feature code.
    """
    return Pipeline([
        ("collapse", FunctionTransformer(collapse_no_internet)),
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(model_type)),  # final features from config
        ("model", model),
    ])


def make_model(name: str, params: dict):
    """Unweighted models: imbalance is handled by the threshold, so probabilities stay calibrated."""
    if name == "lr":
        # saga supports l1_ratio (elastic-net) and shuffles the data -> random_state for repeatability
        return LogisticRegression(**params, solver="saga", max_iter=5000, random_state=RS)
    if name == "dt":
        return DecisionTreeClassifier(**params, random_state=RS)
    if name == "xgb":
        # n_jobs=1: cross_validate already runs the folds in parallel (avoids oversubscription)
        return XGBClassifier(**params, eval_metric="logloss", random_state=RS, n_jobs=1)
    raise ValueError(f"unknown model {name!r}")


def suggest_params(trial, name: str) -> dict:
    """Optuna search spaces. log=True for parameters that matter by order of magnitude."""
    if name == "lr":
        return {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),  # inverse regularisation strength
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),  # 0 = L2 ... 1 = L1
        }
    if name == "dt":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 1e-2, log=True),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(f"unknown model {name!r}")


def tuning_cv() -> StratifiedKFold:
    """The same 5 folds for every trial and for the OOF threshold, so comparisons are fair."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)


def tune(name: str, X, y, n_trials: int) -> dict:
    """Optuna TPE search maximising mean CV ROC-AUC on the training data. Returns best params."""
    import optuna  # dev dependency: only training needs it, the deployed app never imports this

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = tuning_cv()

    def objective(trial):
        pipe = build_pipeline(make_model(name, suggest_params(trial, name)), MODEL_TYPES[name])
        scores = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores["test_score"].mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS), study_name=name
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"  {name}: tuned CV ROC-AUC {study.best_value:.4f}  {study.best_params}")
    return study.best_params


def compare_models(best: dict, X, y) -> tuple[pd.DataFrame, dict]:
    """Score each tuned model on fresh folds + its OOF threshold metrics (step 8 scorecard).

    Fresh folds (5x3, a seed Optuna never saw) give a less optimistic score than the tuning
    best_value. Returns the scorecard and the per-model fold scores for paired comparisons.
    """
    fresh_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
    rows, fold_scores = {}, {}
    for name, params in best.items():
        pipe = build_pipeline(make_model(name, params), MODEL_TYPES[name])
        res = cross_validate(pipe, X, y, cv=fresh_cv, scoring=scoring, n_jobs=-1)
        fold_scores[name] = res
        oof = cross_val_predict(pipe, X, y, cv=tuning_cv(), method="predict_proba", n_jobs=-1)
        threshold = pick_thresholds(y, oof[:, 1])["recall75"]
        at_threshold = threshold_metrics(y, oof[:, 1], threshold)
        rows[name] = {
            "roc_auc": res["test_roc_auc"].mean(),
            "roc_auc_std": res["test_roc_auc"].std(),
            "pr_auc": res["test_pr_auc"].mean(),
            "brier": -res["test_brier"].mean(),  # sklearn negates losses so it can maximise
            "threshold": threshold,
            "precision": at_threshold["precision"],
            "recall": at_threshold["recall"],
            "flagged_share": at_threshold["flagged_share"],
            "fit_time_s": res["fit_time"].mean(),
        }
    return pd.DataFrame(rows).T, fold_scores


def paired_difference(fold_scores: dict, a: str, b: str, metric: str = "roc_auc") -> str:
    """Mean per-fold difference a - b on the same folds (a sanity check: folds overlap)."""
    diff = fold_scores[a][f"test_{metric}"] - fold_scores[b][f"test_{metric}"]
    half = 2 * diff.std(ddof=1) / np.sqrt(len(diff))
    return (f"{metric}: {a} - {b} = {diff.mean():+.4f} "
            f"(~95% interval {diff.mean() - half:+.4f} to {diff.mean() + half:+.4f}); "
            f"{a} wins {(diff > 0).sum()}/{len(diff)} folds")


def choose_model(scorecard: pd.DataFrame) -> str:
    """Highest ROC-AUC wins, unless its lead over LR is too small to matter in practice."""
    top = scorecard["roc_auc"].idxmax()
    if top == "lr":
        return "lr"
    auc_gain = scorecard.loc[top, "roc_auc"] - scorecard.loc["lr", "roc_auc"]
    prec_gain = scorecard.loc[top, "precision"] - scorecard.loc["lr", "precision"]
    if auc_gain >= MIN_AUC_GAIN or prec_gain >= MIN_PRECISION_GAIN:
        return top
    return "lr"  # a tie in practice -> the simpler, interpretable model


def build_metadata(name, params, threshold, oof_metrics, test_metrics, n_train, n_test) -> dict:
    """Everything the pipeline itself doesn't store: the threshold, metrics, inputs, versions."""
    return {
        "model": MODEL_NAMES[name],
        "model_key": name,
        "params": params,
        "threshold": threshold,
        "threshold_rule": "max precision with out-of-fold recall >= 0.75",
        "cv_oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        # raw columns the app must supply (tenure_group is built inside the pipeline)
        "input_columns": [c for c in config.NUM_COLS + config.CAT_COLS
                          if c not in config.ENGINEERED_FEATURES],
        "model_features": config.NUM_COLS + config.CAT_COLS,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "random_state": RS,
        "trained_on": datetime.now().astimezone().date().isoformat(),  # local date, tz-aware
        # a joblib file should be loaded with the same library versions it was saved with
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def write_report(scorecard, paired, metadata) -> None:
    """reports/model_comparison.md: CV table for all models + the single test evaluation."""
    cv_table = scorecard.drop(columns="fit_time_s").astype(float).round(4).rename_axis("model")
    test_table = pd.DataFrame({
        "cv_oof (train)": metadata["cv_oof_metrics"],
        "test": metadata["test_metrics"],
    }).rename_axis("metric")
    report = f"""# Model comparison

Generated by `uv run python -m churn.modeling.train` on {metadata["trained_on"]}.

## Cross-validation (train, 5x3 repeated stratified folds, tuned models)
Precision/recall/flagged share at each model's out-of-fold threshold
(best precision with recall >= 0.75).

{df_to_md(cv_table)}

Paired comparison on the same folds:
- {paired["roc_auc"]}
- {paired["pr_auc"]}

## Final model: {metadata["model"]}
Threshold {metadata["threshold"]:.3f}. Selection rule: highest CV ROC-AUC, but a more complex
model must beat LR by >= {MIN_AUC_GAIN} ROC-AUC or >= {MIN_PRECISION_GAIN} precision at
recall >= 0.75 (see notebooks/5_tuning_final.ipynb, step 8).

## Test set (evaluated once)

{df_to_md(test_table)}

![Test curves](figures/{FIGURE_PATH.name})
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main(n_trials: int | None = None) -> None:
    # 1. split
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train {len(X_train):,} rows, test {len(X_test):,} rows")

    # 2. tune each model (train only)
    print("Tuning with Optuna (objective: mean 5-fold CV ROC-AUC)...")
    best = {name: tune(name, X_train, y_train, n_trials or default)
            for name, default in N_TRIALS.items()}

    # 3. compare on fresh folds + OOF thresholds
    print("Comparing tuned models on fresh folds...")
    scorecard, fold_scores = compare_models(best, X_train, y_train)
    print(scorecard.astype(float).round(4).to_string())
    paired = {m: paired_difference(fold_scores, "xgb", "lr", m) for m in ["roc_auc", "pr_auc"]}
    for line in paired.values():
        print(" ", line)

    # 4. selection rule
    chosen = choose_model(scorecard)
    threshold = float(scorecard.loc[chosen, "threshold"])
    print(f"Chosen: {MODEL_NAMES[chosen]}, threshold {threshold:.4f}")

    # 5. refit on all of train, then the one and only test evaluation
    final_pipe = build_pipeline(make_model(chosen, best[chosen]), MODEL_TYPES[chosen])
    oof = cross_val_predict(final_pipe, X_train, y_train, cv=tuning_cv(),
                            method="predict_proba", n_jobs=-1)[:, 1]
    oof_metrics = threshold_metrics(y_train, oof, threshold)
    final_pipe.fit(X_train, y_train)
    p_test = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics = threshold_metrics(y_test, p_test, threshold)
    print(pd.DataFrame({"cv_oof (train)": oof_metrics, "test": test_metrics}).to_string())

    # 6. save the artifact, its metadata, the figure and the report
    metadata = build_metadata(chosen, best[chosen], threshold, oof_metrics, test_metrics,
                              len(X_train), len(X_test))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig = plot_test_curves(y_test, p_test, threshold, name=chosen, path=FIGURE_PATH)
    plt.close(fig)
    write_report(scorecard, paired, metadata)
    print(f"Saved {MODEL_PATH}, {METADATA_PATH}, {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":  # runs only when executed as a script, not when imported
    parser = argparse.ArgumentParser(description="Tune, select, fit and save the churn model.")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optuna trials per model (default: 50 LR, 50 DT, 100 XGB)")
    main(parser.parse_args().n_trials)
