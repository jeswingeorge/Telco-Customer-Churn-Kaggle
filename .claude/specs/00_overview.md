# Overview: Telco Customer Churn Prediction

**Goal:** predict which telecom customers will churn so a retention team can target offers, and show interviewers an end-to-end project: clean structure, careful data work, models judged on metrics that suit imbalanced data, a reasoned decision threshold, monitoring, and a deployed app (Streamlit → Docker → GCP Cloud Run).

**Method:** CRISP-DM for the *process*, Cookiecutter Data Science v2 for the *folder layout*.

**Guiding rule:** every step must either change a decision or be clearly explainable in an interview. If it does neither, leave it out.

## How to use these specs
- One file per phase. Read the current phase's file before starting; tick steps (`[x]`) as you go; record each decision with a date and a one-line reason.
- Status: ✅ Done · 🔄 Next / in progress · ⬜ Not started.
- When a phase finishes, update its status here and the "Current phase" line in `CLAUDE.md`.

## CRISP-DM map and status
CRISP-DM is a loop, not a line: after the first deployment we go round again (cycle 2) to improve.

```
Business Understanding ─► Data Understanding ─► Data Preparation ─► Modeling ─► Evaluation ─► Deployment
        ▲                                                                                       │
        └─────────────────────────────── cycle 2 (improve) ◄────────────────────────────────────┘
```

| # | Phase | CRISP-DM stage | Spec | Notebook / module | Status |
|---|---|---|---|---|---|
| 0 | Setup | (infrastructure) | this file, below | `pyproject.toml`, `churn/config.py` | ✅ |
| 1 | Business understanding | Business Understanding | [01_business_understanding.md](01_business_understanding.md) | n/a | ✅ |
| 2 | Data cleaning | Data Understanding + Preparation | [02_data_cleaning.md](02_data_cleaning.md) | `1_data-explore.ipynb`, `churn/dataset.py` | ✅ |
| 3 | EDA | Data Understanding | [03_eda.md](03_eda.md) | `2_bi_multivariate_analysis.ipynb` | ✅ |
| 4 | Feature engineering | Data Preparation | [04_feature_engineering.md](04_feature_engineering.md) | `3_feature_engg.ipynb`, `churn/features.py` | ✅ |
| 5 | Modelling | Modeling | [05_modelling.md](05_modelling.md) | `4_baseline_models.ipynb`, `churn/modeling/train.py` | 🔄 |
| 6 | Evaluation & threshold | Evaluation | [06_evaluation_threshold.md](06_evaluation_threshold.md) | `5_evaluation_threshold.ipynb`, `churn/evaluate.py` | ⬜ |
| 7 | Drift monitoring | Deployment (monitoring) | [07_drift_monitoring.md](07_drift_monitoring.md) | `6_drift_monitoring.ipynb`, `churn/monitoring.py` | ⬜ |
| 8 | Streamlit app | Deployment | [08_streamlit_app.md](08_streamlit_app.md) | `app/streamlit_app.py` | ⬜ |
| 9 | Docker | Deployment | [09_docker.md](09_docker.md) | `Dockerfile` | ⬜ |
| 10 | Cloud Run + polish | Deployment | [10_cloud_run.md](10_cloud_run.md) | gcloud, `tests/`, README | ⬜ |
| 11 | Cycle 2 improvements | (iteration) | [11_cycle2_improvements.md](11_cycle2_improvements.md) | Optuna, SMOTE experiment | ⬜ |

## Global rules (apply to every phase)
**No leakage:** the full checklist is in phases 5 and 6 (L1–L10).
1. **Split first:** stratified 80/20 train/test, `random_state=42`, before anything learns from the data.
2. **Everything that learns goes inside the sklearn `Pipeline`** (scaler, encoder categories, tuning), so it is fitted on training folds only. CV and search objects always receive the whole Pipeline.
3. **The threshold comes from out-of-fold predictions** on train, never from the test set.
4. **The test set is used once**, at the very end.
5. **Every feature must be known before the outcome** (no post-churn information).

**Metrics:** Precision, Recall, F1, ROC-AUC, **PR-AUC** and the confusion matrix. Accuracy is for reference only: about 26.5% churn, so a model that always says "stays" gets 73.5%. For reading scores, a random model gets ROC-AUC 0.5 and PR-AUC 0.265 (the churn rate).

**Reproducibility:** `RANDOM_STATE = 42` and all paths live in `churn/config.py`. Nothing is hard-coded elsewhere.

**Code flow:** prototype in the notebook first, then move to `churn/*.py` (the user writes the module code from a skeleton).

## Phase 0: Setup ✅
- [x] CCDS v2 layout created by hand. The `ccds` generator couldn't be used: Windows Application Control blocks executables run from uv's cache, so `uvx` fails.
- [x] uv + Python 3.12 (`.python-version`; the system Python 3.14 is avoided because xgboost/shap wheels lag).
- [x] `churn/` is installed in editable mode via the `uv_build` backend (`module-name = "churn"`, `module-root = ""`), so notebooks can `import churn`.
- [x] `.gitignore`: `/data/`, `archive.zip`, `.venv/`, caches, checkpoints, `.env`. `models/churn_model.joblib` **is** committed so the Docker build is self-contained. Empty folders hold a `.gitkeep`.
- [x] Dependencies. Runtime: pandas, numpy, scikit-learn, xgboost, joblib, shap, streamlit, matplotlib, seaborn, pyarrow. Dev: jupyterlab, ipykernel, optuna, pytest, ruff.

## Folder layout (CCDS v2)
```
Telco-Customer-Churn-Kaggle/
├── README.md                 # public face: problem, results, insights, how to run
├── CLAUDE.md                 # instructions for Claude
├── pyproject.toml / uv.lock / .python-version
├── Dockerfile  .dockerignore # phase 9
├── data/{raw,interim,processed,external}/   # git-ignored
├── models/                   # churn_model.joblib, metadata.json, reference_profile.json
├── notebooks/                # N_topic.ipynb (the user's naming style)
├── reports/figures/  reports/model_comparison.md
├── references/data_dictionary.md
├── churn/                    # source package
│   ├── config.py  dataset.py  features.py  stats.py
│   ├── evaluate.py  monitoring.py  plots.py
│   └── modeling/{train.py, predict.py}
├── app/streamlit_app.py
├── tests/
└── .claude/specs/            # these files
```

## Sanity target
Tuned models usually reach test ROC-AUC ≈ 0.83–0.85 on this dataset. A much higher score is treated as a leakage alarm.
