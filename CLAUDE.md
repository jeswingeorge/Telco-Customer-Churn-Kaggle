# CLAUDE.md

Telco customer churn prediction: a portfolio project for data science interviews. The full spec is in [.claude/specs/project_spec.md](.claude/specs/project_spec.md). Read it before starting any phase, and update it when a decision changes.

## Current state
- CCDS folders are created, and the `churn/` package has empty module stubs; `churn/config.py` holds paths and constants. The package is installed in editable mode through the `uv_build` backend in `pyproject.toml`, so notebooks can `import churn`.
- Raw data is at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle `blastchar/telco-customer-churn`, 7,043 rows).
- Phase 1 (scaffolding) is committed. `openpyxl` was added to runtime deps (the notebook writes an `.xlsx`).
- Phase 2 (data cleaning) is **in progress** in `notebooks/1_data-explore.ipynb`: univariate analysis of all columns is done, and `TotalCharges` is converted to numeric with its 11 blanks filled with 0. Output goes to `data/interim/telco_customer_churn_interim.xlsx`.
- Still to do in Phase 2: harmonise `SeniorCitizen`, drop `customerID`, map `Churn` to 1/0, check duplicates, move the cleaning into `churn/dataset.py` (still a stub), and write `references/data_dictionary.md`.
- Cleaned data is saved as **Excel** (`.xlsx`, user's decision), not parquet. Excel loses dtypes, so re-check them after `pd.read_excel`.
- The notebook name `1_data-explore.ipynb` differs from the spec's convention (`1.0-jg-data-cleaning.ipynb`). Ask the user before renaming it.
- `.claude/skills/churn-tutor/` is a user-invoked tutor/interviewer skill (Socratic hints + interview quizzes).
- Update this section as phases are completed.

## Working with the user
- This is interview prep: the user must be able to explain every choice. When writing code or notebooks, state *why* (e.g. why PR-AUC, why fit inside a Pipeline) in markdown cells, in comments, or in your reply.
- The user does the analysis (cleaning, EDA, modelling) themselves. Help, review and scaffold when asked; don't finish whole phases unprompted.

## Stack & commands
- Python **3.12** managed with **uv**, pinned in `.python-version` (the system Python is 3.14; don't use it, since xgboost/shap wheels lag). Run everything via `uv run ...`; add deps with `uv add` (runtime) or `uv add --dev`.
- Commands (the modules are stubs until their phase is built):
  - `uv sync` installs dependencies and `churn` in editable mode
  - `uv run python -m churn.dataset` builds cleaned data
  - `uv run python -m churn.modeling.train` tunes, fits and saves `models/churn_model.joblib` + `models/metadata.json`
  - `uv run pytest` / `uv run ruff check .`
  - `uv run streamlit run app/streamlit_app.py`
  - `docker build -t churn-app . && docker run -p 8080:8080 churn-app`
- Windows machine: shell is PowerShell/Git Bash; `make` may be missing, so keep Makefile targets as thin wrappers that also document the `uv run` equivalent.
- Windows Application Control blocks executables run from uv's cache, so `uvx <tool>` fails. Install tools into the project instead (`uv add --dev`) and use `uv run`.
- Docker Desktop and gcloud are not installed yet (needed for the deploy phases).

## Project conventions
- Layout follows CCDS v2. Reusable logic goes in the `churn/` package; notebooks (`notebooks/N.0-jg-<topic>.ipynb`) import from it and tell the story.
- `RANDOM_STATE = 42` and paths live in `churn/config.py`; nothing is hard-coded elsewhere.
- **No leakage:** all preprocessing (encoding, scaling, engineered features) sits inside an sklearn `Pipeline`, fitted only on training folds. Stratified 80/20 split; the test set is evaluated once, at the end.
- **Metrics:** Precision, Recall, F1, ROC-AUC, PR-AUC and the confusion matrix. Accuracy is shown for reference only (about 26.5% of customers churn). The decision threshold comes from out-of-fold predictions on train and is saved in `metadata.json`.
- Known data quirks: `TotalCharges` has 11 blank strings (all `tenure == 0`, so fill with 0); `SeniorCitizen` is 0/1 while other binary columns are Yes/No; drop `customerID`.
- `data/` is git-ignored; the final model artifact in `models/` **is** committed so the Docker build is self-contained.
- The Streamlit app must use the same saved pipeline as training, with no separate feature logic.
