# CLAUDE.md

Telco customer churn prediction: a portfolio project for data science interviews. The full spec is in [.claude/specs/project_spec.md](.claude/specs/project_spec.md). Read it before starting any phase, and update it when a decision changes.

## Current state
- CCDS folders are created, and the `churn/` package has empty module stubs; `churn/config.py` holds paths and constants. The package is installed in editable mode through the `uv_build` backend in `pyproject.toml`, so notebooks can `import churn`.
- Raw data is at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle `blastchar/telco-customer-churn`, 7,043 rows).
- Phase 1 (scaffolding) is committed. `pyarrow` is in the runtime deps for parquet I/O (`openpyxl` is also installed but no longer needed).
- Phase 2 (data cleaning) is **done**. `notebooks/1_data-explore.ipynb` has the univariate analysis and writes `data/interim/telco_customer_churn_interim.parquet` (used by the EDA notebook).
- `churn/dataset.py` is built (`uv run python -m churn.dataset`): `load_raw()` → `clean()` (TotalCharges → numeric + 11 blanks → 0, `Churn` → 1/0, `SeniorCitizen` → Yes/No) → `save()` to `config.CLEAN_DATA_FILE` (`data/processed/telco_churn_clean.parquet`, 7,043 × 21, no nulls), **the input for modelling**. Columns are **not** dropped in the parquet; the pipeline selects features. Small leftover: the comment on `load_raw()`'s `read_csv` line ("blank strings become NaN") is inaccurate; `to_numeric` does that.
- pandas here is v3 with Copy-on-Write: `df[col].method(..., inplace=True)` silently does nothing. Always assign back (`df[col] = df[col].method(...)`).
- Cleaned data is saved as **parquet** (`.parquet`, user's decision; replaced an earlier Excel choice). It keeps dtypes, so `pd.read_parquet` needs no re-casting. Use `engine="pyarrow"`, `index=False`.
- Phase 3 (EDA) is **done** in `notebooks/2_bi_multivariate_analysis.ipynb` (the spec calls it `2.0-jg-eda.ipynb`); the user is moving to modelling.
  - The notebook ends with the tenure log-odds conclusion (raw `tenure` ~linear in log-odds, R² 0.92, except a steep first-6-months kink; LR: raw + test a `tenure_group`/new-customer flag; `log1p` fits worse) and 7 business insights. The README has a short version.
  - Figures are saved to `reports/figures/` **after** modelling (user's decision).
  - `churn/stats.py` holds the user's Cramér's V / Theil's U helpers (`cramer_matrix`/`theil_matrix` only draw heatmaps and return nothing, by the user's choice).
  - `references/data_dictionary.md` is written. Duplicates were checked in the univariate notebook (none; 22 identical profiles once `customerID` is removed, kept).
  - **Decision (2026-10-02):** drop `TotalCharges` from the model features and keep `tenure` + `MonthlyCharges` (it is about `tenure × MonthlyCharges`; removes collinearity). Validate with CV with/without it in the baseline phase. The planned `avg_monthly_charge` feature is removed from the spec (almost identical to `MonthlyCharges`).
  - Model-excluded columns and their reasons live in `churn.config.DROP_COLS` (customerID, TotalCharges, gender, PhoneService, StreamingMovies); they stay in the cleaned data and the pipeline doesn't select them.
  - `churn.features.collapse_no_internet` (a Pipeline step) maps "No internet service" → "No" in `config.NO_INTERNET_COLS`. The EDA notebook applies the same replace to its own `df`, so the function isn't used there.
  - EDA findings feeding Phase 4: a count of add-on services is strong (churn 55% with 0 → 5% with 5 among internet customers) but is an exact sum of the 5 binaries, so for LR use one or the other; `has_family` would lose information; keep all 4 `PaymentMethod` levels.
- **Next:** `build_preprocessor` in `churn/features.py` → baselines (notebook 3.0, loading `config.CLEAN_DATA_FILE`).
- The notebook name `1_data-explore.ipynb` differs from the spec's convention (`1.0-jg-data-cleaning.ipynb`). Ask the user before renaming it.
- `.claude/skills/churn-tutor/` is a user-invoked tutor skill (Socratic hints, business focus; no interview quizzes unless asked).
- Update this section as phases are completed.

## Working with the user
- This is interview prep: the user must be able to explain every choice. When writing code or notebooks, state *why* (e.g. why PR-AUC, why fit inside a Pipeline) in markdown cells, in comments, or in your reply.
- The user does the analysis (cleaning, EDA, modelling) themselves. Help, review and scaffold when asked; don't finish whole phases unprompted.
- **The user's background:** they have always worked in Jupyter notebooks and have **never used** a package/CCDS folder structure (`.py` modules, imports between files, `python -m`, editable installs), **Docker** or **Streamlit** (or Cloud Run/gcloud). For these, give **detailed, step-by-step help**:
  - Explain what each new file or folder is for and *why* it exists, before writing it. Compare it to the notebook way ("in a notebook you'd run cells top to bottom; here `main()` does that").
  - Give the exact commands to run (`uv run ...`), say where to run them (the repo root), and what output to expect.
  - Show how notebook code maps to module code (cells → functions, hard-coded `../data/...` paths → `churn.config`, `if __name__ == "__main__":`).
  - Introduce one new concept at a time, then verify it works together (run it, check the output) before moving on.
  - When something fails, explain the error message in plain terms, not just the fix.
- Before editing a notebook from outside VS Code, make sure the user has it closed (or tell them to use *File → Revert File* afterwards). An open notebook saved from the editor overwrites cells added on disk; this already lost an inserted cell once.

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
- Known data quirks: `TotalCharges` has 11 blank strings (all `tenure == 0`, so fill with 0); `SeniorCitizen` is 0/1 in the raw data (mapped to Yes/No in `dataset.clean()`); drop `customerID`; columns excluded from the model are listed with reasons in `config.DROP_COLS`; "No internet service" duplicates `InternetService == "No"` and is collapsed in the pipeline.
- `data/` is git-ignored; the final model artifact in `models/` **is** committed so the Docker build is self-contained.
- The Streamlit app must use the same saved pipeline as training, with no separate feature logic.
