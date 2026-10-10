# CLAUDE.md

Telco customer churn prediction: a portfolio project for data science interviews, run with **CRISP-DM** (process) and **Cookiecutter Data Science v2** (layout).

## Specs
- Index and status of every phase: [.claude/specs/00_overview.md](.claude/specs/00_overview.md). There is one spec file per phase (`01_` … `11_`).
- Before working on a phase, read its spec. Tick steps (`[x]`) as they are done and record each decision there (date + one-line reason), not in this file.
- When a phase finishes: set its status in `00_overview.md` and update "Current phase" below.
- Every planned step must either change a decision or be clearly explainable in an interview. No ritual steps.

## Current phase
**Phase 5, Modelling** → [.claude/specs/05_modelling.md](.claude/specs/05_modelling.md) (notebook `notebooks/4_baseline_models.ipynb`). Phases 0–4 are done. In phase 5, setup, leakage checks L1/L2/L3/L5 and step 2 (LR + feature decisions) are done; next is step 3 (KNN/SVM). The CV helper is `churn.evaluate.run_cv(pipe, name, X, y)`, and CV settings live in config (`N_SPLITS`, `SCORING_METRICS`).

## Working with the user
- This is interview prep: the user must be able to explain every choice. When writing code or notebooks, state *why* (e.g. why PR-AUC, why fit inside a Pipeline) in markdown cells, in comments, or in your reply.
- The user does the analysis (cleaning, EDA, modelling) themselves. Help, review and scaffold when asked; don't finish whole phases unprompted.
- **The user's background:** they have always worked in Jupyter notebooks and have **never used** a package/CCDS folder structure (`.py` modules, imports between files, `python -m`, editable installs), **Docker** or **Streamlit** (or Cloud Run/gcloud). For these, give **detailed, step-by-step help**:
  - Explain what each new file or folder is for and *why* it exists, before writing it. Compare it to the notebook way ("in a notebook you'd run cells top to bottom; here `main()` does that").
  - Give the exact commands to run (`uv run ...`), say where to run them (the repo root), and what output to expect.
  - Show how notebook code maps to module code (cells → functions, hard-coded `../data/...` paths → `churn.config`, `if __name__ == "__main__":`).
  - Introduce one new concept at a time, then verify it works together (run it, check the output) before moving on.
  - When something fails, explain the error message in plain terms, not just the fix.
- **Notebook first, then package:** the user prototypes logic in the notebook and usually writes `churn/*.py` themselves from a skeleton. Check with them before editing `churn/*.py`.
- `.claude/skills/churn-tutor/` is a user-invoked tutor skill (Socratic hints, business focus; no interview quizzes unless asked).

## Stack & commands
- Python **3.12** managed with **uv**, pinned in `.python-version` (the system Python is 3.14; don't use it, since xgboost/shap wheels lag). Run everything via `uv run ...`; add deps with `uv add` (runtime) or `uv add --dev`.
- Run from the repo root:
  - `uv sync`: installs dependencies and `churn` in editable mode
  - `uv run python -m churn.dataset`: raw CSV → `data/processed/telco_churn_clean.parquet`
  - `uv run python -m churn.modeling.train`: fits and saves `models/churn_model.joblib` + `models/metadata.json` (phase 6)
  - `uv run pytest` / `uv run ruff check .`
  - `uv run streamlit run app/streamlit_app.py`
  - `docker build -t churn-app . && docker run -p 8080:8080 churn-app`
- Windows machine: the shell is PowerShell/Git Bash, and `make` may be missing.

## Project conventions
- Reusable logic goes in the `churn/` package; notebooks import from it and tell the story. Notebook names follow the user's style: `notebooks/N_topic.ipynb` (ask before renaming existing ones).
- `RANDOM_STATE = 42`, paths, feature lists and `DROP_COLS` (excluded columns with reasons) live in `churn/config.py`; nothing is hard-coded elsewhere.
- **No leakage:** split first (stratified 80/20) · everything that learns sits inside the sklearn `Pipeline` (`collapse_no_internet` → `add_features` → `build_preprocessor` → model) · the threshold comes from out-of-fold train predictions · the test set is used once.
- **Metrics:** Precision, Recall, F1, ROC-AUC, PR-AUC and the confusion matrix. Accuracy is for reference only (about 26.5% churn).
- Cleaned data is **parquet** (`engine="pyarrow"`, `index=False`), with all columns kept; the Pipeline selects the features.
- `data/` is git-ignored; the final artifact in `models/` **is** committed so the Docker build is self-contained.
- The Streamlit app uses the same saved Pipeline as training, with no separate feature logic. `tenure` is limited to 0–72 months.
- `uv run ruff check churn/` must pass before a commit (ruff 0.16 defaults, line length 88; the user is new to ruff: it's a linter that catches undefined/unused names and keeps style consistent).

## Gotchas
- pandas here is v3 with Copy-on-Write: `df[col].method(..., inplace=True)` silently does nothing. Always assign back.
- Windows Application Control blocks executables run from uv's cache, so `uvx <tool>` fails. Install tools with `uv add --dev` and use `uv run`.
- Before editing a notebook from outside VS Code, make sure the user has it closed (or tell them to use *File → Revert File* afterwards). An open notebook saved from the editor overwrites cells added on disk.
- Docker Desktop and gcloud are not installed yet (needed for phases 9–10).
