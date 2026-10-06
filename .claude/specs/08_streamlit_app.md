# Phase 8: Streamlit App

**CRISP-DM:** Deployment · **Status:** ⬜ Not started · **Where:** `app/streamlit_app.py`

## Goal
A simple web app where a retention analyst can score one customer or a CSV of customers, using **the same saved Pipeline** as training.

## New concept: what Streamlit is
Streamlit turns a Python script into a web page. Each widget (`st.selectbox`, `st.number_input`, `st.button`) is one line. **The whole script reruns top to bottom every time a widget changes**, much like re-running all notebook cells. That is why expensive work (loading the model) is cached.

## Inputs → Outputs
- **In:** `models/churn_model.joblib`, `models/metadata.json`, `models/reference_profile.json`; `churn.modeling.predict`, `churn.monitoring`
- **Out:** `app/streamlit_app.py`, running locally with `uv run streamlit run app/streamlit_app.py` (opens http://localhost:8501)

## Steps
- [ ] **1. Hello world.** A 3-line app (`st.title`, `st.write`) to learn the run → edit → auto-reload loop.
- [ ] **2. Load the model once.** Use `@st.cache_resource` on a function that loads the artifact + metadata via `churn.modeling.predict`. Why: without the cache, the model would reload on every click.
- [ ] **3. Single-customer tab.**
  - A sidebar form with every **raw** input field the Pipeline needs (not the engineered ones: the Pipeline builds those).
  - `tenure` is limited with `min_value=0, max_value=72` (phase 4 decision).
  - Output: churn probability, a risk label based on the saved threshold, and a SHAP waterfall of the top reasons.
- [ ] **4. Batch tab.**
  - Upload a CSV → run `drift_report` (phase 7) → show warnings → score the valid rows. Rows with `tenure` outside 0–72 are flagged, not scored.
  - Show the scored table with a download button (`st.download_button`).
- [ ] **5. About tab.** A metrics table and a short model card from `metadata.json` (model, threshold and why, test metrics, training date).
- [ ] **6. Same-prediction check.** The same test customer gets the same probability in the notebook and in the app.

## Rules
- No separate feature logic in the app: everything goes through the saved Pipeline (`collapse_no_internet` → `add_features` → preprocessor → model).
- The threshold is read from `metadata.json`, never hard-coded.

## Done when
- All 3 tabs work locally, and the same-prediction check passes.
