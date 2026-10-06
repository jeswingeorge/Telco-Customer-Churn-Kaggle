# Phase 9: Docker

**CRISP-DM:** Deployment · **Status:** ⬜ Not started · **Where:** `Dockerfile`, `.dockerignore`

## Goal
Package the app so it runs the same way on any machine, and in the cloud (phase 10).

## New concepts
- **Image:** a frozen snapshot of everything the app needs (Linux, Python 3.12, packages, code, model). It is built once from a `Dockerfile`.
- **Container:** a running copy of an image. "It works on my machine" stops being a problem because the machine travels with the app.
- **Dockerfile:** a recipe, read top to bottom like notebook cells. Each line adds a layer.
- **`.dockerignore`:** like `.gitignore`, but for what is *not* copied into the image.

## Steps
- [ ] **1. Install Docker Desktop** (Windows; WSL 2 backend). Check: `docker --version` and `docker run hello-world`.
- [ ] **2. Write `.dockerignore`:** `data/`, `notebooks/`, `.venv/`, `reports/`, `tests/`, `.claude/`, `.git/`. Why: a smaller, faster image that contains no data.
- [ ] **3. Write the `Dockerfile`**, line by line with comments:
  - `FROM python:3.12-slim`
  - Install uv; copy `pyproject.toml` + `uv.lock`; `uv sync --frozen --no-dev` (runtime deps only; `--frozen` = exactly the locked versions)
  - Copy `churn/`, `app/`, `models/`
  - Create and switch to a non-root user (security)
  - `CMD streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true`
- [ ] **4. Build and run locally** (from the repo root): `docker build -t churn-app .`, then `docker run -p 8080:8080 churn-app`. Open http://localhost:8080.
- [ ] **5. Same-prediction check:** the same test customer gives the same probability as the local Streamlit app.

## Done when
- The container runs locally and passes the same-prediction check.
