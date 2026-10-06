# Phase 10: GCP Cloud Run Deployment + Polish

**CRISP-DM:** Deployment · **Status:** ⬜ Not started · **Where:** gcloud CLI, `tests/`, `README.md`

## Goal
Put the Docker image on a public URL, then finish the repo for interviewers (tests, README).

## New concept: Cloud Run
Cloud Run runs your container on Google's servers and gives it a URL. With `min-instances 0` it shuts down when nobody uses it (free tier), so the first visit after a pause takes a few seconds (a "cold start").

## Steps: deploy
- [ ] **1. Prerequisites:** install the Google Cloud SDK (`gcloud`); create a GCP project with billing enabled; `gcloud auth login`; `gcloud config set project <project>`.
- [ ] **2. Enable services:** `gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com`
- [ ] **3. Create an image repository:** `gcloud artifacts repositories create churn-repo --repository-format=docker --location=<region>`
- [ ] **4. Build in the cloud:** `gcloud builds submit --tag <region>-docker.pkg.dev/<project>/churn-repo/churn-app:v1`
- [ ] **5. Deploy:** `gcloud run deploy churn-app --image <region>-docker.pkg.dev/<project>/churn-repo/churn-app:v1 --region <region> --allow-unauthenticated --memory 1Gi --min-instances 0 --max-instances 2`
- [ ] **6. Same-prediction check:** the Cloud Run URL gives the same prediction for the same test customer as local and Docker.

## Steps: polish
- [ ] **7. Tests (`uv run pytest`):**
  - the cleaning output has no nulls and a numeric `TotalCharges`
  - the Pipeline fits and predicts on a sample
  - `predict.py` returns probabilities in [0, 1] and respects the threshold
  - the leakage fit check (L10)
  - `psi` returns ≈ 0 for identical distributions
- [ ] **8. Reproducibility:** `uv run python -m churn.dataset && uv run python -m churn.modeling.train` rebuilds the artifact with the same metrics; notebooks run top to bottom.
- [ ] **9. README:** live URL + screenshot, the Results table, the cold-start note, the monitoring summary, and "what I would do next".
- [ ] `uv run ruff check .` passes.

## Done when
- The public URL works, the tests pass, and the README is complete. **Cycle 1 is finished.**
