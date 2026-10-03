# Recipe for the churn app image. Build and run from the repo root:
#   docker build -t churn-app .
#   docker run -p 8080:8080 churn-app      -> open http://localhost:8080
#
# Each instruction below adds a "layer". Docker caches layers, so if nothing above a line
# changed, that step is reused instead of re-run (that's why the order matters).

# 1. Start from a small Linux image that already has Python 3.12 (same version as .python-version).
FROM python:3.12-slim

# 2. Copy the uv binary from uv's official image (same version as on the laptop),
#    so we install packages exactly as `uv sync` does locally.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv

# 3. Settings for uv:
#    - use the Python from this image, never download another one
#    - precompile .py -> .pyc so the app starts faster
#    - copy files instead of hard-linking (avoids a warning inside Docker)
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 4. All the following commands run inside /app (created if missing).
WORKDIR /app

# 5. Install the dependencies FIRST, using only pyproject.toml + uv.lock.
#    --frozen: use the exact versions in uv.lock (same as the laptop, so the saved model loads)
#    --no-dev: skip jupyter, optuna, pytest, ruff (not needed to serve the app)
#    --no-install-project: our own `churn` code isn't copied yet
#    Why separately: this is the slow step. As long as the two files don't change, Docker
#    reuses this layer, and editing app code only re-runs the quick steps below.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 6. Now copy our code and the trained model, then install the `churn` package itself.
COPY churn/ churn/
COPY app/ app/
COPY models/ models/
RUN uv sync --frozen --no-dev

# 7. Security: don't run the app as root. Create a normal user and switch to it.
RUN useradd --create-home appuser
USER appuser

# 8. Put the project's virtual environment first on PATH, so `streamlit` means .venv's streamlit
#    (like `uv run` does locally).
ENV PATH="/app/.venv/bin:$PATH"

# 9. Documentation only: the app listens on 8080. Cloud Run sets $PORT (8080 by default).
EXPOSE 8080

# 10. The command run when the container starts.
#     --server.address=0.0.0.0 : accept connections from outside the container
#                                (the default, localhost, would only accept the container itself)
#     --server.headless=true   : don't try to open a browser (there is none in the container)
#     ${PORT:-8080}            : use $PORT if set, else 8080. `sh -c` is needed so $PORT is expanded.
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
