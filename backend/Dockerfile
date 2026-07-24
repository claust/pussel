# Build from the REPO ROOT, not from backend/ — the backend is a uv-workspace
# member that depends on shared/puzzle_shapes:
#
#   docker build -f backend/Dockerfile .
#
FROM python:3.12-slim

# tesseract powers the barcode lookup's piece-count OCR
# (app/services/piece_count_estimator.py); without it the estimator
# gracefully returns no estimate.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer: lockfile plus every workspace member's manifest (uv needs
# them all to resolve the workspace), but no source code — so code-only changes
# reuse this cached layer instead of re-downloading the torch-sized deps.
COPY pyproject.toml uv.lock ./
COPY backend/pyproject.toml backend/
COPY network/pyproject.toml network/
COPY shared/puzzle_shapes/pyproject.toml shared/puzzle_shapes/
RUN uv sync --frozen --package pussel-backend --no-dev --no-install-workspace

COPY backend/ backend/
COPY shared/puzzle_shapes/ shared/puzzle_shapes/
RUN uv sync --frozen --package pussel-backend --no-dev

WORKDIR /app/backend

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
