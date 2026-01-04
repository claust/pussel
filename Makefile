# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check check-backend check-network check-shared check-frontend \
        format format-backend format-network format-shared format-frontend \
        test-backend install-dev-backend install-dev-network \
        start-backend start-frontend stop-backend stop-frontend

# Run all checks (Python + Next.js)
check: check-backend check-network check-shared check-frontend

# Backend checks (uses uv to run tools from the backend venv)
check-backend:
	cd backend && uv run black . --check
	cd backend && uv run isort . --check-only
	cd backend && uv run flake8 . --config=../.flake8
	cd backend && uv run pyright .

# Network checks
check-network:
	cd network && black . --check --line-length=120
	cd network && isort . --check-only --profile=black --line-length=120
	cd network && flake8 . --config=../.flake8
	cd network && pyright .

# Shared library checks (uses backend's uv venv since puzzle-shapes is installed there)
check-shared:
	cd backend && uv run black ../shared/puzzle_shapes --check --line-length=120
	cd backend && uv run isort ../shared/puzzle_shapes --check-only --profile=black --line-length=120
	cd backend && uv run flake8 ../shared/puzzle_shapes --config=../.flake8
	cd backend && uv run pyright ../shared/puzzle_shapes

# Frontend checks (Next.js with Bun)
check-frontend:
	cd frontend && bun run check

# Auto-format all code (Python + Next.js)
format: format-backend format-network format-frontend

# Auto-format backend (uses uv to run tools from the backend venv)
format-backend:
	cd backend && uv run black .
	cd backend && uv run isort .

# Auto-format network
format-network:
	cd network && black . --line-length=120
	cd network && isort . --profile=black --line-length=120

# Auto-format shared library
format-shared:
	cd shared/puzzle_shapes && black . --line-length=120
	cd shared/puzzle_shapes && isort . --profile=black --line-length=120

# Auto-format frontend
format-frontend:
	cd frontend && bun run format

# Run backend tests with coverage (uses uv)
test-backend:
	cd backend && uv run pytest -v --cov=app --cov-report=term-missing

# Install dev dependencies (uses uv)
install-dev-backend:
	cd backend && uv sync --all-extras

install-dev-network:
	cd network && pip install -r requirements.txt
	pip install black isort flake8 flake8-docstrings flake8-import-order flake8-bugbear flake8-comprehensions flake8-pytest-style pyright

# Start development servers (uses uv)
start-backend:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

start-frontend:
	cd frontend && bun run dev

# Stop development servers (kills processes on their ports)
stop-backend:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "Backend not running on port 8000"

stop-frontend:
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "Frontend not running on port 3000"
