# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check check-backend check-network check-shared check-frontend \
        format format-backend format-network format-shared format-frontend \
        test-backend install-dev-backend install-dev-network \
        start-backend start-frontend stop-backend stop-frontend

# Run all checks (Python + Next.js)
check: check-backend check-network check-shared check-frontend

# Backend checks
check-backend:
	cd backend && black . --check --line-length=120 --exclude venv
	cd backend && isort . --check-only --profile=black --line-length=120 --skip venv
	cd backend && flake8 . --config=../.flake8
	cd backend && pyright .

# Network checks
check-network:
	cd network && black . --check --line-length=120
	cd network && isort . --check-only --profile=black --line-length=120
	cd network && flake8 . --config=../.flake8
	cd network && pyright .

# Shared library checks
check-shared:
	cd shared/puzzle_shapes && black . --check --line-length=120
	cd shared/puzzle_shapes && isort . --check-only --profile=black --line-length=120
	cd shared/puzzle_shapes && flake8 . --config=../../.flake8
	cd shared/puzzle_shapes && pyright .

# Frontend checks (Next.js with Bun)
check-frontend:
	cd frontend && bun run check

# Auto-format all code (Python + Next.js)
format: format-backend format-network format-frontend

# Auto-format backend
format-backend:
	cd backend && black . --line-length=120 --exclude venv
	cd backend && isort . --profile=black --line-length=120 --skip venv
	pre-commit run requirements-txt-fixer --all-files || true

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

# Run backend tests with coverage
test-backend:
	cd backend && pytest -v --cov=app --cov-report=term-missing

# Install dev dependencies
install-dev-backend:
	pip install -e shared/puzzle_shapes
	cd backend && pip install -r requirements.txt && pip install -e .

install-dev-network:
	cd network && pip install -r requirements.txt
	pip install black isort flake8 flake8-docstrings flake8-import-order flake8-bugbear flake8-comprehensions flake8-pytest-style pyright

# Start development servers
start-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

start-frontend:
	cd frontend && bun run dev

# Stop development servers (kills processes on their ports)
stop-backend:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "Backend not running on port 8000"

stop-frontend:
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "Frontend not running on port 3000"
