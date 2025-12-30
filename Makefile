# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check check-backend check-network check-shared check-frontend \
        format format-backend format-network format-shared format-frontend \
        test-backend install-dev-backend install-dev-network

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
	cd backend && pip install -r requirements.txt && pip install -e .

install-dev-network:
	cd network && pip install -r requirements.txt
	pip install black isort flake8 flake8-docstrings flake8-import-order flake8-bugbear flake8-comprehensions flake8-pytest-style pyright
