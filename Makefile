# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check check-backend check-network check-frontend \
        format format-backend format-network format-frontend \
        test-backend install-dev-backend install-dev-network

# Run all checks (Python + Dart)
check: check-backend check-network check-frontend

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

# Frontend checks
check-frontend:
	cd frontend && dart analyze
	cd frontend && dart format --output=none --set-exit-if-changed lib/

# Auto-format all code (Python + Dart)
format: format-backend format-network format-frontend

# Auto-format backend
format-backend:
	cd backend && black . --line-length=120 --exclude venv
	cd backend && isort . --profile=black --line-length=120 --skip venv

# Auto-format network
format-network:
	cd network && black . --line-length=120
	cd network && isort . --profile=black --line-length=120

# Auto-format frontend
format-frontend:
	cd frontend && dart format lib/

# Run backend tests with coverage
test-backend:
	cd backend && pytest -v --cov=app --cov-report=term-missing

# Install dev dependencies
install-dev-backend:
	cd backend && pip install -r requirements.txt && pip install -e .

install-dev-network:
	cd network && pip install -r requirements.txt
	pip install black isort flake8 flake8-docstrings flake8-import-order flake8-bugbear flake8-comprehensions flake8-pytest-style pyright
