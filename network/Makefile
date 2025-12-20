# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check format lint typecheck install-dev isort-check black-check format-check

# Install development dependencies (same flake8 plugins as pre-commit)
install-dev:
	pip install -r requirements.txt
	pip install black isort flake8 flake8-docstrings flake8-import-order flake8-bugbear flake8-comprehensions flake8-pytest-style pyright

# Run all checks (same as CI)
check: format-check lint typecheck

# Check formatting with black
black-check:
	black . --check --line-length=120

# Check import sorting with isort
isort-check:
	isort . --check-only --profile=black --line-length=120

# Run all formatting checks (isort + black)
format-check: isort-check black-check

lint:
	flake8 . --config=../.flake8

# Run pyright type checking (same as VS Code Pylance)
typecheck:
	pyright .

# Auto-format code
format:
	black . --line-length=120
	isort . --profile=black --line-length=120
