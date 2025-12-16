# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check format lint typecheck install-dev isort-check black-check format-check

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install black isort flake8 flake8-docstrings mypy

# Run all checks (same as CI)
check: format-check lint typecheck

# Check formatting with black
black-check:
	black . --check

# Check import sorting with isort
isort-check:
	isort . --check-only --profile=black --line-length=88

# Run all formatting checks (isort + black)
format-check: isort-check black-check

# Run flake8 linting
lint:
	flake8 . --extend-ignore=I100,I201

# Run mypy type checking
typecheck:
	mypy $$(find . -name "*.py" -not -path "./venv/*")

# Auto-format code
format:
	black .
	isort . --profile=black --line-length=88
