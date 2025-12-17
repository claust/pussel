# Pussel - Puzzle Solver Project Instructions

## Project Overview

Pussel is a computer vision-based puzzle solver application that helps users solve jigsaw puzzles. The application consists of:
- **Backend**: A Python FastAPI backend for image processing and puzzle piece detection
- **Frontend**: A Flutter mobile app (planned/coming soon)

The backend provides REST API endpoints for uploading puzzle images and processing individual puzzle pieces to determine their position, rotation, and placement confidence.

## Tech Stack

### Backend (Python)
- **Framework**: FastAPI 0.109.2
- **Python Version**: 3.12
- **Key Libraries**:
  - Pydantic 2.6.1 (data validation and settings)
  - Pillow 10.2.0 (image processing)
  - Uvicorn 0.27.1 (ASGI server)
  - python-multipart (file uploads)

### Development Tools
- **Testing**: pytest, pytest-cov
- **Type Checking**: mypy (strict mode enabled)
- **Linting**: flake8 with plugins (flake8-docstrings, flake8-import-order, flake8-bugbear)
- **Formatting**: black (line length: 88), isort
- **Pre-commit Hooks**: Automated code quality checks
- **CI/CD**: GitHub Actions
- **Coverage**: Codecov integration

## Project Structure

```
pussel/
├── .github/
│   └── workflows/
│       └── backend-ci.yml          # CI/CD pipeline configuration
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app and API endpoints
│   │   ├── config.py               # Pydantic settings configuration
│   │   ├── models/
│   │   │   └── puzzle_model.py     # Pydantic response models
│   │   └── services/
│   │       └── image_processor.py  # Image processing logic
│   ├── tests/
│   │   └── test_main.py            # API endpoint tests
│   ├── .flake8                     # Flake8 configuration
│   ├── .pylintrc                   # Pylint configuration
│   ├── mypy.ini                    # Mypy configuration
│   ├── pyproject.toml              # Black and isort configuration
│   ├── requirements.txt            # Python dependencies
│   └── setup.py                    # Package setup
├── .pre-commit-config.yaml         # Pre-commit hooks
├── codecov.yml                     # Codecov configuration
└── README.md
```

## Coding Guidelines

### Python Style and Conventions

1. **Code Formatting**:
   - Use **black** for code formatting (line length: 88 characters)
   - Use **isort** for import sorting with black profile
   - Format before committing

2. **Import Order** (isort with black profile):
   - Future imports
   - Standard library imports
   - Third-party imports
   - First-party imports (`app` module)
   - Local folder imports

3. **Type Hints**:
   - **REQUIRED**: All function definitions must include type hints for parameters and return values
   - Use modern Python 3.12 syntax: `dict[str, str]` instead of `Dict[str, str]`
   - Enable strict mypy checking - all functions must be fully typed
   - No implicit Optional types allowed

4. **Docstrings**:
   - Use **Google-style docstrings** for all public functions, classes, and modules
   - Include Args, Returns, and Raises sections
   - Test files can omit some docstrings (per flake8 config)

5. **Linting**:
   - Max line length: 79 characters (flake8)
   - Max complexity: 10 (cyclomatic complexity)
   - Follow flake8-bugbear recommendations
   - Ignore E203, W503 (conflicts with Black)

6. **Error Handling**:
   - Use FastAPI's HTTPException for API errors
   - Include descriptive error messages
   - Use appropriate HTTP status codes (400, 404, 413, etc.)

7. **File Organization**:
   - Keep models in `app/models/`
   - Keep business logic in `app/services/`
   - Main API routes in `app/main.py`
   - Configuration in `app/config.py`

### Testing Guidelines

1. **Test Framework**: Use pytest with type hints
2. **Coverage**: Maintain high test coverage (aim for >90%)
3. **Test Structure**:
   - Use fixtures for setup and cleanup
   - Test file naming: `test_*.py`
   - Test function naming: `test_*`
4. **Testing Commands**:
   ```bash
   # Run tests with coverage
   pytest -v --cov=app --cov-report=term-missing
   
   # Generate XML coverage report (for CI)
   pytest -v --cov=app --cov-report=xml
   ```

## Build and Development Commands

### Initial Setup (from backend directory)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Running the Application
```bash
# From backend directory
uvicorn app.main:app --reload
```
API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Code Quality Checks
```bash
# Format code (auto-fix)
black .
isort .

# Check formatting (no changes)
black . --check
isort . --check-only

# Lint
flake8 .

# Type check
mypy app
```

### Testing
```bash
# Run all tests with coverage
pytest -v --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_main.py -v

# Run with coverage XML (for CI)
pytest -v --cov=app --cov-report=xml
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/backend-ci.yml`) runs on:
- Push to master/main branch (if backend or CI config changes)
- Pull requests to master/main branch (if backend or CI config changes)

Pipeline steps:
1. Set up Python 3.12
2. Install dependencies (`pip install -r requirements.txt` and `pip install -e .`)
3. Check formatting with black (`black . --check`)
4. Check imports with isort (`isort . --check-only`)
5. Lint with flake8 (`flake8 .`)
6. Type check with mypy (`mypy app`)
7. Run tests with coverage (`pytest -v --cov=app --cov-report=xml`)
8. Upload coverage to Codecov

**Important**: All checks must pass for CI to succeed. Always run these checks locally before committing.

## API Endpoints

### Health Check
- `GET /health` - Returns API health status

### Puzzle Upload
- `POST /api/v1/puzzle/upload`
- Accepts: multipart/form-data with image file
- Max file size: 10MB
- Returns: `PuzzleResponse` with puzzle_id

### Process Puzzle Piece
- `POST /api/v1/puzzle/{puzzle_id}/piece`
- Accepts: multipart/form-data with piece image
- Returns: `PieceResponse` with predicted position, confidence, and rotation

## Configuration

Settings are managed via Pydantic Settings (`app/config.py`):
- Can be overridden with `.env` file
- Key settings:
  - `API_V1_STR`: "/api/v1"
  - `PROJECT_NAME`: "Puzzle Solver"
  - `UPLOAD_DIR`: "uploads"
  - `MAX_UPLOAD_SIZE`: 10MB
  - `BACKEND_CORS_ORIGINS`: ["*"] (development)

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- trailing-whitespace removal
- end-of-file-fixer
- check-yaml validation
- check-added-large-files
- debug-statements check
- requirements-txt-fixer
- black formatting
- isort import sorting
- flake8 linting
- mypy type checking

Run manually: `pre-commit run --all-files`

## Key Development Practices

1. **Always type hint**: All functions must have complete type annotations
2. **Write tests first**: Add tests for new features before implementing
3. **Run quality checks**: Before committing, ensure all linting, type checking, and tests pass
4. **Keep changes minimal**: Make focused, atomic commits
5. **Document with docstrings**: Use Google-style docstrings for public APIs
6. **Follow existing patterns**: Match the structure and style of existing code
7. **Don't commit**: Build artifacts, `__pycache__`, `venv`, `uploads/`, `.env` files

## Resources

- Main README: `/README.md`
- Backend README: `/backend/README.md`
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Pydantic Documentation: https://docs.pydantic.dev/
- Python Type Hints: https://docs.python.org/3/library/typing.html
- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
