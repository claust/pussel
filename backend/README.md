# Puzzle Solver Backend

A FastAPI-based backend service for the Puzzle Solver application that helps users solve jigsaw puzzles using computer vision.

## Features

- Upload complete puzzle images
- Process individual puzzle pieces
- Mock implementation of puzzle piece detection (to be replaced with ML model)
- REST API endpoints
- File upload handling
- Comprehensive test suite
- Type checking with pyright
- Code formatting with black and isort
- Linting with flake8
- Pre-commit hooks for code quality
- Continuous Integration with GitHub Actions
- Code coverage reporting with Codecov

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies and package in development mode:
```bash
pip install -r requirements.txt
pip install -e .
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development

### Code Quality Tools

The project uses several tools to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting with additional plugins:
  - flake8-docstrings
  - flake8-import-order
  - flake8-bugbear
- **pyright**: Static type checking
- **pre-commit**: Git hooks for code quality checks

### Running Code Quality Checks

```bash
# Format code
black .
isort .

# Run linting
flake8

# Run type checking
pyright .
```

## Running the Application

Start the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## API Endpoints

### Upload Complete Puzzle
```
POST /api/v1/puzzle/upload
```
- Accepts multipart form data with image file
- Returns puzzle ID and optional image URL
- Maximum file size: 10MB
- Supported formats: Image files only

### Process Puzzle Piece
```
POST /api/v1/puzzle/{puzzle_id}/piece
```
- Accepts multipart form data with piece image
- Returns:
  - Predicted position (x, y coordinates)
  - Confidence score (0.5-1.0)
  - Rotation angle (0, 90, 180, or 270 degrees)
- Requires existing puzzle ID
- Supported formats: Image files only

### Health Check
```
GET /health
```
- Returns API health status

## Running Tests

```bash
pytest -v --cov=app --cov-report=term-missing
```

For XML coverage report (used by CI):
```bash
pytest -v --cov=app --cov-report=xml
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application and endpoints
│   ├── config.py         # Configuration settings
│   ├── models/
│   │   └── puzzle_model.py   # Pydantic models
│   └── services/
│       └── image_processor.py # Image processing logic
├── tests/
│   └── test_main.py      # API endpoint tests
├── .flake8              # Flake8 configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── pyrightconfig.json   # Pyright configuration
├── requirements.txt     # Project dependencies
└── README.md
```

## Configuration

The application uses Pydantic settings for configuration:

- `API_V1_STR`: API version prefix ("/api/v1")
- `PROJECT_NAME`: Project name ("Puzzle Solver")
- `UPLOAD_DIR`: Directory for uploaded files ("uploads")
- `MAX_UPLOAD_SIZE`: Maximum file upload size (10MB)
- `BACKEND_CORS_ORIGINS`: CORS settings (currently "*" for development)

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration, which:
- Runs on Ubuntu latest
- Tests with Python 3.12
- Installs dependencies and package in development mode
- Checks code formatting with black
- Verifies import order with isort
- Runs linting with flake8
- Performs type checking with pyright
- Executes test suite with coverage reporting
- Uploads coverage reports to Codecov

The workflow is triggered on:
- Push to master/main branch
- Pull requests to master/main branch
- Only when changes affect backend code or CI configuration

## Future Improvements

1. Implement actual computer vision model for puzzle piece detection
2. Add authentication and authorization
3. Add rate limiting
4. Implement proper image validation and sanitization
5. Add image compression and optimization
6. Implement proper error handling and logging
7. Add database for storing puzzle and piece information
8. Add CI/CD pipeline
9. Add Docker support
10. Implement caching for processed images
