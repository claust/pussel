# Puzzle Solver Backend

A FastAPI-based backend service for the Puzzle Solver application that helps users solve jigsaw puzzles using computer vision.

## Features

- Upload complete puzzle images
- Process individual puzzle pieces
- **⚠️ MOCK IMPLEMENTATION**: Current puzzle piece detection uses random values, not a trained ML model
- See [MODEL_ARCHITECTURE_ANALYSIS.md](MODEL_ARCHITECTURE_ANALYSIS.md) for detailed analysis and improvement plan
- REST API endpoints
- File upload handling
- Comprehensive test suite
- Type checking with mypy
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
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality checks

### Running Code Quality Checks

```bash
# Format code
black .
isort .

# Run linting
flake8

# Run type checking
mypy app
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
- **⚠️ MOCK IMPLEMENTATION**: Returns random position/rotation with 0.0 confidence
- Currently NO actual image analysis or ML prediction occurs
- See [MODEL_ARCHITECTURE_ANALYSIS.md](MODEL_ARCHITECTURE_ANALYSIS.md) for implementation roadmap
- Accepts multipart form data with piece image
- Returns:
  - Predicted position (x, y coordinates) - Currently random values (0-100)
  - Confidence score - Set to 0.0 to indicate no real prediction
  - Rotation angle (0, 90, 180, or 270 degrees) - Currently random
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
├── mypy.ini             # Mypy configuration
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
- Performs type checking with mypy
- Executes test suite with coverage reporting
- Uploads coverage reports to Codecov

The workflow is triggered on:
- Push to master/main branch
- Pull requests to master/main branch
- Only when changes affect backend code or CI configuration

## Future Improvements

### Critical - Machine Learning Model (See MODEL_ARCHITECTURE_ANALYSIS.md)
1. **Collect training data**: Need 1,000+ puzzle images with labeled pieces
2. **Implement ML model**: Siamese network or CNN-based architecture
3. **Train and validate**: Achieve 85-95% accuracy before production
4. **Replace mock implementation**: Deploy actual inference pipeline

### Infrastructure and Features
5. Add authentication and authorization
6. Add rate limiting
7. Implement proper image validation and sanitization
8. Add image compression and optimization
9. Implement proper error handling and logging
10. Add database for storing puzzle and piece information
11. Add Docker support
12. Implement caching for processed images
