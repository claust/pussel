# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pussel is a computer vision-based puzzle solver application with three main components:
1. **Backend** (`backend/`) - FastAPI service for puzzle image processing and piece matching
2. **Frontend** (`frontend/`) - Flutter mobile app for capturing puzzle pieces and displaying solutions
3. **Network** (`network/`) - PyTorch Lightning-based CNN model for predicting puzzle piece positions and rotations

## Development Setup

### Backend (Python/FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

### Frontend (Flutter)
```bash
cd frontend
flutter pub get
```

### Network (ML Training)
```bash
# Create and activate virtual environment (from repo root)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

cd network
pip install -r requirements.txt
```

**IMPORTANT**: Always activate the virtual environment before working with network code:
```bash
source venv/bin/activate  # From repo root, or:
source ../venv/bin/activate  # From network/ directory
```

## Common Commands

### Backend Development
```bash
# Run development server
cd backend
uvicorn app.main:app --reload

# Run tests with coverage
cd backend
pytest -v --cov=app --cov-report=term-missing

# Run single test file
cd backend
pytest tests/test_main.py -v

# Code quality checks
cd backend
black .                    # Format code
isort .                    # Sort imports
flake8                     # Lint
mypy app                   # Type check

# Run all pre-commit hooks manually
pre-commit run --all-files
```

### Frontend Development
```bash
cd frontend
flutter run                # Run on connected device/emulator
flutter test               # Run tests
dart analyze              # Static analysis
dart format --output=write lib/  # Format code
```

### ML Model Training
**Note**: Always activate the venv first: `source ../venv/bin/activate` (from network/)

```bash
cd network
source ../venv/bin/activate  # Activate venv before running any commands

python train.py            # Train with default config

# Custom training parameters
python train.py --backbone efficientnet_b0 --batch_size 32 --max_epochs 50

# Monitor training with TensorBoard
tensorboard --logdir=logs

# Dataset utilities
python puzzle_generator.py datasets/example/puzzle_001.jpg
python resize_puzzles.py datasets/example --output-dir datasets/example/resized
python visualize_piece.py datasets/example/puzzle_001.jpg datasets/example/pieces/piece_001.png

# Code quality checks (same as CI pipeline)
make check                 # Run all checks (format, lint, typecheck)
make format                # Auto-format code with black and isort
make lint                  # Run flake8 linting
make typecheck             # Run mypy type checking
make install-dev           # Install dev dependencies (black, isort, flake8, mypy)
```

## Code Architecture

### Backend Architecture
- **`app/main.py`** - FastAPI application with CORS, endpoints, and in-memory puzzle storage
- **`app/config.py`** - Pydantic settings for configuration (upload dir, size limits, CORS)
- **`app/models/puzzle_model.py`** - Pydantic models for API requests/responses
- **`app/services/image_processor.py`** - Mock image processing (to be replaced with ML model)
- **`app/services/storage.py`** - File storage utilities

#### API Endpoints
- `POST /api/v1/puzzle/upload` - Upload complete puzzle image, returns puzzle_id
- `POST /api/v1/puzzle/{puzzle_id}/piece` - Process piece, returns position (x,y), rotation (0/90/180/270), confidence
- `GET /health` - Health check

### Frontend Architecture
- **`lib/main.dart`** - App entry point
- **`lib/ui/`** - Screens and widgets
- **`lib/services/`** - API client and business logic
- **`lib/models/`** - Data models
- **`lib/config/`** - App configuration

Uses Provider for state management, Camera/ImagePicker for image capture, Dio/HTTP for API calls.

### Network Architecture (ML Model)
- **`model.py`** - Dual-backbone CNN architecture:
  - Two identical CNN backbones (one for piece, one for puzzle)
  - Feature fusion layer
  - Position prediction head (bbox regression with Sigmoid activation)
  - Rotation prediction head (4-class classification)
- **`dataset.py`** - PuzzleDataModule for PyTorch Lightning with train/val splits
- **`train.py`** - Training script with CLI args for hyperparameters
- **`config.py`** - Default training configuration
- **Utilities**:
  - `puzzle_generator.py` - Generates training pieces from puzzle images
  - `resize_puzzles.py` - Standardizes puzzle image sizes
  - `visualize_piece.py` - Visualizes piece placement on puzzle

#### Dataset Format
Pieces stored as: `puzzle_XXX_piece_YYY_xX1_yY1_xX2_yY2_rROTATION.png`
- X1, Y1, X2, Y2: Bounding box coordinates
- ROTATION: 0, 90, 180, or 270 degrees

### Code Quality Standards
All code follows strict quality standards enforced via pre-commit hooks:
- **Line length**: 88 characters (Black default)
- **Python formatting**: Black + isort (with `--profile black`)
- **Python linting**: flake8 with plugins (docstrings, import-order, bugbear, comprehensions, pytest-style)
- **Type checking**: mypy with strict mode (`disallow_untyped_defs`, etc.)
- **Docstring style**: Google format
- **Flutter**: dart-analyze, dart-format, dart-fix

### CI/CD
GitHub Actions workflows test/deploy on push to master/main/dev:
- **Backend CI** (`.github/workflows/backend-ci.yml`):
  - Code quality: black, isort, flake8, mypy
  - Tests with coverage (uploads to Codecov)
  - Azure deployment (dev branch only)
- **Frontend CI** (`.github/workflows/frontend-ci.yml`): Flutter analysis and tests
- **Network CI** (`.github/workflows/network-ci.yml`): Python quality checks for ML code

## Important Notes

### Package Installation
The backend uses both `requirements.txt` AND `setup.py`. Always run:
```bash
pip install -r requirements.txt
pip install -e .
```

### Type Checking
mypy is configured strictly - all functions must have type annotations. See `backend/mypy.ini` for configuration details.

### Testing
- Backend tests are in `backend/tests/`
- Always run with coverage: `pytest -v --cov=app --cov-report=term-missing`
- CI requires `.env.test` file for environment variables

### ML Model Integration
Current backend uses mock image processor in `app/services/image_processor.py`. The trained model from `network/` should eventually replace this mock implementation.

### Pre-commit Hooks
Pre-commit is configured at the root level and applies to:
- Python files (backend, network, and root Python scripts)
- Dart files (frontend only)
- Bicep files (infrastructure only)

Run `pre-commit install` after cloning to enable automatic checks.

## Workflow Guidelines

### Before Committing Changes

**IMPORTANT**: Always run the appropriate checks before committing to ensure CI will pass.

#### For Network (ML) code changes:
```bash
cd network
source ../venv/bin/activate  # IMPORTANT: Activate venv first!
make check    # Runs black, isort, flake8, and mypy - same as CI
```

**IMPORTANT**: You must activate the venv before running `make check` manually. However, **git commit hooks automatically activate the venv**, so committing from VS Code or the terminal works without manual activation.

If checks fail, run `make format` to auto-fix formatting issues, then address any remaining lint or type errors.

#### For Backend code changes:
```bash
cd backend
black . --check && isort . --check-only && flake8 && mypy app
```

Or simply run: `pre-commit run --all-files` from the repo root.

#### For Frontend code changes:
```bash
cd frontend
dart analyze && dart format --output=none --set-exit-if-changed lib/
```

### Commit Workflow
1. Make your changes
2. Run the appropriate checks for the module you modified
3. Fix any issues found
4. Commit (pre-commit hooks will run automatically if installed)
5. Push and verify CI passes
