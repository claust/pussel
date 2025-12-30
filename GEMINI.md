# Gemini Context: Pussel (Puzzle Solver)

A computer vision-based application designed to help users solve jigsaw puzzles by predicting piece positions and rotations.

## Project Overview

The project is a multi-component system consisting of a mobile frontend, a FastAPI backend, and a machine learning pipeline for puzzle piece analysis.

### Core Components

- **Backend (`/backend`):** A FastAPI service (Python 3.12) that handles image uploads, coordinates processing, and serves the API.
- **Frontend (`/frontend`):** A Flutter mobile application for users to capture images of puzzles and pieces.
- **Network (`/network`):** Machine learning utilities and training scripts using PyTorch Lightning for puzzle piece detection and orientation prediction. Uses a dual-backbone CNN architecture (one for piece, one for puzzle).
- **Infrastructure (`/infrastructure`):** Azure cloud infrastructure definitions using Bicep.

## Building and Running

### Backend
1. **Setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   pre-commit install
   ```
2. **Run:**
   ```bash
   uvicorn app.main:app --reload
   ```
3. **Test:**
   ```bash
   pytest -v --cov=app --cov-report=term-missing
   ```

### Frontend
1. **Setup:**
   ```bash
   cd frontend
   flutter pub get
   ```
2. **Run:**
   ```bash
   flutter run
   ```
3. **Test:**
   ```bash
   flutter test
   ```

### Network (ML Utilities)
1. **Setup:**
   ```bash
   # From root
   python -m venv venv
   source venv/bin/activate
   cd network
   pip install -r requirements.txt
   ```
2. **Development Commands (via Makefile):**
   ```bash
   make format      # Auto-format with black and isort
   make check       # Run all checks (format, lint, typecheck)
   make lint        # Run flake8 linting
   make typecheck   # Run pyright type checking
   ```
3. **Training & Utilities:**
   ```bash
   python train.py            # Train with default config
   python puzzle_generator.py # Generate training pieces
   tensorboard --logdir=logs  # Monitor training
   ```

## Development Conventions

### Python (Backend & Network)
- **Formatting:** `black` (88 chars) and `isort` (with `--profile black`).
- **Linting:** `flake8` with `bugbear`, `docstrings`, and `import-order` plugins.
- **Type Checking:**
  - Backend uses `pyright` (standard mode).
  - Network uses `pyright`.
- **Pre-commit:** configured at the root level; run `pre-commit run --all-files` to check all components.

### Flutter (Frontend)
- **State Management:** `provider` package.
- **Networking:** `dio` and `http`.
- **Linting:** `dart analyze` (uses `very_good_analysis`).
- **Formatting:** `dart format`.

### CI/CD
- GitHub Actions workflows are located in `.github/workflows/` for backend, frontend, and network components.
- Code coverage is tracked via Codecov.

## Key Files & Directories
- `backend/app/main.py`: Entry point for the FastAPI application.
- `network/model.py`: CNN architecture (dual-backbone).
- `network/dataset.py`: PyTorch Lightning DataModule.
- `frontend/lib/main.dart`: Entry point for the Flutter app.
- `infrastructure/main.bicep`: Main Azure deployment file.
- `CLAUDE.md`: Detailed guidance for Claude Code.
- `implementation_plan.md`: Roadmap for the ML component.
