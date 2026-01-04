# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pussel is a computer vision-based puzzle solver application with three main components:
1. **Backend** (`backend/`) - FastAPI service for puzzle image processing and piece matching
2. **Frontend** (`frontend/`) - Next.js 15 web app with Bun for capturing puzzle pieces and displaying solutions
3. **Network** (`network/`) - PyTorch Lightning-based CNN model for predicting puzzle piece positions and rotations

## Development Setup

### Backend (Python/FastAPI)
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install backend dependencies (creates .venv automatically)
cd backend
uv sync --all-extras

# Install pre-commit hooks (from repo root)
pre-commit install
```

### Frontend (Next.js/Bun)
```bash
cd frontend
bun install
```

### Network (ML Training)
```bash
# Install network dependencies (uv manages its own .venv automatically)
cd network
uv sync --all-extras
```

## Common Commands

### Backend Development
```bash
# Run development server
cd backend
uv run uvicorn app.main:app --reload

# Run tests with coverage
cd backend
uv run pytest -v --cov=app --cov-report=term-missing

# Run single test file
cd backend
uv run pytest tests/test_main.py -v

# Code quality checks (from repo root)
make check-backend         # Run all checks (format, lint, typecheck)
make format-backend        # Auto-format code with black and isort
make test-backend          # Run tests with coverage

# Run all pre-commit hooks manually
pre-commit run --all-files
```

### Frontend Development
```bash
cd frontend
bun run dev                # Run development server with Turbopack
bun run build              # Build for production
bun run test               # Run tests with Vitest
bun run lint               # Lint with OxLint (type-aware)
bun run typecheck          # TypeScript type checking
bun run check              # Run all checks (lint, typecheck, prettier)
bun run format             # Auto-format with Prettier
```

### ML Model Training (Local)
```bash
cd network
uv run python train.py            # Train with default config

# Custom training parameters
uv run python train.py --backbone efficientnet_b0 --batch_size 32 --max_epochs 50

# Monitor training with TensorBoard
uv run tensorboard --logdir=logs

# Dataset utilities
uv run python puzzle_generator.py datasets/example/puzzle_001.jpg
uv run python resize_puzzles.py datasets/example --output-dir datasets/example/resized
uv run python visualize_piece.py datasets/example/puzzle_001.jpg datasets/example/pieces/piece_001.png

# Code quality checks (from repo root, same as CI pipeline)
make check-network         # Run all checks (format, lint, typecheck)
make format-network        # Auto-format code with black and isort
```

### ML Model Training (RunPod GPU)
For faster training, use RunPod with NVIDIA GPUs (RTX 4090 is ~10x faster than M4 Mac).

**Prepare and Upload**:
```bash
cd network/experiments/exp20_realistic_pieces
./runpod/prepare_package.sh  # Creates network/runpod_package/

# Upload to RunPod (get IP/PORT from RunPod dashboard)
scp -P <PORT> -i ~/.ssh/runpod_key runpod_package/runpod_training.tar.gz root@<IP>:/workspace/
```

**Run Training on RunPod**:
```bash
ssh -p <PORT> -i ~/.ssh/runpod_key root@<IP>
cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh
```

**Download Results**:
```bash
scp -P <PORT> -i ~/.ssh/runpod_key "root@<IP>:/workspace/outputs/*" ./outputs/
```

## Running Development Servers

### Quick Start (from repo root)
```bash
make start-backend         # Start FastAPI server on http://localhost:8000
make start-frontend        # Start Next.js dev server on http://localhost:3000
```

### Stopping Servers
```bash
make stop-backend          # Kill process on port 8000
make stop-frontend         # Kill process on port 3000
```

### Running in Background
When using `run_in_background: true`, ensure the working directory is repo root:

```bash
make start-backend         # Start FastAPI server on http://localhost:8000
make start-frontend        # Start Next.js dev server on http://localhost:3000
```

### Server URLs
- **Backend API**: http://localhost:8000
- **Backend Docs**: http://localhost:8000/docs (Swagger UI)
- **Frontend**: http://localhost:3000

### Debugging Tips
1. **Check if servers are running**: `lsof -i:8000` (backend) or `lsof -i:3000` (frontend)
2. **View backend logs**: The uvicorn `--reload` flag enables auto-reload on file changes
3. **Frontend hot reload**: Next.js with Turbopack automatically reloads on changes
4. **API testing**: Use the Swagger UI at `/docs` or curl commands
5. **Check health**: `curl http://localhost:8000/health`

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
- **`src/app/`** - Next.js App Router pages
  - `page.tsx` - Home page with navigation
  - `puzzle/page.tsx` - Main puzzle workflow (capture puzzle, add pieces)
  - `test-mode/` - Test mode for bundled puzzle images
  - `about/page.tsx` - About page
- **`src/components/`** - React components
  - `camera/` - Camera capture and file upload components
  - `puzzle/` - Puzzle display, piece cards, grid overlay
  - `ui/` - shadcn/ui components (button, card, dialog, etc.)
- **`src/hooks/`** - Custom React hooks (useCamera)
- **`src/stores/`** - Zustand state management
- **`src/lib/`** - Utilities (API client, image utils, test puzzles)
- **`src/types/`** - TypeScript type definitions

Tech stack:
- **Runtime**: Bun
- **Framework**: Next.js 15 with App Router and Turbopack
- **State**: Zustand
- **Styling**: Tailwind CSS 4 + shadcn/ui
- **Linting**: OxLint with type-aware rules
- **Testing**: Vitest
- **Theming**: next-themes (dark mode support)

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
- **Line length**: 120 characters
- **Python formatting**: Black + isort (with `--profile black`, `--line-length=120`)
- **Python linting**: flake8 with plugins (docstrings, import-order, bugbear, comprehensions, pytest-style)
- **Type checking**: pyright with standard mode
- **Docstring style**: Google format
- **Frontend**: OxLint (type-aware), Prettier, TypeScript strict mode

### CI/CD
GitHub Actions workflows test/deploy on push to master/main/dev:
- **Backend CI** (`.github/workflows/backend-ci.yml`):
  - Code quality: black, isort, flake8, pyright
  - Tests with coverage (uploads to Codecov)
  - Azure deployment (dev branch only)
- **Frontend CI** (`.github/workflows/frontend-ci.yml`):
  - OxLint type-aware linting
  - TypeScript type checking
  - Prettier formatting check
  - Vitest tests
  - Next.js production build
- **Network CI** (`.github/workflows/network-ci.yml`): Python quality checks for ML code

## Important Notes

### Package Installation
The backend uses uv with `pyproject.toml` for dependency management. To install:
```bash
cd backend
uv sync --all-extras
```
This creates a `.venv` directory and installs all dependencies including dev tools.

### Type Checking
pyright is configured with standard mode - all functions must have type annotations. See `backend/pyrightconfig.json` for configuration details.

### Testing
- Backend tests are in `backend/tests/`
- Frontend tests are in `frontend/src/**/*.test.ts`
- Always run with coverage: `pytest -v --cov=app --cov-report=term-missing`
- CI requires `.env.test` file for environment variables

### ML Model Integration
Current backend uses mock image processor in `app/services/image_processor.py`. The trained model from `network/` should eventually replace this mock implementation.

### Pre-commit Hooks
Pre-commit is configured at the root level and applies to:
- Python files (backend, network, and root Python scripts)
- Bicep files (infrastructure only)

Run `pre-commit install` after cloning to enable automatic checks.

## Workflow Guidelines

### Before Committing Changes

**IMPORTANT**: Always run the appropriate checks before committing to ensure CI will pass.

All checks can be run from the repo root using the root Makefile:

```bash
# Auto-format all code
make format                # Format both backend, network, and frontend
make format-backend        # Format backend only
make format-network        # Format network only
make format-frontend       # Format frontend only

# Check all code
make check

# Check individual projects
make check-backend         # Backend: black, isort, flake8, pyright
make check-network         # Network: black, isort, flake8, pyright
make check-frontend        # Frontend: oxlint

```

If checks fail, run the appropriate format command to auto-fix formatting issues, then address any remaining lint or type errors.

Or simply run: `pre-commit run --all-files` from the repo root.

### Commit Workflow
1. Make your changes
2. Run `make check`
3. Fix any issues found (use format commands for auto-fixable issues)
4. Commit (pre-commit hooks will run automatically if installed)
5. Push and verify CI passes
