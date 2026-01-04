# Pussel - Puzzle Solver Project Instructions

## Project Overview

Pussel is a computer vision-based puzzle solver application that helps users solve jigsaw puzzles. The application consists of three main components:

1. **Backend** (`backend/`) - Python FastAPI service for puzzle image processing and piece matching
2. **Frontend** (`frontend/`) - Next.js 15 web app with Bun for capturing puzzle pieces and displaying solutions
3. **Network** (`network/`) - PyTorch Lightning-based CNN model for predicting puzzle piece positions and rotations

The backend provides REST API endpoints for uploading puzzle images and processing individual puzzle pieces to determine their position, rotation, and placement confidence.

## Tech Stack

### Backend (Python/FastAPI)
- **Framework**: FastAPI
- **Python Version**: 3.12
- **Key Libraries**:
  - Pydantic 2.6.1 (data validation and settings)
  - Pillow 10.2.0 (image processing)
  - Uvicorn 0.27.1 (ASGI server)
  - python-multipart (file uploads)

### Frontend (Next.js/Bun)
- **Runtime**: Bun
- **Framework**: Next.js 15 with App Router and Turbopack
- **State Management**: Zustand
- **Styling**: Tailwind CSS 4 + shadcn/ui components
- **Linting**: OxLint (type-aware rules)
- **Testing**: Vitest
- **Theming**: next-themes (dark mode support)

### Network (ML Training)
- **Framework**: PyTorch Lightning
- **Model Architecture**: Dual-backbone CNN (one for piece, one for puzzle)
- **Python Version**: 3.12
- **Key Libraries**:
  - PyTorch Lightning
  - torchvision
  - Pillow

### Development Tools (All Components)
- **Python Testing**: pytest, pytest-cov
- **Python Type Checking**: pyright (standard mode enabled)
- **Python Linting**: flake8 with plugins (docstrings, import-order, bugbear)
- **Python Formatting**: black (line length: 120), isort
- **Frontend Testing**: Vitest
- **Frontend Linting**: OxLint (type-aware)
- **Frontend Formatting**: Prettier
- **Pre-commit Hooks**: Automated code quality checks for all components
- **CI/CD**: GitHub Actions (separate workflows for backend, frontend, network)
- **Coverage**: Codecov integration

## Project Structure

```
pussel/
├── .github/
│   ├── copilot-instructions.md     # This file - instructions for Copilot
│   └── workflows/
│       ├── backend-ci.yml          # Backend CI/CD pipeline
│       ├── frontend-ci.yml         # Frontend CI/CD pipeline
│       └── network-ci.yml          # Network (ML) CI/CD pipeline
├── backend/                        # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app and API endpoints
│   │   ├── config.py               # Pydantic settings configuration
│   │   ├── models/
│   │   │   └── puzzle_model.py     # Pydantic response models
│   │   └── services/
│   │       ├── image_processor.py  # Image processing logic (mock)
│   │       └── storage.py          # File storage utilities
│   ├── tests/
│   │   └── test_main.py            # API endpoint tests
│   ├── .flake8                     # Flake8 configuration
│   ├── pyrightconfig.json          # Pyright configuration
│   ├── pyproject.toml              # Black and isort configuration
│   ├── requirements.txt            # Python dependencies
│   └── setup.py                    # Package setup
├── frontend/                       # Next.js web app
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages
│   │   │   ├── page.tsx            # Home page
│   │   │   ├── puzzle/page.tsx     # Main puzzle workflow
│   │   │   ├── test-mode/          # Test mode for bundled puzzles
│   │   │   └── about/page.tsx      # About page
│   │   ├── components/             # React components
│   │   │   ├── camera/             # Camera capture components
│   │   │   ├── puzzle/             # Puzzle display components
│   │   │   └── ui/                 # shadcn/ui components
│   │   ├── hooks/                  # Custom React hooks
│   │   ├── stores/                 # Zustand state management
│   │   ├── lib/                    # Utilities and API client
│   │   └── types/                  # TypeScript type definitions
│   ├── public/                     # Static assets
│   ├── eslint.config.mjs           # ESLint configuration
│   ├── oxlint.json                 # OxLint configuration
│   ├── next.config.ts              # Next.js configuration
│   ├── package.json                # Node dependencies
│   └── tsconfig.json               # TypeScript configuration
├── network/                        # ML model training
│   ├── experiments/                # Training experiments
│   ├── model.py                    # Dual-backbone CNN architecture
│   ├── dataset.py                  # PyTorch Lightning DataModule
│   ├── train.py                    # Training script
│   ├── config.py                   # Training configuration
│   ├── puzzle_generator.py         # Generate training pieces
│   ├── resize_puzzles.py           # Standardize puzzle sizes
│   ├── visualize_piece.py          # Visualize piece placement
│   ├── pyrightconfig.json          # Pyright configuration
│   ├── pyproject.toml              # Black and isort configuration
│   └── requirements.txt            # Python dependencies
├── .pre-commit-config.yaml         # Pre-commit hooks (all components)
├── codecov.yml                     # Codecov configuration
├── Makefile                        # Common commands for all components
├── CLAUDE.md                       # Instructions for Claude Code
├── GEMINI.md                       # Instructions for Gemini
└── README.md                       # Main project README
```

## Coding Guidelines

### Python Style and Conventions (Backend & Network)

1. **Code Formatting**:
   - Use **black** for code formatting (line length: 120 characters)
   - Use **isort** for import sorting with black profile
   - Format before committing
   - Note: flake8 enforces a 120-character maximum line length for all code

2. **Import Order** (isort with black profile):
   - Future imports
   - Standard library imports
   - Third-party imports
   - First-party imports (`app` module for backend)
   - Local folder imports

3. **Type Hints**:
   - **REQUIRED**: All function definitions must include type hints for parameters and return values
   - Use modern Python 3.12 syntax: `dict[str, str]` instead of `Dict[str, str]`
   - Enable pyright type checking - all functions must be fully typed
   - No implicit Optional types allowed

4. **Docstrings**:
   - Use **Google-style docstrings** for all public functions, classes, and modules
   - Include Args, Returns, and Raises sections
   - Test files can omit some docstrings (per flake8 config)

5. **Linting**:
   - Max line length: 120 characters
   - Max complexity: 10 (cyclomatic complexity)
   - Use flake8 with plugins: docstrings, import-order, bugbear, comprehensions, pytest-style

6. **Error Handling**:
   - Backend: Use FastAPI's HTTPException for API errors
   - Include descriptive error messages
   - Use appropriate HTTP status codes (400, 404, 413, etc.)

7. **File Organization**:
   - Backend: Keep models in `app/models/`, business logic in `app/services/`, routes in `app/main.py`, config in `app/config.py`
   - Network: Keep model architecture in `model.py`, dataset handling in `dataset.py`, training in `train.py`

### TypeScript/React Conventions (Frontend)

1. **Code Formatting**:
   - Use **Prettier** for code formatting
   - Use **OxLint** for type-aware linting
   - Format before committing

2. **Component Structure**:
   - Use functional components with hooks
   - Keep components small and focused
   - Use Zustand for state management
   - Use shadcn/ui components for UI elements

3. **Type Safety**:
   - Enable TypeScript strict mode
   - Define types in `src/types/`
   - Use type inference where possible
   - Avoid `any` type

4. **Styling**:
   - Use Tailwind CSS utility classes
   - Follow shadcn/ui patterns for components
   - Support dark mode with next-themes

5. **File Organization**:
   - Pages in `src/app/` (App Router)
   - Components in `src/components/`
   - Hooks in `src/hooks/`
   - State management in `src/stores/`
   - Utilities in `src/lib/`
   - Types in `src/types/`

### Testing Guidelines

1. **Backend Testing** (pytest):
   - Test file naming: `test_*.py`
   - Test function naming: `test_*`
   - Use fixtures for setup and cleanup
   - Maintain high test coverage (aim for >90%)
   - Run: `pytest -v --cov=app --cov-report=term-missing`
   - Generate XML coverage: `pytest -v --cov=app --cov-report=xml`

2. **Frontend Testing** (Vitest):
   - Test file naming: `*.test.ts` or `*.test.tsx`
   - Use React Testing Library patterns
   - Test user interactions and component behavior
   - Run: `bun run test`

3. **Network Testing**:
   - Test utilities and data processing functions
   - Use pytest for Python testing
   - Test file naming: `test_*.py`

## Build and Development Commands

### Initial Setup

**Backend (Python/FastAPI)**:
```bash
# Create virtual environment at repo root (shared with network)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies (using uv)
cd backend
uv sync --all-extras
pre-commit install
```

**Frontend (Next.js/Bun)**:
```bash
cd frontend
bun install
```

**Network (ML Training)**:
```bash
# Create and activate virtual environment (from repo root)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

cd network
pip install -r requirements.txt
```

**IMPORTANT**: Always activate the virtual environment before working with Python code:
```bash
source venv/bin/activate  # From repo root, or:
source ../venv/bin/activate  # From backend/ or network/ directory
```

### Running Applications

**Backend (FastAPI)**:
```bash
# From backend directory
cd backend
uvicorn app.main:app --reload

# Or from repo root using Makefile
make start-backend
```
API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

**Frontend (Next.js)**:
```bash
# From frontend directory
cd frontend
bun run dev

# Or from repo root using Makefile
make start-frontend
```
Web app will be available at `http://localhost:3000`

**Stop Servers**:
```bash
make stop-backend          # Kill process on port 8000
make stop-frontend         # Kill process on port 3000
```

### Code Quality Checks

**Backend**:
```bash
# From backend directory or repo root
cd backend

# Format code (auto-fix)
black .
isort .

# Check formatting (no changes)
black . --check
isort . --check-only

# Lint
flake8 .

# Type check
pyright .

# Or use Makefile from repo root
make format-backend        # Auto-format
make check-backend         # Run all checks
```

**Frontend**:
```bash
# From frontend directory
cd frontend

# Format code
bun run format             # Auto-format with Prettier

# Check code
bun run check              # Run all checks (lint, typecheck, prettier)
bun run lint               # Lint with OxLint (type-aware)
bun run typecheck          # TypeScript type checking

# Or use Makefile from repo root
make format-frontend       # Auto-format
make check-frontend        # Run all checks
```

**Network**:
```bash
# Activate venv first!
source venv/bin/activate   # From repo root

cd network

# Format code
black .
isort .

# Check formatting
black . --check
isort . --check-only

# Lint
flake8 .

# Type check
pyright .

# Or use Makefile from repo root
make format-network        # Auto-format
make check-network         # Run all checks
```

**All Components**:
```bash
# From repo root
make format                # Format all components
make check                 # Check all components

# Or use pre-commit hooks
pre-commit run --all-files
```

### Testing

**Backend**:
```bash
# From backend directory
cd backend

# Run all tests with coverage
pytest -v --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_main.py -v

# Run with coverage XML (for CI)
pytest -v --cov=app --cov-report=xml

# Or use Makefile from repo root
make test-backend
```

**Frontend**:
```bash
# From frontend directory
cd frontend

# Run tests
bun run test

# Or use Makefile from repo root
make test-frontend
```

**Network**:
```bash
# Activate venv first!
source venv/bin/activate   # From repo root

cd network
pytest -v
```

### ML Model Training

**Local Training**:
```bash
# Activate venv first!
source venv/bin/activate   # From repo root

cd network

# Train with default config
python train.py

# Custom training parameters
python train.py --backbone efficientnet_b0 --batch_size 32 --max_epochs 50

# Monitor training with TensorBoard
tensorboard --logdir=logs

# Dataset utilities
python puzzle_generator.py datasets/example/puzzle_001.jpg
python resize_puzzles.py datasets/example --output-dir datasets/example/resized
python visualize_piece.py datasets/example/puzzle_001.jpg datasets/example/pieces/piece_001.png
```

**RunPod GPU Training** (for faster training):
```bash
cd network/experiments/exp20_realistic_pieces
./runpod/prepare_package.sh  # Creates network/runpod_package/

# Upload to RunPod (get IP/PORT from RunPod dashboard)
scp -P <PORT> -i ~/.ssh/runpod_key runpod_package/runpod_training.tar.gz root@<IP>:/workspace/

# Run training on RunPod
ssh -p <PORT> -i ~/.ssh/runpod_key root@<IP>
cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh

# Download results
scp -P <PORT> -i ~/.ssh/runpod_key "root@<IP>:/workspace/outputs/*" ./outputs/
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration with separate workflows for each component:

### Backend CI (`.github/workflows/backend-ci.yml`)
Runs on:
- Push to master/main branch (if backend or CI config changes)
- Pull requests to master/main branch (if backend or CI config changes)

Pipeline steps:
1. Install uv
2. Install dependencies (`uv sync --locked --all-extras`)
3. Check formatting with black (`black . --check`)
4. Check imports with isort (`isort . --check-only`)
5. Lint with flake8 (`flake8 .`)
6. Type check with pyright (`pyright .`)
7. Run tests with coverage (`pytest -v --cov=app --cov-report=xml`)
8. Upload coverage to Codecov

### Frontend CI (`.github/workflows/frontend-ci.yml`)
Runs on:
- Push to master/main branch (if frontend or CI config changes)
- Pull requests to master/main branch (if frontend or CI config changes)

Pipeline steps:
1. Set up Bun
2. Install dependencies (`bun install`)
3. Check formatting with Prettier
4. Lint with OxLint (type-aware)
5. Type check with TypeScript
6. Run tests with Vitest
7. Build for production

### Network CI (`.github/workflows/network-ci.yml`)
Runs on:
- Push to master/main branch (if network or CI config changes)
- Pull requests to master/main branch (if network or CI config changes)

Pipeline steps:
1. Set up Python 3.12
2. Install dependencies
3. Check formatting with black
4. Check imports with isort
5. Lint with flake8
6. Type check with pyright

**Important**: All checks must pass for CI to succeed. Always run these checks locally before committing.

## Code Architecture

### Backend Architecture
- **`app/main.py`** - FastAPI application with CORS, endpoints, and in-memory puzzle storage
- **`app/config.py`** - Pydantic settings for configuration (upload dir, size limits, CORS)
- **`app/models/puzzle_model.py`** - Pydantic models for API requests/responses
- **`app/services/image_processor.py`** - Mock image processing (to be replaced with ML model)
- **`app/services/storage.py`** - File storage utilities

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

**Dataset Format**: Pieces stored as: `puzzle_XXX_piece_YYY_xX1_yY1_xX2_yY2_rROTATION.png`
- X1, Y1, X2, Y2: Bounding box coordinates
- ROTATION: 0, 90, 180, or 270 degrees

**ML Model Integration Note**: Current backend uses mock image processor in `app/services/image_processor.py`. The trained model from `network/` should eventually replace this mock implementation.

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

Configured in `.pre-commit-config.yaml` at the repository root and applies to all components:

**General Checks**:
- trailing-whitespace removal
- end-of-file-fixer
- check-yaml validation
- check-added-large-files
- debug-statements check

**Python (Backend & Network)**:
- requirements-txt-fixer
- black formatting (line length: 120)
- isort import sorting (with black profile)
- flake8 linting
- pyright type checking

**Frontend**:
- Prettier formatting
- OxLint type-aware linting (via frontend CI)

Run manually: `pre-commit run --all-files`

**Note**: Pre-commit hooks handle virtual environment activation automatically for network checks.

## Key Development Practices

1. **Always type hint**: All Python functions must have complete type annotations; TypeScript should use strict mode
2. **Write tests**: Add tests for new features and bug fixes
3. **Run quality checks**: Before committing, ensure all linting, type checking, and tests pass
   - Use `make check` to check all components
   - Use `make format` to auto-format all components
4. **Keep changes minimal**: Make focused, atomic commits
5. **Document with docstrings**: Use Google-style docstrings for Python; JSDoc for TypeScript where needed
6. **Follow existing patterns**: Match the structure and style of existing code
7. **Virtual environment**: Always activate venv when working with Python code (backend, network)
8. **Don't commit**: Build artifacts, `__pycache__`, `venv`, `node_modules`, `uploads/`, `.env` files, `.next/`, `dist/`
9. **Package installation**: Backend uses uv - run `uv sync --all-extras` in the backend directory

## Resources

### Documentation
- Main README: `/README.md`
- Backend README: `/backend/README.md`
- Network README: `/network/README.md`
- Frontend README: `/frontend/README.md`
- Claude Instructions: `/CLAUDE.md`
- Gemini Instructions: `/GEMINI.md`

### Technology Documentation
**Backend**:
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Pydantic Documentation: https://docs.pydantic.dev/
- Python Type Hints: https://docs.python.org/3/library/typing.html
- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

**Frontend**:
- Next.js Documentation: https://nextjs.org/docs
- Bun Documentation: https://bun.sh/docs
- Zustand Documentation: https://docs.pmnd.rs/zustand/
- shadcn/ui Documentation: https://ui.shadcn.com/
- Tailwind CSS Documentation: https://tailwindcss.com/docs

**Network**:
- PyTorch Lightning Documentation: https://lightning.ai/docs/pytorch/
- PyTorch Documentation: https://pytorch.org/docs/

### Useful Commands Quick Reference
```bash
# From repo root - works for all components
make format                # Format all code
make check                 # Check all code
make start-backend         # Start backend server
make start-frontend        # Start frontend server
make test-backend          # Run backend tests
make test-frontend         # Run frontend tests

# Component-specific
make format-backend        # Format backend only
make format-frontend       # Format frontend only
make format-network        # Format network only
make check-backend         # Check backend only
make check-frontend        # Check frontend only
make check-network         # Check network only
```
