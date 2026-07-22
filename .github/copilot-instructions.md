# Pussel — Project Instructions

Pussel is a computer vision puzzle solver: photograph an assembled puzzle, then
photograph a loose piece to get its position, rotation, and confidence.

## Components

- `backend/` — FastAPI service (Python 3.12): piece matching, puzzle store,
  Google auth. Deployed to Azure App Service from `main`.
- `ios/` — native SwiftUI app (iOS 26), the shipped client.
- `frontend/` — Next.js 16 + Bun + Tailwind web app. Dev/test client, **not
  deployed**.
- `network/` — PyTorch Lightning experiments for learned matching. Research,
  not the shipped path.
- `shared/puzzle_shapes` — Python library used by backend and network.

Piece matching defaults to a classical SIFT → NCC hybrid
(`MATCHER=classical` in `backend/app/config.py`,
`backend/app/services/classical_matcher.py`); the CNN in
`backend/app/services/image_processor.py` is opt-in via `MATCHER=cnn`.

The backend's puzzle store is in-memory — a `puzzle_id` does not survive a
restart.

## Setup

`backend`, `network`, and `shared/puzzle_shapes` are one uv workspace with a
single root `uv.lock` and `.venv`. Install from the repo root:

```bash
uv sync --all-extras
pre-commit install
cd frontend && bun install
```

## Commands

From the repo root:

```bash
make start-backend   # http://localhost:8000 (docs at /docs)
make start-frontend  # http://localhost:3000
make ios-run         # build + launch on the Simulator

make check           # backend, network, shared, frontend, iOS
make format          # auto-format everything
make test-backend    # pytest with coverage
```

Component-scoped variants exist (`make check-backend`, `make format-frontend`,
…). Frontend tests run from `frontend/`: `bun run test` (Vitest) and
`bun run test:e2e` (Playwright).

## Conventions

**Python** — Black + isort (profile black), flake8 (docstrings, import-order,
bugbear, comprehensions, pytest-style), pyright standard mode. 120-char lines,
complete type annotations on every function, Google-style docstrings, modern
`dict[str, str]` syntax. FastAPI `HTTPException` with accurate status codes for
API errors. Models in `app/models/`, logic in `app/services/`, routes in
`app/main.py`, settings in `app/config.py`.

**Frontend** — OxLint (type-aware), Prettier, TypeScript strict mode. Functional
components, Zustand for state, shadcn/ui + Tailwind, dark mode via next-themes.
Pages in `src/app/`, components in `src/components/`, types in `src/types/`.

**iOS** — swift-format (bundled with Xcode) owns layout; SwiftLint enforces
`ios/.swiftlint.yml`. Both run in `make check-ios`.

**All** — add tests with new features, keep commits focused, follow the
surrounding code's patterns, and never commit build artifacts, `.venv`,
`node_modules`, `uploads/`, or `.env` files.

## CI

Per-component workflows in `.github/workflows/` (`backend-ci`, `frontend-ci`,
`ios-ci`, `network-ci`, plus a Dependabot lockfile job) run the same checks as
`make check`, on pushes and PRs touching that component. Backend CI also builds
the container and deploys to Azure on `main`. All checks must pass — run
`make check` locally first.

## More detail

`README.md` (overview), `CLAUDE.md` (agent guidance), and the per-component
`backend/README.md`, `ios/README.md`, `frontend/README.md`,
`network/README.md`.
