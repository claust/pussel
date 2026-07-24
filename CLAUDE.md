# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Overview

Pussel is a computer vision puzzle solver: photograph an assembled puzzle, then
photograph a loose piece to get its position, rotation, and confidence.

- `backend/` — FastAPI service (piece matching, puzzle store, Google auth).
  Deployed to the home server (see `backend/deploy/README.md`): merging to
  `main` triggers a pull-based redeploy at https://pussel.sabeltiger.dk.
- `ios/` — native SwiftUI app, the shipped client. See `ios/README.md`.
- `frontend/` — Next.js 16 + Bun web app. Dev/test client, **not deployed**.
- `network/` — PyTorch Lightning experiments. Research, not the shipped path.
- `shared/puzzle_shapes` — Python library used by backend and network.

Piece matching defaults to a classical SIFT → NCC hybrid (`MATCHER=classical`
in `backend/app/config.py`); the CNN is opt-in via `MATCHER=cnn`.

## Setup

`backend`, `network`, and `shared/puzzle_shapes` are **one uv workspace** —
one root `uv.lock`, one root `.venv`. Add or bump a dependency in a member's
`pyproject.toml`, then run `uv lock` at the root; CI fails on a stale lock.

```bash
uv sync --all-extras      # from repo root
pre-commit install
cd frontend && bun install
```

## Commands

All from the repo root:

```bash
make start-backend   # http://localhost:8000 (docs at /docs)
make start-frontend  # http://localhost:3000
make stop-backend    # and make stop-frontend
make ios-run         # build + launch on the Simulator (see ios/README.md)

make check           # backend, network, shared, frontend, iOS
make format          # auto-format everything
make test-backend    # pytest with coverage
```

Component-scoped variants exist for each: `make check-backend`,
`make format-frontend`, etc. Frontend tests are `bun run test` (Vitest) and
`bun run test:e2e` (Playwright) from `frontend/` — there is no
`make test-frontend`.

## Conventions

- **Python**: Black + isort (profile black), flake8, pyright standard mode.
  120-char lines, full type annotations, Google-style docstrings.
- **Frontend**: OxLint (type-aware), Prettier, TypeScript strict.
- **iOS**: swift-format (bundled with Xcode) for layout, SwiftLint for rules.
- Match the surrounding code's naming, comment density, and idiom.

## Before committing

Run `make check` and fix what it reports (`make format` handles the
auto-fixable parts), then commit — pre-commit hooks run the same checks.
CI runs per-component workflows in `.github/workflows/`.

## Notes

- The backend's puzzle store is **in-memory**: a `puzzle_id` dies when the
  backend restarts. The iOS app keeps the trimmed image locally and offers
  one-tap re-upload.
- The backend loads a model checkpoint from `network/experiments/` only when
  `MATCHER=cnn`; it is not committed, and predictions fall back to a neutral
  result when it is missing.
- Frontend env lives in `frontend/.env.local`, backend env in `backend/.env`
  (both gitignored). New worktrees need them copied over.
