# Pussel Backend

FastAPI service behind the Pussel puzzle solver: stores an uploaded puzzle,
matches loose pieces against it, and returns each piece's position, rotation,
and confidence. Deployed to Azure App Service from `main`.

## Setup

The backend is part of the repo-wide uv workspace, so install from the **repo
root**, not from here:

```bash
uv sync --all-extras
pre-commit install
```

Copy `.env` into place for local runs (Google auth, JWT secret, storage — see
`app/config.py` for the full set of settings).

## Running

```bash
uv run uvicorn app.main:app --reload    # or: make start-backend from the root
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs` (`/redoc` for the alternative)

`api.http` has ready-made requests for the main endpoints.

## Piece matching

Two matchers, selected by the `MATCHER` setting:

- **`classical`** (default) — SIFT feature matching with an NCC fallback, in
  `app/services/classical_matcher.py`. This is what production serves.
- **`cnn`** — the dual-backbone model from `network/`, in
  `app/services/image_processor.py`. Loads a checkpoint from
  `network/experiments/`; the checkpoint is not committed, so a missing file
  degrades to a neutral prediction rather than failing.

Either way the result is snapped to the puzzle's grid when one is known.

## API

Auth is a Google ID token exchanged for a backend JWT; most endpoints require
the resulting `Bearer` token.

| Endpoint                                         | Purpose                                        |
| ------------------------------------------------ | ---------------------------------------------- |
| `GET /health`                                    | Health check                                   |
| `POST /api/v1/auth/google`                       | Exchange a Google ID token for a backend JWT   |
| `GET /api/v1/auth/me`                            | Current user                                   |
| `POST /api/v1/puzzle/upload`                     | Upload the assembled puzzle, get a `puzzle_id` |
| `GET /api/v1/puzzles`                            | List stored puzzles                            |
| `POST /api/v1/puzzle/detect-frame`               | Detect the puzzle's frame for trimming         |
| `GET /api/v1/puzzle/barcode/{ean}`               | Look up a puzzle by barcode                    |
| `POST /api/v1/piece/preview`                     | Live piece preview (detection only)            |
| `POST /api/v1/puzzle/{puzzle_id}/piece`          | Match a piece → position, rotation, confidence |
| `.../piece/geometry` (POST/GET/DELETE)           | Per-piece geometry records                     |
| `POST /api/v1/puzzle/{puzzle_id}/generate-piece` | Generate a synthetic piece                     |
| `POST /api/v1/puzzle/{puzzle_id}/cut-all`        | Cut the puzzle into all its pieces             |

Uploads are capped at 10MB and must be images. The puzzle store is
**in-memory**: a `puzzle_id` does not survive a backend restart.

## Tests and checks

```bash
uv run pytest -v --cov=app --cov-report=term-missing    # or: make test-backend
make check-backend       # black, isort, flake8, pyright — what CI runs
make format-backend      # auto-fix formatting
```

## Layout

```
backend/
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   ├── config.py        # Pydantic settings
│   ├── auth/            # Google ID-token verification, JWT issuing
│   ├── models/          # Pydantic request/response models
│   └── services/        # matching, detection, geometry, storage, lookups
├── tests/
├── scripts/             # dev helpers (e.g. generate_test_token.py)
├── api.http             # sample requests
└── Dockerfile           # image built and deployed by CI
```
