# Pussel Backend

FastAPI service behind the Pussel puzzle solver: stores an uploaded puzzle,
matches loose pieces against it, and returns each piece's position, rotation,
and confidence. Deployed to the home server at https://pussel.sabeltiger.dk —
merging to `main` triggers a pull-based redeploy (see
[deploy/README.md](deploy/README.md)).

## Setup

The backend is part of the repo-wide uv workspace, so install from the **repo
root**, not from here:

```bash
uv sync --all-extras
pre-commit install
```

Copy `.env.example` to `.env` for local runs (Google auth, JWT secret,
storage — see `app/config.py` for the full set of settings).

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

| Method & path                                                 | Purpose                                        |
| ------------------------------------------------------------- | ---------------------------------------------- |
| `GET /health`                                                 | Health check                                   |
| `POST /api/v1/auth/google`                                    | Exchange a Google ID token for a backend JWT   |
| `GET /api/v1/auth/me`                                         | Current user                                   |
| `POST /api/v1/puzzle/upload`                                  | Upload the assembled puzzle, get a `puzzle_id` |
| `GET /api/v1/puzzles`                                         | List stored puzzles                            |
| `POST /api/v1/puzzle/detect-frame`                            | Detect the puzzle's frame for trimming         |
| `GET /api/v1/puzzle/barcode/{ean}`                            | Look up a puzzle by barcode                    |
| `POST /api/v1/piece/preview`                                  | Live piece preview (detection only)            |
| `POST /api/v1/puzzle/{puzzle_id}/piece`                       | Match a piece → position, rotation, confidence |
| `POST /api/v1/puzzle/{puzzle_id}/piece/geometry`              | Save a piece's geometry record                 |
| `GET /api/v1/puzzle/{puzzle_id}/piece/geometry`               | List geometry records                          |
| `DELETE /api/v1/puzzle/{puzzle_id}/piece/geometry/{piece_id}` | Delete a geometry record                       |
| `POST /api/v1/puzzle/{puzzle_id}/generate-piece`              | Generate a synthetic piece                     |
| `POST /api/v1/puzzle/{puzzle_id}/cut-all`                     | Cut the puzzle into all its pieces             |

Uploads are capped at 10MB; endpoints decode and validate the image when they
need to process it (e.g. `upload` when a `piece_count` is given). The puzzle
store is **in-memory**: a `puzzle_id` does not survive a backend restart.

## Tests and checks

```bash
uv run pytest -v --cov=app --cov-report=term-missing    # or: make test-backend
make check-backend       # black, isort, flake8, pyright — what CI runs
make format-backend      # auto-fix formatting
```

## Deployment

Runs on the home server as a Docker Compose stack, routed through the
Appwrite stack's Traefik and fronted by Cloudflare at
https://pussel.sabeltiger.dk and https://pussel.thomasen.dk (the URL in the
iOS app's Release config).

Merging to `main` deploys: `pussel-deploy.timer` on the server polls every
5 minutes and rebuilds/restarts on a new commit; a failed build leaves the
running container untouched. The image builds from the **repo root**
(`docker build -f backend/Dockerfile .`), since the backend depends on
`shared/puzzle_shapes`. First-time setup, secrets layout, and operations
commands live in [deploy/README.md](deploy/README.md).

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
├── deploy/              # home-server compose stack + systemd deploy units
└── Dockerfile           # container image (build from the repo root)
```
