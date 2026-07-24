# Pussel - Puzzle Solver

A computer vision app that helps you solve jigsaw puzzles: photograph the
assembled puzzle, then photograph a loose piece and get back where it belongs
(position, rotation, confidence).

## Project Structure

```
pussel/
├── backend/          # Python FastAPI service — piece matching, puzzle storage, auth
├── ios/              # Native SwiftUI iOS app (the primary client)
├── frontend/         # Next.js 16 web app (development/test client, run locally)
├── network/          # PyTorch Lightning experiments for learned piece matching
├── shared/           # puzzle_shapes — Python library shared by backend and network
```

## Components

**Backend** — FastAPI service exposing puzzle upload, frame detection, piece
matching, piece geometry, and Google-based auth. Piece matching defaults to a
classical SIFT → NCC hybrid (`MATCHER=classical`); the CNN from `network/` is
selectable with `MATCHER=cnn`. Deployed to the home server at
https://pussel.sabeltiger.dk — merging to `main` triggers a pull-based
redeploy (see `backend/deploy/README.md`).

**iOS app** — SwiftUI, iOS 26, Google Sign-In, on-device persistence of every
puzzle and its pieces. This is the real mobile client; see
[ios/README.md](ios/README.md) for setup (copy
`ios/Config/Secrets.example.xcconfig` to `Secrets.xcconfig`) and the
`make ios-*` workflows.

**Web frontend** — Next.js 16 + Bun + Tailwind, used for capture/test flows
during development. Not deployed anywhere; it runs locally against a local or
the deployed backend via `NEXT_PUBLIC_API_URL`.

**Network** — training code, datasets tooling, and experiments for learned
position/rotation prediction. Current production matching is classical, so
these are research artifacts rather than a shipped dependency.

## Getting Started

The Python projects (`backend`, `network`, `shared/puzzle_shapes`) form a
single uv workspace with one root lockfile:

```bash
uv sync --all-extras     # from the repo root
pre-commit install
```

Then, from the repo root:

```bash
make start-backend       # http://localhost:8000 (docs at /docs)
make start-frontend      # http://localhost:3000
make ios-run             # build + launch the iOS app on the Simulator
```

Per-component details:

- [Backend README](backend/README.md)
- [iOS README](ios/README.md)
- [Frontend README](frontend/README.md)
- [Network README](network/README.md)

## Development

```bash
make check               # backend, network, shared, frontend, iOS
make format              # auto-format everything
make test-backend        # pytest with coverage
```

Quality is enforced by pre-commit hooks and GitHub Actions: Black/isort/flake8/
pyright for Python, OxLint/Prettier/TypeScript/Vitest/Playwright for the
frontend, swift-format + SwiftLint for iOS, and Codecov for coverage.

## Deployment

The backend is the only deployed component. It runs on the home server as a
Docker Compose stack behind Cloudflare, reachable at
https://pussel.sabeltiger.dk and https://pussel.thomasen.dk (the URL the iOS
app ships with).

Merging to `main` is the deploy: a systemd timer on the server polls the
repository every 5 minutes and, on a new commit, rebuilds the image natively
and restarts the stack — no registry and no deploy secrets in CI, which only
verifies that the image still builds. Setup and operations are documented in
[backend/deploy/README.md](backend/deploy/README.md).

## License

MIT.
