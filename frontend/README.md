# Pussel Frontend

The web client for Pussel: capture a puzzle, add pieces, and see where each one
belongs. Next.js 16 (App Router, Turbopack) on Bun, with NextAuth for Google
sign-in against the FastAPI backend.

This app is a development and test client — it is **not deployed anywhere**. The
shipped mobile client is the native SwiftUI app in [`../ios`](../ios/README.md).

## Setup

```bash
bun install
cp .env.example .env.local     # then fill in the values
```

`.env.local` needs Google OAuth credentials (`GOOGLE_CLIENT_ID`,
`GOOGLE_CLIENT_SECRET`), an `AUTH_SECRET` (`openssl rand -base64 32`),
`AUTH_URL`, and `NEXT_PUBLIC_API_URL` pointing at the backend — `http://localhost:8000`
locally, or the Azure backend. See `.env.example` for the full list.

Start the backend first (`make start-backend` from the repo root), then:

```bash
bun run dev                    # http://localhost:3000
```

## Scripts

```bash
bun run dev            # dev server (Turbopack)
bun run build          # production build
bun run check          # oxlint (type-aware) + tsc + prettier check
bun run format         # auto-format with Prettier
bun run test           # unit tests (Vitest)
bun run test:e2e       # end-to-end tests (Playwright, needs a running backend)
```

`make check-frontend` / `make format-frontend` from the repo root wrap the same
tooling. CI runs the equivalent steps directly (`bun run lint`, `bun run
typecheck`, a Prettier check, then Vitest, the production build, and Playwright
e2e).

## Structure

- `src/app/` — App Router pages: home, `puzzle/` (main capture + solve flow),
  `test-mode/` (bundled sample puzzles, no camera needed), `about/`
- `src/components/` — `camera/` (capture + upload), `puzzle/` (display, piece
  cards, grid overlay), `ui/` (shadcn/ui primitives)
- `src/hooks/`, `src/stores/` (Zustand), `src/lib/` (API client, image utils),
  `src/types/`
- `e2e/` — Playwright specs
