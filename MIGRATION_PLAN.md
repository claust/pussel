# Flutter to Next.js Migration Plan

This document outlines the migration strategy for the Pussel frontend from Flutter to React/Next.js with Bun.

## Tech Stack

| Current (Flutter) | New (React) |
|-------------------|-------------|
| Flutter SDK | Next.js 15 (App Router) |
| Dart | TypeScript |
| Provider (unused) | Zustand or React Context |
| Dio | Native fetch / SWR |
| camera plugin | Browser MediaDevices API |
| image package | Canvas API / Browser Image APIs |
| Material Design 3 | Tailwind CSS + shadcn/ui |
| dart-analyze | OxLint + TypeScript |
| - | Bun (runtime + package manager) |

---

## Phase 1: Project Setup

### 1.1 Initialize Next.js Project

```bash
cd /Users/claus/Repos/pussel
bun create next-app frontend-next --typescript --tailwind --eslint --app --src-dir
cd frontend-next
```

### 1.2 Install Dependencies

```bash
# UI Components
bun add @radix-ui/react-dialog @radix-ui/react-slot class-variance-authority clsx tailwind-merge lucide-react

# State Management
bun add zustand

# HTTP & Data Fetching
bun add swr

# Image Processing (optional, for advanced cropping)
bun add browser-image-compression

# Dev Dependencies
bun add -d @types/node
```

### 1.3 Configure OxLint

```bash
# Install oxlint
bun add -d oxlint

# Create oxlint config
```

Create `oxlint.json`:
```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["typescript", "react", "nextjs"],
  "rules": {
    "no-unused-vars": "warn",
    "no-console": "warn",
    "eqeqeq": "error",
    "no-var": "error",
    "prefer-const": "error",
    "react/jsx-key": "error",
    "react/no-array-index-key": "warn",
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn",
    "typescript/no-explicit-any": "warn",
    "nextjs/no-html-link-for-pages": "error"
  },
  "ignorePatterns": [
    "node_modules",
    ".next",
    "out",
    "*.config.js",
    "*.config.ts"
  ]
}
```

Update `package.json` scripts:
```json
{
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "next build",
    "start": "next start",
    "lint": "oxlint --config oxlint.json src/",
    "lint:fix": "oxlint --config oxlint.json --fix src/",
    "typecheck": "tsc --noEmit",
    "check": "bun run lint && bun run typecheck",
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

### 1.4 Configure shadcn/ui

```bash
bunx shadcn@latest init
bunx shadcn@latest add button card dialog
```

### 1.5 Project Structure

```
frontend-next/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── layout.tsx          # Root layout with theme
│   │   ├── page.tsx            # Home screen
│   │   ├── puzzle/
│   │   │   └── page.tsx        # Puzzle workflow screen
│   │   └── test-mode/
│   │       ├── page.tsx        # Grid selection
│   │       ├── puzzle/[id]/
│   │       │   └── page.tsx    # Test puzzle screen
│   │       └── select/
│   │           └── page.tsx    # Puzzle selection
│   ├── components/
│   │   ├── ui/                 # shadcn components
│   │   ├── camera/
│   │   │   ├── camera-view.tsx
│   │   │   └── camera-modal.tsx
│   │   ├── puzzle/
│   │   │   ├── puzzle-detail.tsx
│   │   │   ├── grid-overlay.tsx
│   │   │   └── piece-card.tsx
│   │   └── rotation-selector.tsx
│   ├── hooks/
│   │   ├── use-camera.ts       # Camera access hook
│   │   └── use-puzzle.ts       # Puzzle state hook
│   ├── lib/
│   │   ├── api.ts              # API client
│   │   ├── image-utils.ts      # Image processing
│   │   └── utils.ts            # General utilities
│   ├── stores/
│   │   └── puzzle-store.ts     # Zustand store
│   └── types/
│       └── index.ts            # TypeScript types
├── public/
│   └── test-puzzles/           # Bundled test images
├── oxlint.json                 # OxLint configuration
├── next.config.ts
├── tailwind.config.ts
└── package.json
```

---

## Phase 2: Core Infrastructure

### 2.1 TypeScript Types (`src/types/index.ts`)

```typescript
export interface Position {
  x: number;  // 0-1 normalized
  y: number;  // 0-1 normalized
  normalized: boolean;
}

export interface Piece {
  position: Position;
  confidence: number;  // 0-1
  rotation: 0 | 90 | 180 | 270;
  imageData?: string;  // base64 or blob URL
}

export interface Puzzle {
  puzzleId: string;
  imageUrl?: string;
}

export type GridSize = '2x2' | '3x3';

export interface TestPuzzle {
  id: string;
  name: string;
  path: string;
}

export const GRID_DIMENSIONS: Record<GridSize, { dimension: number; totalCells: number }> = {
  '2x2': { dimension: 2, totalCells: 4 },
  '3x3': { dimension: 3, totalCells: 9 },
};
```

### 2.2 API Client (`src/lib/api.ts`)

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function checkHealth(): Promise<boolean> {
  const res = await fetch(`${API_BASE}/health`);
  return res.ok;
}

export async function uploadPuzzle(imageBlob: Blob): Promise<Puzzle> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'puzzle.jpg');

  const res = await fetch(`${API_BASE}/api/v1/puzzle/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Failed to upload puzzle');
  return res.json();
}

export async function processPiece(puzzleId: string, pieceBlob: Blob): Promise<Piece> {
  const formData = new FormData();
  formData.append('file', pieceBlob, 'piece.jpg');

  const res = await fetch(`${API_BASE}/api/v1/puzzle/${puzzleId}/piece`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Failed to process piece');
  return res.json();
}
```

### 2.3 Zustand Store (`src/stores/puzzle-store.ts`)

```typescript
import { create } from 'zustand';
import type { Puzzle, Piece } from '@/types';

interface PuzzleState {
  puzzle: Puzzle | null;
  puzzleImage: string | null;  // blob URL or base64
  pieces: Piece[];
  isLoading: boolean;
  error: string | null;

  setPuzzle: (puzzle: Puzzle, imageUrl: string) => void;
  addPiece: (piece: Piece) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const usePuzzleStore = create<PuzzleState>((set) => ({
  puzzle: null,
  puzzleImage: null,
  pieces: [],
  isLoading: false,
  error: null,

  setPuzzle: (puzzle, imageUrl) => set({ puzzle, puzzleImage: imageUrl, pieces: [] }),
  addPiece: (piece) => set((state) => ({ pieces: [...state.pieces, piece] })),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () => set({ puzzle: null, puzzleImage: null, pieces: [], error: null }),
}));
```

---

## Phase 3: Component Migration

### 3.1 Camera Hook (`src/hooks/use-camera.ts`)

```typescript
import { useState, useRef, useCallback, useEffect } from 'react';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 1920, height: 1080 }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsReady(true);
      }
    } catch {
      setError('Camera access denied');
    }
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    setIsReady(false);
  }, []);

  const capture = useCallback(async (): Promise<Blob | null> => {
    if (!videoRef.current || !isReady) return null;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx?.drawImage(videoRef.current, 0, 0);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.9);
    });
  }, [isReady]);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { videoRef, isReady, error, start, stop, capture };
}
```

### 3.2 Screen Mapping

| Flutter Screen | Next.js Route | Component |
|----------------|---------------|-----------|
| HomeScreen | `/` | `app/page.tsx` |
| CameraScreen | Modal component | `components/camera/camera-modal.tsx` |
| PuzzleScreen | `/puzzle` | `app/puzzle/page.tsx` |
| GridSelectionScreen | `/test-mode` | `app/test-mode/page.tsx` |
| PuzzleSelectionScreen | `/test-mode/select` | `app/test-mode/select/page.tsx` |
| TestPuzzleScreen | `/test-mode/puzzle/[id]` | `app/test-mode/puzzle/[id]/page.tsx` |
| PieceSelectionScreen | Modal component | `components/puzzle/piece-selector-modal.tsx` |

### 3.3 Widget Mapping

| Flutter Widget | React Component |
|----------------|-----------------|
| PuzzleDetail | `components/puzzle/puzzle-detail.tsx` |
| GridOverlay | `components/puzzle/grid-overlay.tsx` |
| RotationSelectorDialog | `components/rotation-selector.tsx` |
| PlatformImageWidget | Native `<img>` or `next/image` |

---

## Phase 4: Feature Implementation Order

### Sprint 1: Foundation
1. [ ] Project setup with Bun + Next.js
2. [ ] OxLint + TypeScript configuration
3. [ ] Tailwind + shadcn/ui configuration
4. [ ] TypeScript types
5. [ ] API client
6. [ ] Basic theme (colors from Flutter)

### Sprint 2: Core Pages
1. [ ] Home page with navigation
2. [ ] Zustand store setup
3. [ ] Camera hook implementation
4. [ ] Camera modal component

### Sprint 3: Puzzle Workflow
1. [ ] Puzzle page layout
2. [ ] Puzzle upload flow
3. [ ] Piece capture flow
4. [ ] PuzzleDetail component
5. [ ] Piece grid display

### Sprint 4: Test Mode
1. [ ] Grid selection page
2. [ ] Puzzle selection page
3. [ ] Test puzzle page
4. [ ] GridOverlay component
5. [ ] Piece selector modal
6. [ ] Rotation selector

### Sprint 5: Polish
1. [ ] Error handling & loading states
2. [ ] Responsive design
3. [ ] Dark mode support
4. [ ] Animation/transitions
5. [ ] Testing

---

## Phase 5: Theme Migration

### Color Palette (tailwind.config.ts)

```typescript
const config = {
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4A6572',
          foreground: '#FFFFFF',
        },
        secondary: {
          DEFAULT: '#F9AA33',
          foreground: '#222222',
        },
        background: '#F5F5F5',
        foreground: '#222222',
        muted: {
          DEFAULT: '#F5F5F5',
          foreground: '#757575',
        },
        destructive: {
          DEFAULT: '#B00020',
          foreground: '#FFFFFF',
        },
      },
    },
  },
};
```

---

## Phase 6: Image Processing

### Crop Cell Function (`src/lib/image-utils.ts`)

```typescript
export async function cropCell(
  imageBlob: Blob,
  gridSize: '2x2' | '3x3',
  cellIndex: number,
  rotationDegrees: number
): Promise<Blob> {
  const dimension = gridSize === '2x2' ? 2 : 3;
  const img = await createImageBitmap(imageBlob);

  const cellWidth = img.width / dimension;
  const cellHeight = img.height / dimension;
  const row = Math.floor(cellIndex / dimension);
  const col = cellIndex % dimension;

  const canvas = document.createElement('canvas');
  canvas.width = cellWidth;
  canvas.height = cellHeight;
  const ctx = canvas.getContext('2d')!;

  // Apply rotation
  ctx.translate(cellWidth / 2, cellHeight / 2);
  ctx.rotate((rotationDegrees * Math.PI) / 180);
  ctx.translate(-cellWidth / 2, -cellHeight / 2);

  ctx.drawImage(
    img,
    col * cellWidth, row * cellHeight, cellWidth, cellHeight,
    0, 0, cellWidth, cellHeight
  );

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.9);
  });
}
```

---

## Phase 7: Testing Strategy

### Unit Tests (Vitest)
```bash
bun add -d vitest @testing-library/react @testing-library/jest-dom jsdom @vitejs/plugin-react
```

Create `vitest.config.ts`:
```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### E2E Tests (Playwright)
```bash
bun add -d @playwright/test
bunx playwright install
```

### Test Coverage
- [ ] API client functions
- [ ] Zustand store actions
- [ ] Image processing utilities
- [ ] Camera hook (mocked)
- [ ] Component rendering
- [ ] User flows (E2E)

---

## Linting & Code Quality

### OxLint vs ESLint

OxLint is chosen over ESLint for:
- **Speed**: 50-100x faster than ESLint
- **Zero config**: Works out of the box
- **TypeScript support**: Native TypeScript parsing
- **React/Next.js plugins**: Built-in support

### Pre-commit Hook

Install Husky:
```bash
bun add -d husky lint-staged
bunx husky init
```

Configure `.husky/pre-commit`:
```bash
#!/usr/bin/env sh
bun run lint-staged
```

Configure `lint-staged` in `package.json`:
```json
{
  "lint-staged": {
    "*.{ts,tsx}": [
      "oxlint --fix",
      "prettier --write"
    ],
    "*.{json,md}": [
      "prettier --write"
    ]
  }
}
```

### Prettier (for formatting)
```bash
bun add -d prettier prettier-plugin-tailwindcss
```

Create `.prettierrc`:
```json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100,
  "plugins": ["prettier-plugin-tailwindcss"]
}
```

Update scripts:
```json
{
  "scripts": {
    "format": "prettier --write \"src/**/*.{ts,tsx,json,md}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,json,md}\""
  }
}
```

---

## Migration Checklist

### Pre-Migration
- [ ] Document all current features
- [ ] Screenshot current UI for reference
- [ ] Ensure backend API is stable

### During Migration
- [ ] Keep Flutter app running in parallel
- [ ] Migrate one screen at a time
- [ ] Test against same backend
- [ ] Compare behavior with Flutter version

### Post-Migration
- [ ] Full regression testing
- [ ] Performance comparison
- [ ] Remove Flutter frontend directory
- [ ] Update CI/CD pipelines
- [ ] Update documentation

---

## Key Differences from Flutter

| Aspect | Flutter | Next.js |
|--------|---------|---------|
| Rendering | Skia canvas | DOM/CSS |
| State | StatefulWidget | React hooks + Zustand |
| Routing | Navigator | File-based routing |
| Camera | Native plugin | MediaDevices API |
| Images | File/Uint8List | Blob/base64/URL |
| Styling | Widget props | Tailwind classes |
| Platform detection | Platform.isWeb | typeof window |
| Linting | dart-analyze | OxLint |

---

## Risk Mitigation

### Camera Access
- **Risk**: Browser camera API more limited than Flutter plugin
- **Mitigation**: Graceful degradation, file upload fallback

### Image Processing
- **Risk**: Canvas API performance for large images
- **Mitigation**: Web Workers for heavy processing, image compression

### Mobile Experience
- **Risk**: PWA less native feel than Flutter
- **Mitigation**: Consider React Native for mobile-first later, or keep as responsive web app

### Cross-Origin Issues
- **Risk**: CORS with backend API
- **Mitigation**: Backend already has CORS configured, verify Next.js API routes if needed

---

## Environment Variables

```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## CI/CD Updates

Update `.github/workflows/frontend-ci.yml`:

```yaml
name: Frontend CI (Next.js)

on:
  push:
    branches: [master, main, dev]
    paths:
      - 'frontend-next/**'
  pull_request:
    paths:
      - 'frontend-next/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: oven-sh/setup-bun@v2
        with:
          bun-version: latest

      - name: Install dependencies
        run: cd frontend-next && bun install

      - name: Lint (OxLint)
        run: cd frontend-next && bun run lint

      - name: Type check
        run: cd frontend-next && bun run typecheck

      - name: Format check
        run: cd frontend-next && bun run format:check

      - name: Build
        run: cd frontend-next && bun run build

      - name: Test
        run: cd frontend-next && bun test
```

---

## Next Steps

1. **Review this plan** and approve approach
2. **Set up initial project** with Bun + Next.js
3. **Implement Phase 1-2** (foundation + infrastructure)
4. **Iterate on remaining phases**

The Flutter frontend can remain operational during migration, allowing side-by-side testing.
