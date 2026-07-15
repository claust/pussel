# Pussel iOS

Native SwiftUI app covering the web frontend's **real mode**: photograph the
assembled puzzle, then photograph individual pieces and see where each one
belongs (position, rotation, confidence) — powered by the FastAPI backend.

## Requirements

- Xcode 26+ (deployment target iOS 26)
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`
- A running backend: `make start-backend` from the repo root (Debug builds
  point at `http://localhost:8000`; Release points at
  `https://pussel.thomasen.dk`)

## Setup

```bash
cd ios
cp Config/Secrets.example.xcconfig Config/Secrets.xcconfig  # fill in real values
xcodegen generate           # creates Pussel.xcodeproj (gitignored)
open Pussel.xcodeproj
```

`Secrets.xcconfig` needs three values (see comments in the example file):

- `GOOGLE_IOS_CLIENT_ID` — an OAuth client of type **iOS** (bundle id
  `dk.delectosoft.pussel`) created in the same Google Cloud project as the
  existing web client
- `GOOGLE_URL_SCHEME` — the iOS client id reversed
  (`com.googleusercontent.apps.<prefix>`)
- `GOOGLE_SERVER_CLIENT_ID` — the existing web `GOOGLE_CLIENT_ID` from
  `backend/.env` (the backend verifies ID-token audience against it)

## Architecture

- **App/** — `AppModel` (root object graph), `AppFlow` (phase state machine:
  capture → confirm trim → solving), `AppActions` (shared wizard actions)
- **Networking/** — `APIClient` (async URLSession, Bearer auth, central
  401 → silent re-auth → retry), `MultipartFormData`
- **Features/Auth/** — Google Sign-In via the GoogleSignIn SDK; the backend
  JWT lives in memory only and is reminted from Google's persisted session
- **Features/Capture/** — puzzle photo (camera or PhotosPicker) →
  `detect-frame` → accept/retake
- **Features/Solve/** — persistent camera with manual shutter (device),
  PhotosPicker fallback (Simulator), serial prediction queue, overlay
  rendering that mirrors `frontend/src/components/puzzle/puzzle-detail.tsx`

The backend's puzzle store is in-memory, so a `puzzle_id` dies when the
backend restarts; the app keeps the trimmed image and offers one-tap
re-upload, then re-queues affected pieces.

## Build & test from the CLI

```bash
xcodebuild build -project Pussel.xcodeproj -scheme Pussel \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath .build

xcodebuild test -project Pussel.xcodeproj -scheme Pussel \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath .build

xcrun simctl install booted .build/Build/Products/Debug-iphonesimulator/Pussel.app
xcrun simctl launch booted dk.delectosoft.pussel
```

## Debug driving (Simulator has no camera)

Debug builds accept two escape hatches, both compiled out of Release:

- **Auth bypass** — launch with a JWT from
  `backend/scripts/generate_test_token.py`:

  ```bash
  TOKEN=$(cd backend && uv run python scripts/generate_test_token.py)
  SIMCTL_CHILD_PUSSEL_DEBUG_TOKEN=$TOKEN xcrun simctl launch booted dk.delectosoft.pussel
  ```

- **Command channel** — drive the wizard without tapping. Write a command URL
  to the command file (default `/tmp/pussel-debug-command`, overridable with
  `SIMCTL_CHILD_PUSSEL_DEBUG_COMMAND_FILE`) and post a Darwin notification:

  ```bash
  echo "pusseldebug://trim?puzzle=/absolute/host/path.jpg" > /tmp/pussel-debug-command
  xcrun simctl spawn booted notifyutil -p dk.delectosoft.pussel.debug
  ```

  Commands: `trim?puzzle=<path>`, `accept`, `piece?path=<path>`, `reupload`,
  `reset`. Simulator apps can read host file paths directly, so fixtures can
  live anywhere (e.g. `frontend/public/test-puzzles/`).
