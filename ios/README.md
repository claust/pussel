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
- **Persistence/** — `PuzzleStore`, on-device storage of every puzzle under
  `Documents/Puzzles/<uuid>/` (a `manifest.json`, the `trimmed.jpg` picture,
  and a `pieces/` folder of per-piece image files). No server storage: a
  session is saved on every change (create, prediction done, remove, reupload)
  and rehydrated purely from disk when reopened
- **Networking/** — `APIClient` (async URLSession, Bearer auth, central
  401 → silent re-auth → retry), `MultipartFormData`
- **Features/Auth/** — Google Sign-In via the GoogleSignIn SDK; the backend
  JWT lives in memory only and is reminted from Google's persisted session
- **Features/Capture/** — puzzle photo (camera or PhotosPicker) →
  `detect-frame` → accept/retake
- **Features/Solve/** — persistent camera with manual shutter (device),
  PhotosPicker fallback (Simulator), serial prediction queue, overlay
  rendering that mirrors `frontend/src/components/puzzle/puzzle-detail.tsx`
- **Features/Library/** — `SavedPuzzlesSection`, the list of stored puzzles on
  the home screen below the capture buttons (tap to reopen, long-press to
  delete)

The backend's puzzle store is in-memory, so a `puzzle_id` dies when the
backend restarts; the app keeps the trimmed image and offers one-tap
re-upload, then re-queues affected pieces.

## Build & test from the CLI

The repo-root `Makefile` wraps the common flows (all run `xcodegen generate`
first):

```bash
make ios-run       # build → install → launch on the Simulator
make ios-test      # run the unit tests on the Simulator
make ios-deploy    # build → install → launch on a connected iPhone
```

Override the simulator or target device on the command line:

```bash
make ios-run IOS_SIMULATOR="iPhone 17 Pro Max"
make ios-deploy IOS_DEVICE=<name-or-udid>   # device is auto-detected otherwise
```

`ios-deploy` needs `DEVELOPMENT_TEAM` set in `Config/Secrets.xcconfig` (device
signing) and a device that's connected and trusts this Mac.

<details>
<summary>Equivalent raw <code>xcodebuild</code> / <code>simctl</code> commands</summary>

```bash
xcodebuild build -project Pussel.xcodeproj -scheme Pussel \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath .build

xcodebuild test -project Pussel.xcodeproj -scheme Pussel \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath .build

xcrun simctl install booted .build/Build/Products/Debug-iphonesimulator/Pussel.app
xcrun simctl launch booted dk.delectosoft.pussel
```

</details>

## Formatting

Swift code is formatted with Apple's [swift-format](https://github.com/swiftlang/swift-format)
(bundled with Xcode 26 — no extra install), using its default style (no
project config).

```bash
make format-ios    # format all Swift sources in place
make check-ios     # lint formatting without modifying files (--strict)
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
  `reset`, `open?index=<n>` / `delete?index=<n>` (index into the saved-puzzles
  list on the home screen). Simulator apps can read host file paths directly,
  so fixtures can live anywhere (e.g. `frontend/public/test-puzzles/`).
