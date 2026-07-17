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
make ios-deploy    # build → install → launch on a connected device (iPhone/iPad)
```

Override the simulator or target device on the command line:

```bash
make ios-run IOS_SIMULATOR="iPhone 17 Pro Max"
make ios-deploy IOS_DEVICE=<name-or-udid>   # device is auto-detected otherwise
```

`ios-deploy` needs `DEVELOPMENT_TEAM` set in `Config/Secrets.xcconfig` (device
signing) and a device that's connected and trusts this Mac.

## Screenshots

`make ios-screenshot` captures a booted Simulator or a connected device at
native resolution (a device shot takes ~2-3s):

```bash
make ios-screenshot                          # → /tmp/iphone-<timestamp>.png
make ios-screenshot OUT=/path/shot.png
```

With no target given it uses whichever one is available. When several are — a
connected device *and* a Simulator, two booted Simulators, or two connected
devices — it lists them and exits instead of guessing, since a screenshot of
the wrong target looks like a successful capture of a broken app. (`simctl io
booted` has exactly that failure mode: with two Simulators booted it silently
picks one.) Pick a target with:

```bash
make ios-screenshot TARGET=device
make ios-screenshot TARGET=simulator
make ios-screenshot SIM="iPhone 17 Pro"      # name or udid; implies simulator
make ios-screenshot DEV=<udid>               # implies device
```

A connected device must be paired/trusted and **unlocked**, and needs
`uv tool install pymobiledevice3`. Simulator capture needs only Xcode. ffmpeg
is used if present (see below) but is not required.

Reach for a device rather than the Simulator whenever the camera is involved —
the Simulator has none, so the piece-capture UI can only be exercised on real
hardware.

Note that the obvious alternatives for *device* capture do not work on iOS 17+,
which moved the screenshot service from `lockdownd` to RemoteXPC:

- `idevicescreenshot` fails with "Could not start screenshotr service: Invalid
  service". Its suggestion to mount the Developer disk image is a red herring —
  the DDI is already mounted (`ideviceimagemounter list` → `Status: Complete`).
- `xcrun devicectl` has no screenshot subcommand.
- The iPhone's AVFoundation entries are Continuity Camera, not the device screen.

`scripts/ios_screenshot.sh` therefore uses `pymobiledevice3 developer dvt
screenshot <out.png> --userspace` for devices (and `xcrun simctl io <udid>
screenshot` for Simulators). The `--userspace` flag opens the required iOS 17+
RSD tunnel without root; without it the command demands
`sudo pymobiledevice3 remote tunneld`.

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

## Formatting & linting

Two complementary tools:

- **Formatting** — Apple's [swift-format](https://github.com/swiftlang/swift-format)
  (bundled with Xcode 26, no extra install) owns layout, using its default
  style (2-space indent, no project config).
- **Linting** — [SwiftLint](https://github.com/realm/SwiftLint)
  (`brew install swiftlint`) enforces the style rules in `ios/.swiftlint.yml`;
  this is what iOS CI runs. The two formatting rules swift-format owns
  (`opening_brace`, `trailing_comma`) are disabled there so the tools don't
  fight each other.

```bash
make format-ios    # format all Swift sources in place (swift-format)
make check-ios     # swift-format --strict lint + SwiftLint (matches CI)
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
  list on the home screen), `camera[?open=0]` (force the piece camera cover
  open/closed — needed on the Simulator, which has no
  `PieceCameraSession.isCameraAvailable`), `previewloop?path=<path>` /
  `previewloop?stop=1` (feed a host image through the live piece-preview
  overlay pipeline on a ~3Hz timer, since the Simulator has no camera to
  stream real frames from). Simulator apps can read host file paths
  directly, so fixtures can live anywhere (e.g. `frontend/public/test-puzzles/`).
