# Makefile for running CI checks locally
# Run `make check` to run all checks, or individual targets

.PHONY: check check-backend check-network check-shared check-frontend check-ios \
        format format-backend format-network format-shared format-frontend format-ios \
        test-backend install-dev-backend install-dev-network \
        start-backend start-frontend stop-backend stop-frontend \
        ios-generate ios-run ios-deploy ios-test

# Run all checks (Python + Next.js + iOS)
# check-ios self-skips where swift-format is unavailable (non-macOS), so this
# aggregate stays cross-platform. CI also calls the per-component targets
# directly rather than this aggregate.
check: check-backend check-network check-shared check-frontend check-ios

# Backend checks (uses uv to run tools from the backend venv)
check-backend:
	cd backend && uv run black . --check
	cd backend && uv run isort . --check-only
	cd backend && uv run flake8 . --config=../.flake8
	cd backend && uv run pyright .

# Network checks (uses uv to run tools from the network venv)
check-network:
	cd network && uv run black . --check
	cd network && uv run isort . --check-only
	cd network && uv run flake8 . --config=../.flake8
	cd network && uv run pyright .

# Shared library checks (uses backend's uv venv since puzzle-shapes is installed there)
check-shared:
	cd backend && uv run black ../shared/puzzle_shapes --check --line-length=120
	cd backend && uv run isort ../shared/puzzle_shapes --check-only --profile=black --line-length=120
	cd backend && uv run flake8 ../shared/puzzle_shapes --config=../.flake8
	cd backend && uv run pyright ../shared/puzzle_shapes

# Frontend checks (Next.js with Bun)
check-frontend:
	cd frontend && bun run check

# iOS checks — swift-format formatting lint + SwiftLint, matching iOS CI.
# swift-format (bundled with Xcode; --strict fails on any formatting drift)
# owns layout; SwiftLint enforces the style rules in ios/.swiftlint.yml. Each
# step skips cleanly when its tool is absent (e.g. non-macOS), so the `check`
# aggregate stays cross-platform.
check-ios:
	@if xcrun --find swift-format >/dev/null 2>&1; then \
		cd ios && xcrun swift-format lint --strict --recursive Pussel PusselTests; \
	else \
		echo "Skipping swift-format lint (not found — needs macOS + Xcode)"; \
	fi
	@if command -v swiftlint >/dev/null 2>&1; then \
		cd ios && swiftlint lint --quiet; \
	else \
		echo "Skipping SwiftLint (not found — install with: brew install swiftlint)"; \
	fi

# Auto-format all code (Python + Next.js + iOS)
# Mirrors the `check` aggregate, including format-shared (the shared library is
# linted by check-shared, so it must be formatted here too).
format: format-backend format-network format-shared format-frontend format-ios

# Auto-format backend (uses uv to run tools from the backend venv)
format-backend:
	cd backend && uv run black .
	cd backend && uv run isort .

# Auto-format network (uses uv to run tools from the network venv)
format-network:
	cd network && uv run black .
	cd network && uv run isort .

# Auto-format shared library (uses backend's uv venv for consistency with check-shared)
format-shared:
	cd backend && uv run black ../shared/puzzle_shapes --line-length=120
	cd backend && uv run isort ../shared/puzzle_shapes --profile=black --line-length=120

# Auto-format frontend
format-frontend:
	cd frontend && bun run format

# Auto-format iOS Swift code in place (macOS + Xcode only). Uses Apple's
# official swift-format (bundled with Xcode) with its default style (no
# project config file). Skips cleanly where swift-format is unavailable
# (non-macOS), so the `format` aggregate stays cross-platform.
format-ios:
	@if xcrun --find swift-format >/dev/null 2>&1; then \
		cd ios && xcrun swift-format format --in-place --recursive Pussel PusselTests; \
	else \
		echo "Skipping format-ios (swift-format not found — needs macOS + Xcode)"; \
	fi

# Run backend tests with coverage (uses uv)
test-backend:
	cd backend && uv run pytest -v --cov=app --cov-report=term-missing

# Install dev dependencies (uses uv)
install-dev-backend:
	cd backend && uv sync --all-extras

install-dev-network:
	cd network && uv sync --all-extras

# Start development servers (uses uv)
start-backend:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

start-frontend:
	cd frontend && bun run dev

# Stop development servers (kills processes on their ports)
stop-backend:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "Backend not running on port 8000"

stop-frontend:
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "Frontend not running on port 3000"

# ---------------------------------------------------------------------------
# iOS app (macOS + Xcode 26 only)
# ---------------------------------------------------------------------------
# Config knobs — override on the command line, e.g.
#   make ios-run IOS_SIMULATOR="iPhone 17 Pro Max"
#   make ios-deploy IOS_DEVICE=<name-or-udid>
IOS_SCHEME     = Pussel
# Built product name (<IOS_APP_NAME>.app). Keep it in sync with IOS_SCHEME's
# target PRODUCT_NAME if you override the scheme.
IOS_APP_NAME   = Pussel
IOS_BUNDLE_ID  = dk.delectosoft.pussel
IOS_PROJECT    = ios/Pussel.xcodeproj
IOS_SIMULATOR ?= iPhone 17 Pro
IOS_DERIVED    = ios/.build

# Canonical (gitignored, machine-local) copy of the real secrets. It lives
# outside every worktree so a fresh worktree or a `git clean` never loses it;
# ios-generate restores ios/Config/Secrets.xcconfig from here automatically.
IOS_SECRETS_CANONICAL ?= $(HOME)/.config/pussel/Secrets.xcconfig
# Make does not expand a leading ~/, and the path is used inside quotes below,
# so a `make IOS_SECRETS_CANONICAL=~/path` override would be taken literally.
# Rewrite a leading ~/ to $(HOME)/ so such overrides work. `override` is needed
# because a plain assignment cannot rewrite a command-line-supplied value.
override IOS_SECRETS_CANONICAL := $(patsubst ~/%,$(HOME)/%,$(IOS_SECRETS_CANONICAL))

# Regenerate the (gitignored) Xcode project from project.yml. Requires
# `brew install xcodegen`; needs Config/Secrets.xcconfig to exist first.
ios-generate:
	@command -v xcodegen >/dev/null 2>&1 || { \
		echo "xcodegen not found. Install it with: brew install xcodegen"; exit 1; }
	@if [ ! -f ios/Config/Secrets.xcconfig ] && [ -f "$(IOS_SECRETS_CANONICAL)" ]; then \
		echo "ios/Config/Secrets.xcconfig missing; restoring from $(IOS_SECRETS_CANONICAL)"; \
		cp "$(IOS_SECRETS_CANONICAL)" ios/Config/Secrets.xcconfig; \
	fi
	@test -f ios/Config/Secrets.xcconfig || { \
		echo "ios/Config/Secrets.xcconfig is missing."; \
		echo "The Debug/Release xcconfigs #include it, so xcodebuild fails without it."; \
		echo "Restore your real values with: cp \"$(IOS_SECRETS_CANONICAL)\" ios/Config/Secrets.xcconfig"; \
		echo "Or start from the template: cp ios/Config/Secrets.example.xcconfig ios/Config/Secrets.xcconfig"; \
		exit 1; }
	cd ios && xcodegen generate

# Build, install, and launch on the iOS Simulator. Boots the target simulator
# and opens Simulator.app if it isn't already running.
ios-run: ios-generate
	xcrun simctl boot "$(IOS_SIMULATOR)" 2>/dev/null || true
	open -a Simulator
	xcodebuild build -project "$(IOS_PROJECT)" -scheme "$(IOS_SCHEME)" \
		-destination 'platform=iOS Simulator,name=$(IOS_SIMULATOR)' -derivedDataPath "$(IOS_DERIVED)"
	xcrun simctl install booted "$(IOS_DERIVED)/Build/Products/Debug-iphonesimulator/$(IOS_APP_NAME).app"
	xcrun simctl launch booted "$(IOS_BUNDLE_ID)"

# Run the unit tests on the Simulator.
ios-test: ios-generate
	xcodebuild test -project "$(IOS_PROJECT)" -scheme "$(IOS_SCHEME)" \
		-destination 'platform=iOS Simulator,name=$(IOS_SIMULATOR)' -derivedDataPath "$(IOS_DERIVED)"

# Build a Debug build, then install + launch on a connected device (iPhone/iPad).
# Requires DEVELOPMENT_TEAM in ios/Config/Secrets.xcconfig (device signing).
# The device is auto-detected; override with IOS_DEVICE=<name-or-udid>.
ios-deploy: ios-generate
	@DEVICE="$(IOS_DEVICE)"; \
	if [ -z "$$DEVICE" ]; then \
		DEVICE=$$(xcrun devicectl list devices 2>/dev/null | awk '{ ok=0; for (i=1;i<=NF;i++) if ($$i=="connected") ok=1; if (ok) for (i=1;i<=NF;i++) if ($$i ~ /^[0-9A-Fa-f]{8}-[0-9A-Fa-f]/) {print $$i; exit} }'); \
	fi; \
	if [ -z "$$DEVICE" ]; then \
		echo "No connected device found. Connect and trust a device, or pass IOS_DEVICE=<name-or-udid>."; \
		exit 1; \
	fi; \
	echo "Deploying to device: $$DEVICE"; \
	xcodebuild build -project "$(IOS_PROJECT)" -scheme "$(IOS_SCHEME)" \
		-destination 'generic/platform=iOS' -derivedDataPath "$(IOS_DERIVED)" -allowProvisioningUpdates && \
	xcrun devicectl device install app --device "$$DEVICE" \
		"$(IOS_DERIVED)/Build/Products/Debug-iphoneos/$(IOS_APP_NAME).app" && \
	xcrun devicectl device process launch --device "$$DEVICE" "$(IOS_BUNDLE_ID)"
