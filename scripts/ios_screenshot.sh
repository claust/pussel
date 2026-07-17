#!/bin/bash

# Capture a screenshot of a connected (physical) iPhone/iPad.
# Usage: ios_screenshot.sh [output.png]
# Defaults to a timestamped file under /tmp.
#
# The device must be connected via USB, unlocked, and paired/trusted.
# iOS 17+ serves the screenshot over RemoteXPC rather than lockdownd, so the
# older idevicescreenshot cannot reach it; pymobiledevice3 --userspace opens
# the required tunnel without needing root.

set -e

OUT="${1:-/tmp/iphone-$(date +%Y%m%d-%H%M%S).png}"

if ! command -v pymobiledevice3 &> /dev/null; then
    echo "Error: pymobiledevice3 is not installed."
    echo "Install it with: uv tool install pymobiledevice3"
    exit 1
fi

if ! idevice_id -l 2>/dev/null | grep -q .; then
    echo "Error: no paired iOS device found."
    echo "Connect the device via USB, unlock it, and tap Trust."
    exit 1
fi

pymobiledevice3 developer dvt screenshot "$OUT" --userspace

# The capture is 16-bit RGB (~9 MB); flattening to 8-bit cuts that to ~3 MB
# with no visible loss. Optional -- skipped when ffmpeg is unavailable.
if command -v ffmpeg &> /dev/null; then
    TMP="$(mktemp -t ios_screenshot).png"
    if ffmpeg -y -loglevel error -i "$OUT" -pix_fmt rgb24 "$TMP" 2>/dev/null; then
        mv "$TMP" "$OUT"
    else
        rm -f "$TMP"
    fi
fi

echo "Saved: $OUT"
