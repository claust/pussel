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

# Keep the two failure modes distinct: a broken usbmuxd is not the same problem
# as an absent device, and reporting it as one sends you looking at the cable.
if ! DEVICES="$(pymobiledevice3 usbmux list 2>&1)"; then
    echo "Error: could not query usbmuxd for connected devices."
    echo "$DEVICES"
    exit 1
fi

if ! grep -q '"Identifier"' <<< "$DEVICES"; then
    echo "Error: no paired iOS device found."
    echo "Connect the device via USB, unlock it, and tap Trust."
    exit 1
fi

pymobiledevice3 developer dvt screenshot "$OUT" --userspace

# The capture is 16-bit RGB (~9 MB); flattening to 8-bit cuts that to ~3 MB
# with no visible loss. Optional -- skipped when ffmpeg is unavailable.
# The scratch file sits next to $OUT so the .png suffix lets ffmpeg infer the
# format and the final mv stays within one filesystem.
if command -v ffmpeg &> /dev/null; then
    TMP="$OUT.tmp.$$.png"
    trap 'rm -f "$TMP"' EXIT
    if ffmpeg -y -loglevel error -i "$OUT" -pix_fmt rgb24 "$TMP" 2>/dev/null; then
        mv "$TMP" "$OUT"
    fi
fi

echo "Saved: $OUT"
