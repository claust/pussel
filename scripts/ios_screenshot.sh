#!/bin/bash

# Capture a screenshot from a connected iPhone/iPad or a booted Simulator.
#
# Usage: ios_screenshot.sh [--device | --simulator[=<name-or-udid>]] [OUT]
#
# With no target flag the script uses whichever target is available. When more
# than one is available it stops and prints how to choose rather than guessing:
# `simctl io booted` silently picks one of several booted simulators, and a
# screenshot of the wrong target is worse than an error, because it looks like
# a successful capture of a broken app.
#
# Physical devices must be connected via USB, paired/trusted, and unlocked.
# iOS 17+ serves the screenshot over RemoteXPC rather than lockdownd, so the
# older idevicescreenshot cannot reach it; pymobiledevice3 --userspace opens
# the required tunnel without needing root.

set -euo pipefail

TARGET="auto"
SIM_SPEC=""
DEV_SPEC=""
DEV_UDID=""
OUT=""

usage() {
    cat <<'USAGE'
Usage: ios_screenshot.sh [--device[=<udid>] | --simulator[=<name-or-udid>]] [OUT]

  --device                     Capture the connected physical device.
  --device=<udid>              Capture a specific connected device.
  --simulator                  Capture the booted Simulator.
  --simulator=<name-or-udid>   Capture a specific booted Simulator.
  -h, --help                   Show this message.

OUT defaults to a timestamped file under /tmp. With no target flag, the only
available target is used; if several are available the script lists them and
exits rather than picking one.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --device) TARGET="device" ;;
        --device=*) TARGET="device"; DEV_SPEC="${1#*=}" ;;
        --simulator|--sim) TARGET="simulator" ;;
        --simulator=*|--sim=*) TARGET="simulator"; SIM_SPEC="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Error: unknown option '$1'." >&2; echo >&2; usage >&2; exit 2 ;;
        # An empty positional is what `make ios-screenshot` passes when OUT is
        # unset, so it must mean "use the default", not an empty filename.
        *) [ -n "$1" ] && OUT="$1" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Discover targets
# ---------------------------------------------------------------------------

# Lines of "<udid><TAB><label>" for each connected device.
DEVICES=""
USBMUX_ERROR=""

# Read `pymobiledevice3 usbmux list` JSON on stdin, emit "<udid><TAB><label>"
# per device. A real JSON parser rather than line-oriented matching: fields are
# read per object, so a device missing one cannot borrow another's value, and
# whitespace/formatting carries no meaning. python3 ships with the Xcode command
# line tools this repo already needs, and pymobiledevice3 is itself Python.
# ProductType rather than DeviceName: the name arrives with \uXXXX escapes.
parse_devices() {
    python3 -c '
import json, sys
try:
    devices = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if not isinstance(devices, list):
    sys.exit(0)
for d in devices:
    if not isinstance(d, dict):
        continue
    udid = d.get("UniqueDeviceID") or d.get("Identifier")
    if not udid:
        continue
    label = "%s (iOS %s)" % (d.get("ProductType", "device"), d.get("ProductVersion", "unknown"))
    print("%s\t%s" % (udid, label))
'
}

# Indent a block of text; awk rather than sed so a multi-line value stays intact.
indent() { awk -v p="$1" '{print p $0}'; }

if command -v pymobiledevice3 &> /dev/null; then
    # Keep the two failure modes distinct: a broken usbmuxd is not the same
    # problem as an absent device, and conflating them sends you to the cable.
    if DEVICE_JSON="$(pymobiledevice3 usbmux list 2>&1)"; then
        if command -v python3 &> /dev/null; then
            DEVICES="$(parse_devices <<< "$DEVICE_JSON")"
        else
            # Say so rather than let an empty list read as "nothing plugged in".
            USBMUX_ERROR="python3 is needed to read the device list, but is not installed."
        fi
    else
        USBMUX_ERROR="$DEVICE_JSON"
    fi
fi

# Lines of "<udid><TAB><name>" for each booted simulator.
SIMS=""
if command -v xcrun &> /dev/null; then
    SIMS="$(xcrun simctl list devices booted 2>/dev/null \
        | sed -n 's/^ *\(.*\) (\([0-9A-Fa-f-]\{36\}\)) (Booted).*/\2	\1/p' || true)"
fi

# wc -l miscounts a single unterminated line, and an empty list is not one line.
count_lines() {
    if [ -z "$1" ]; then echo 0; else grep -c '' <<< "$1"; fi
}

SIM_COUNT="$(count_lines "$SIMS")"
DEVICE_COUNT="$(count_lines "$DEVICES")"

list_devices() {
    if [ -n "$DEVICES" ]; then
        while IFS="	" read -r udid label; do
            echo "  device     $label ($udid)"
        done <<< "$DEVICES"
    fi
}

list_sims() {
    if [ -n "$SIMS" ]; then
        while IFS="	" read -r udid name; do
            echo "  simulator  $name ($udid)"
        done <<< "$SIMS"
    fi
}

list_targets() {
    list_devices
    list_sims
}

# The runnable command for each target, so the message can be pasted as-is.
# A single device needs no udid, so keep the common suggestion short.
suggest_devices() {
    if [ "$DEVICE_COUNT" -eq 1 ]; then
        echo "  make ios-screenshot TARGET=device"
    elif [ -n "$DEVICES" ]; then
        while IFS="	" read -r udid label; do
            echo "  make ios-screenshot DEV=$udid   # $label"
        done <<< "$DEVICES"
    fi
}

suggest_sims() {
    if [ -n "$SIMS" ]; then
        while IFS="	" read -r udid name; do
            echo "  make ios-screenshot SIM=$udid   # $name"
        done <<< "$SIMS"
    fi
}

# ---------------------------------------------------------------------------
# Resolve the target
# ---------------------------------------------------------------------------

if [ "$TARGET" = "auto" ]; then
    TOTAL=$((DEVICE_COUNT + SIM_COUNT))
    if [ "$TOTAL" -eq 0 ]; then
        {
            echo "Error: no screenshot target found."
            echo
            echo "Checked:"
            if [ -n "$USBMUX_ERROR" ]; then
                echo "  connected device   could not query usbmuxd:"
                indent "                       " <<< "$USBMUX_ERROR"
            elif command -v pymobiledevice3 &> /dev/null; then
                echo "  connected device   none (pymobiledevice3 usbmux list)"
            else
                echo "  connected device   pymobiledevice3 is not installed"
            fi
            echo "  booted simulator   none (xcrun simctl list devices booted)"
            echo
            echo "Fix by doing one of:"
            echo "  - Connect an iPhone via USB, unlock it, and tap Trust."
            if ! command -v pymobiledevice3 &> /dev/null; then
                echo "    Physical devices also need: uv tool install pymobiledevice3"
            fi
            echo "  - Boot a simulator and launch the app: make ios-run"
        } >&2
        exit 1
    fi
    if [ "$TOTAL" -gt 1 ]; then
        {
            echo "Error: $TOTAL screenshot targets are available; pick one explicitly."
            echo
            echo "Available:"
            list_targets
            echo
            echo "Re-run with one of:"
            suggest_devices
            suggest_sims
        } >&2
        exit 1
    fi
    if [ "$DEVICE_COUNT" -eq 1 ]; then TARGET="device"; else TARGET="simulator"; fi
fi

SIM_UDID=""
if [ "$TARGET" = "simulator" ]; then
    if [ -z "$SIMS" ]; then
        {
            echo "Error: no booted simulator found."
            echo
            echo "Boot one with: make ios-run"
            echo "Or list what exists: xcrun simctl list devices"
        } >&2
        exit 1
    fi
    if [ -n "$SIM_SPEC" ]; then
        SIM_UDID="$(awk -F'\t' -v s="$SIM_SPEC" '$1==s || $2==s {print $1; exit}' <<< "$SIMS")"
        if [ -z "$SIM_UDID" ]; then
            {
                echo "Error: no booted simulator matches '$SIM_SPEC'."
                echo
                echo "Booted simulators:"
                list_sims
                echo
                echo "Match is exact, on either the name or the udid."
            } >&2
            exit 1
        fi
    elif [ "$SIM_COUNT" -gt 1 ]; then
        {
            echo "Error: $SIM_COUNT simulators are booted; pick one explicitly."
            echo
            echo "Booted simulators:"
            list_sims
            echo
            echo "Re-run with one of:"
            suggest_sims
        } >&2
        exit 1
    else
        SIM_UDID="$(cut -f1 <<< "$SIMS")"
    fi
fi

if [ "$TARGET" = "device" ]; then
    if ! command -v pymobiledevice3 &> /dev/null; then
        {
            echo "Error: pymobiledevice3 is not installed, so no device can be captured."
            echo
            echo "Install it with: uv tool install pymobiledevice3"
        } >&2
        exit 1
    fi
    if [ -n "$USBMUX_ERROR" ]; then
        {
            echo "Error: could not query usbmuxd for connected devices."
            echo
            indent "  " <<< "$USBMUX_ERROR"
        } >&2
        exit 1
    fi
    if [ -z "$DEVICES" ]; then
        {
            echo "Error: no paired iOS device found."
            echo
            echo "Connect the device via USB, unlock it, and tap Trust."
            echo "Then confirm it is visible with: pymobiledevice3 usbmux list"
        } >&2
        exit 1
    fi
    if [ -n "$DEV_SPEC" ]; then
        DEV_UDID="$(awk -F'\t' -v s="$DEV_SPEC" '$1==s {print $1; exit}' <<< "$DEVICES")"
        if [ -z "$DEV_UDID" ]; then
            {
                echo "Error: no connected device matches udid '$DEV_SPEC'."
                echo
                echo "Connected devices:"
                list_devices
            } >&2
            exit 1
        fi
    elif [ "$DEVICE_COUNT" -gt 1 ]; then
        # Without a udid pymobiledevice3 picks one of several devices on its
        # own, so stop here for the same reason as the simulator case.
        {
            echo "Error: $DEVICE_COUNT devices are connected; pick one explicitly."
            echo
            echo "Connected devices:"
            list_devices
            echo
            echo "Re-run with one of:"
            suggest_devices
        } >&2
        exit 1
    else
        DEV_UDID="$(cut -f1 <<< "$DEVICES")"
    fi
fi

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

# $$ as well as the timestamp: two runs within the same second would otherwise
# land on the same default path and the first shot would be lost.
OUT="${OUT:-/tmp/iphone-$(date +%Y%m%d-%H%M%S)-$$.png}"

if [ "$TARGET" = "device" ]; then
    # --udid rather than letting pymobiledevice3 choose: the choice is already
    # made above, and leaving it implicit is what allows a silent wrong target.
    pymobiledevice3 developer dvt screenshot "$OUT" --userspace --udid "$DEV_UDID"
else
    # simctl narrates to stderr on success ("Detected file type...", "Note: No
    # display specified..."), so hold its output and surface it only on failure.
    if ! SIM_ERR="$(xcrun simctl io "$SIM_UDID" screenshot "$OUT" 2>&1 > /dev/null)"; then
        {
            echo "Error: simctl could not screenshot simulator $SIM_UDID."
            echo
            indent "  " <<< "$SIM_ERR"
        } >&2
        exit 1
    fi
fi

# Device captures are 16-bit RGB (~9 MB); flattening to 8-bit cuts that to ~3 MB
# with no visible loss, and drops the simulator's unused alpha channel.
# Optional -- skipped when ffmpeg is unavailable.
# The scratch file sits next to $OUT so the .png suffix lets ffmpeg infer the
# format and the final mv stays within one filesystem.
if command -v ffmpeg &> /dev/null; then
    TMP="$OUT.tmp.$$.png"
    trap 'rm -f "$TMP"' EXIT
    if ffmpeg -y -loglevel error -i "$OUT" -pix_fmt rgb24 "$TMP" 2>/dev/null; then
        mv "$TMP" "$OUT"
    fi
fi

if [ "$TARGET" = "device" ]; then
    echo "Saved: $OUT  (device: $(awk -F'\t' -v u="$DEV_UDID" '$1==u {print $2}' <<< "$DEVICES"))"
else
    echo "Saved: $OUT  (simulator: $(awk -F'\t' -v u="$SIM_UDID" '$1==u {print $2}' <<< "$SIMS"))"
fi
