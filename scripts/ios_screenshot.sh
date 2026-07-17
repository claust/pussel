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
OUT=""

usage() {
    cat <<'USAGE'
Usage: ios_screenshot.sh [--device | --simulator[=<name-or-udid>]] [OUT]

  --device                     Capture the connected physical device.
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

DEVICE_LABEL=""
USBMUX_ERROR=""

# First value for a JSON string key, reading stdin. Tolerates both `"k": "v"`
# and `"k" : "v"` spacing so a formatting change upstream degrades to an empty
# label rather than a wrong one.
json_string() {
    sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1
}

# Indent a block of text; awk rather than sed so a multi-line value stays intact.
indent() { awk -v p="$1" '{print p $0}'; }

if command -v pymobiledevice3 &> /dev/null; then
    # Keep the two failure modes distinct: a broken usbmuxd is not the same
    # problem as an absent device, and conflating them sends you to the cable.
    if DEVICE_JSON="$(pymobiledevice3 usbmux list 2>&1)"; then
        if grep -q '"Identifier"' <<< "$DEVICE_JSON"; then
            # ProductType/ProductVersion rather than DeviceName: the name comes
            # back with \uXXXX escapes that bash cannot readably decode.
            PRODUCT="$(json_string ProductType <<< "$DEVICE_JSON")"
            VERSION="$(json_string ProductVersion <<< "$DEVICE_JSON")"
            DEVICE_LABEL="${PRODUCT:-device} (iOS ${VERSION:-unknown})"
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

sim_count() {
    if [ -z "$SIMS" ]; then echo 0; else wc -l <<< "$SIMS" | tr -d ' '; fi
}

SIM_COUNT="$(sim_count)"
DEVICE_COUNT=0
[ -n "$DEVICE_LABEL" ] && DEVICE_COUNT=1

list_sims() {
    if [ -n "$SIMS" ]; then
        while IFS="	" read -r udid name; do
            echo "  simulator  $name ($udid)"
        done <<< "$SIMS"
    fi
}

list_targets() {
    if [ -n "$DEVICE_LABEL" ]; then
        echo "  device     $DEVICE_LABEL"
    fi
    list_sims
}

# The runnable command for each target, so the message can be pasted as-is.
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
            if [ -n "$DEVICE_LABEL" ]; then
                echo "  make ios-screenshot TARGET=device"
            fi
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
    if [ -z "$DEVICE_LABEL" ]; then
        {
            echo "Error: no paired iOS device found."
            echo
            echo "Connect the device via USB, unlock it, and tap Trust."
            echo "Then confirm it is visible with: pymobiledevice3 usbmux list"
        } >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

OUT="${OUT:-/tmp/iphone-$(date +%Y%m%d-%H%M%S).png}"

if [ "$TARGET" = "device" ]; then
    pymobiledevice3 developer dvt screenshot "$OUT" --userspace
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
    echo "Saved: $OUT  (device: $DEVICE_LABEL)"
else
    echo "Saved: $OUT  (simulator: $(awk -F'\t' -v u="$SIM_UDID" '$1==u {print $2}' <<< "$SIMS"))"
fi
