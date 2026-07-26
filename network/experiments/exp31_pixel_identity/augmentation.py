"""Adapter exposing exp31's capture model under the acceptance probe's names.

``ncc_probe.py`` — the section 8 NCC-headroom gate — discovers the pipeline it
is measuring defensively: it looks for a module called ``augmentation`` /
``augment`` / ``aug`` / ``degradation`` / ``asymmetric``, a piece hook called
``augment_piece_rgba`` (RGBA preferred, so the probe can take the mask from
alpha) or ``augment_piece``, an overview hook called ``augment_overview`` /
``degrade_overview`` / ``augment_puzzle``, and a zero-argument-constructible
config class. exp31's real names are :mod:`capture`,
:func:`capture.augment_piece_capture`, :func:`capture.augment_overview_capture`
and :class:`capture.CaptureConfig`, so this module is the (thin) bridge. It
adds no behaviour: every hook forwards to :mod:`capture`.

Two things worth knowing before reading probe output through this adapter.

**1. The two hooks are called separately, so ARP cannot be joint.** In
training, ``capture.augment_view_pair`` fires ARP once with probability
``arp_p`` and then patches **exactly one** of the two views — the guarantee a
unit test enforces. The probe builds the piece view and the overview view in
separate calls, so this adapter instead patches each view independently with
probability ``arp_p / 2``. That reproduces the training path's *marginal*
per-view patch rate exactly, at the cost of a
``(arp_p / 2) ** 2`` = 6.25% (at the default ``arp_p=0.5``) chance of both
views being patched, which training never produces. Both-patched is the
*conservative* case for this measurement (two independent occlusions lower NCC
more than one), so a passing verdict is not an artifact of the deviation.

**2. The adapter does not export the name ``AugmentConfig``.** The probe's
config-name search tries ``AugmentConfig`` first, and exp31's config is a
*subclass* of exp26's — handing it the base class would silently strip every
exp31 field and the hooks would fail on ``config.capture``. The config is
therefore exported only as :class:`Exp31Config`.

The probe passes an RGBA piece already rotated into the overview's orientation
and does **not** pre-frame it for exp31 (unlike its exp30 branch, which calls
``frame_rgba`` first). That is correct: exp31 owns its own framing, including
the crop/bbox jitter, so the hook below runs the whole load-time pipeline and
the probe's scale/rotation/translation search absorbs the framing shift — the
same thing it has to do for exp26's scale and rotation jitter.
"""

from __future__ import annotations

import copy
import os
import random

from PIL import Image

from .capture import (
    CAPTURE_PRESETS,
    CaptureConfig,
    PatchSource,
    apply_arp,
    augment_overview_capture,
    augment_piece_capture,
)

# The probe constructs the config class zero-argument, and has no ablation
# flag of its own, so this is how an ablation gets measured through the gate:
#     EXP31_CAPTURE_PRESET=no_arp uv run python -m ...ncc_probe
# An unset or unknown value falls back to the ``full`` defaults.
PRESET_ENV_VAR = "EXP31_CAPTURE_PRESET"


def Exp31Config() -> CaptureConfig:  # noqa: N802 - the probe searches for a config *name*
    """Return the capture config the probe should measure.

    Named like a class because that is what ``ncc_probe``'s config-name search
    looks for; it only ever calls it with no arguments. Note the probe's search
    order tries ``AugmentConfig`` first, and exp31's config is a *subclass* of
    exp26's — handing over the base class would silently strip every exp31
    field — so this module deliberately does not export that name.

    Returns:
        A fresh :class:`CaptureConfig`, or a deep copy of the preset named by
        ``EXP31_CAPTURE_PRESET``.
    """
    name = os.environ.get(PRESET_ENV_VAR)
    if name and name in CAPTURE_PRESETS:
        print(f"exp31 adapter: using capture preset '{name}' from {PRESET_ENV_VAR}")
        return copy.deepcopy(CAPTURE_PRESETS[name])
    if name:
        print(f"exp31 adapter: WARNING unknown {PRESET_ENV_VAR}={name!r}; using defaults")
    return CaptureConfig()


__all__ = ["PRESET_ENV_VAR", "Exp31Config", "augment_overview", "augment_piece_rgba"]


def _arp_one_view(view: Image.Image, config: CaptureConfig) -> Image.Image:
    """Patch a single view with the adapter's halved per-view probability.

    Args:
        view: The RGB view to (maybe) patch.
        config: Active capture config.

    Returns:
        The view, patched or not.
    """
    if not config.capture or not config.arp:
        return view
    # apply_arp itself decides with ``arp_p``; halve it here so the marginal
    # per-view rate matches the training path's "fire once, pick one of two".
    solo = random.random() < 0.5
    if not solo:
        return view
    patched, _ = apply_arp(view, view, config, PatchSource())
    return patched


def augment_piece_rgba(piece_rgba: Image.Image, config: CaptureConfig | None = None) -> Image.Image:
    """Render one piece as an independent close-up capture, keeping alpha.

    The probe's preferred piece hook: returning RGBA lets it use the exact
    piece mask for the masked NCC instead of falling back to a luminance
    threshold.

    Args:
        piece_rgba: RGBA piece, already rotated into the overview's
            orientation by the caller.
        config: Capture config (defaults to :class:`CaptureConfig`).

    Returns:
        An RGBA image: exp31's piece view in RGB, with the post-augmentation
        piece mask as its alpha channel.
    """
    active = config if config is not None else CaptureConfig()
    rgb, mask = augment_piece_capture(piece_rgba, active)
    rgb = _arp_one_view(rgb, active)
    out = rgb.convert("RGBA")
    out.putalpha(mask)
    return out


def augment_overview(overview_rgb: Image.Image, config: CaptureConfig | None = None) -> Image.Image:
    """Render one overview as an independent wide capture of a glossy box.

    Args:
        overview_rgb: The RGB source puzzle image (the "box art").
        config: Capture config (defaults to :class:`CaptureConfig`).

    Returns:
        exp31's overview view, same size as the input.
    """
    active = config if config is not None else CaptureConfig()
    return _arp_one_view(augment_overview_capture(overview_rgb, active), active)
