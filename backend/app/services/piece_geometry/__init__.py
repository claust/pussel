"""Piece-geometry pipeline: photo -> contour -> corners -> edges -> fingerprint.

Ported from ``network/experiments/exp28_piece_geometry/`` (see that
experiment's ``HANDOFF.md`` for the algorithm decisions and thresholds this
package productionizes). Each submodule carries a provenance note pointing
at the exp28 source file it was ported from; keep algorithm changes in sync
with that source unless a deviation is required for production (e.g. no
scipy, unknown photo orientation) and documented locally.
"""
