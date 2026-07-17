import CoreGraphics

/// Pure sizing math for the piece queue's thumbnails (`PieceQueueView`'s
/// `QueueTile`), factored out so it's unit-testable without SwiftUI.
///
/// The queue draws every piece at one shared scale, so two tiles side by side
/// show which piece is actually bigger. Sizing each thumbnail to its own tile
/// instead would say nothing about the piece: it would only report how close
/// the camera happened to be when that piece was shot.
///
/// The measurement it scales by is `PieceSpan`, which is normalized to the
/// puzzle's width and height *separately*. Those are different lengths on a
/// non-square puzzle, so a span's two components aren't comparable as they
/// stand — `0.1` wide and `0.1` tall is not a square piece. Everything here
/// works in **puzzle-width units**: x is already in them, and y converts by
/// the puzzle's aspect. That makes extents comparable both between the axes
/// and between pieces, since every piece in a session shares one puzzle.
enum PieceThumbnailGeometry {
  /// A piece's measured frame in puzzle-width units — the full image frame
  /// including the backend's transparent margin, in the image's own
  /// (pre-rotation) axes, matching what `PieceSpan` describes.
  static func extent(span: PieceSpan, puzzleAspect: CGFloat) -> CGSize {
    guard puzzleAspect > 0 else { return .zero }
    return CGSize(width: CGFloat(span.width), height: CGFloat(span.height) / puzzleAspect)
  }

  /// The largest single dimension any measured piece reaches, in puzzle-width
  /// units — the divisor that maps measurements onto tiles. Returns nil when
  /// no piece was measured, leaving the caller to fall back to per-tile
  /// sizing.
  ///
  /// The *largest* is what sets the scale because tiles are square and clip:
  /// pinning the biggest piece's longest axis to the tile side is exactly the
  /// point where every piece still fits whole, at any quarter turn, with the
  /// grid no smaller than it has to be.
  static func maxExtent(spans: [PieceSpan?], puzzleAspect: CGFloat) -> CGFloat? {
    let extents = spans.compactMap { span -> CGFloat? in
      guard let span else { return nil }
      let size = extent(span: span, puzzleAspect: puzzleAspect)
      let longest = max(size.width, size.height)
      return longest > 0 ? longest : nil
    }
    return extents.max()
  }

  /// The frame to draw a thumbnail in, in points, *before* `.rotationEffect`
  /// is applied — nil for an unmeasured piece, or when there's nothing to
  /// scale against.
  ///
  /// Not swapped for a quarter turn, unlike `PieceMarkerGeometry.size`: that
  /// one lands its frame on a grid of non-square cells, whereas this one is
  /// centred in a square tile, where a rotated frame occupies the same
  /// bounds either way.
  static func size(
    span: PieceSpan?, maxExtent: CGFloat?, puzzleAspect: CGFloat, tileSide: CGFloat
  ) -> CGSize? {
    guard let span, let maxExtent, maxExtent > 0, tileSide > 0 else { return nil }
    let size = extent(span: span, puzzleAspect: puzzleAspect)
    guard size.width > 0, size.height > 0 else { return nil }
    let scale = tileSide / maxExtent
    return CGSize(width: size.width * scale, height: size.height * scale)
  }
}
