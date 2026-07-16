import CoreGraphics

/// Pure sizing math for a solve-screen piece marker (`PuzzleOverlayView`'s
/// `PieceMarker`), factored out so it's unit-testable without SwiftUI.
enum PieceMarkerGeometry {
  /// The cell-relative multiple a piece's SMALLER image dimension is scaled
  /// to in the null-span fallback (see `size(span:imageSize:canvas:cellSize:rotation:)`).
  /// A jigsaw piece's tabs/blanks make it span more than one bare grid
  /// cell — real pieces overhang their cell by roughly this multiple along
  /// their narrower axis.
  static let fallbackMinCellSpan: CGFloat = 1.1

  /// Returns the marker's frame size, in canvas points, *before*
  /// `.rotationEffect` is applied.
  ///
  /// - Parameters:
  ///   - span: the backend-measured full-image-frame span, or `nil` when
  ///     unmeasured (CNN path / matcher failure).
  ///   - imageSize: size of the *displayed* piece image. In the null-span
  ///     fallback this should be the alpha-trimmed image (see
  ///     `ImageUtilities.croppedToAlphaBounds`), so the backend's
  ///     transparent margin doesn't inflate the marker.
  ///   - canvas: the overlay's full drawing size (the trimmed puzzle
  ///     image's rendered size).
  ///   - cellSize: `canvas` divided by the puzzle's (cols, rows) — one grid
  ///     cell's rendered size.
  ///   - rotation: the piece's predicted rotation in degrees (0/90/180/270).
  ///
  /// When `span` is present, the frame is `canvas × span` directly, in the
  /// displayed image's own (pre-rotation) axes — no cell-overhang factor
  /// and no rotation swap, since the span already describes the full
  /// displayed image frame and `.rotationEffect` handles orientation on
  /// top. When `span` is `nil`, falls back to grid-based sizing: scale
  /// `imageSize` uniformly so its smaller cell-relative dimension renders
  /// at `fallbackMinCellSpan` cells, then swap width/height for a
  /// 90°/270° rotation (cells are generally non-square, so the
  /// pre-rotation, canvas-axis frame must swap when the image's own
  /// width/height swap onto the canvas).
  static func size(
    span: PieceSpan?, imageSize: CGSize, canvas: CGSize, cellSize: CGSize, rotation: Int
  ) -> CGSize {
    if let span {
      return CGSize(width: canvas.width * span.width, height: canvas.height * span.height)
    }
    return fallbackSize(imageSize: imageSize, cellSize: cellSize, rotation: rotation)
  }

  private static func fallbackSize(imageSize: CGSize, cellSize: CGSize, rotation: Int) -> CGSize {
    guard imageSize.width > 0, imageSize.height > 0, cellSize.width > 0, cellSize.height > 0 else {
      return CGSize(
        width: cellSize.width * fallbackMinCellSpan, height: cellSize.height * fallbackMinCellSpan)
    }
    let imageWidthCells = imageSize.width / cellSize.width
    let imageHeightCells = imageSize.height / cellSize.height
    let scale = fallbackMinCellSpan / min(imageWidthCells, imageHeightCells)
    var width = imageSize.width * scale
    var height = imageSize.height * scale

    // A 90°/270° rotation swaps how the piece's own width/height map onto
    // the canvas axes, so the pre-rotation frame must swap too.
    let turns = ((rotation / 90) % 4 + 4) % 4
    if turns == 1 || turns == 3 {
      swap(&width, &height)
    }
    return CGSize(width: width, height: height)
  }
}
