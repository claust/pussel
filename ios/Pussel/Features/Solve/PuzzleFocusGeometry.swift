import CoreGraphics

/// Pure framing math for the zoom viewer (`PuzzleZoomView`), factored out so
/// it's unit-testable without SwiftUI — the same treatment, and for the same
/// reason, as `PieceMarkerGeometry`.
enum PuzzleFocusGeometry {
  /// How much of the puzzle to frame around a focused piece, in grid cells.
  /// Wide enough to read the piece against its surroundings — which is the
  /// point of looking — rather than filling the screen with the piece alone.
  static let cellSpan: CGFloat = 3

  /// Ceiling on the fraction of the puzzle a focused piece may frame, on
  /// either axis.
  ///
  /// `cellSpan` alone measures nothing on a coarse grid: three cells of a
  /// 12-piece puzzle is the whole board, so opening on a piece would land at
  /// fit and the tap would look like it did nothing. Framing at most this much
  /// keeps the jump visible (~2.5× in) whatever the grid, while on any puzzle
  /// fine enough for three cells to be the smaller number, that span wins.
  static let maxFraction: CGFloat = 0.4

  /// Returns the region to open on, in the puzzle's normalized coordinates:
  /// a `cellSpan`-cell box centred on `position`, never wider or taller than
  /// `maxFraction` of the puzzle.
  ///
  /// Deliberately not clamped to the puzzle's bounds — a piece predicted near
  /// an edge yields a rect that runs off it, and the scroll view pins the
  /// resulting offset back to its content. Clamping here instead would shrink
  /// the box and zoom edge pieces in further than middle ones.
  ///
  /// - Parameters:
  ///   - position: the piece's centre, normalized to the puzzle.
  ///   - rows: the puzzle's estimated grid rows.
  ///   - cols: the puzzle's estimated grid columns.
  static func focusRect(position: CGPoint, rows: Int, cols: Int) -> CGRect {
    // A non-positive grid would divide by zero; fall back to the ceiling,
    // which is a sane frame for any puzzle.
    let width = cols > 0 ? min(cellSpan / CGFloat(cols), maxFraction) : maxFraction
    let height = rows > 0 ? min(cellSpan / CGFloat(rows), maxFraction) : maxFraction
    return CGRect(
      x: position.x - width / 2,
      y: position.y - height / 2,
      width: width,
      height: height)
  }
}
