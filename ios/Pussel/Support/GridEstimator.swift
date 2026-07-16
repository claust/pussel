import CoreGraphics
import Foundation

/// Estimates a puzzle's (rows, cols) grid from a target piece count and the
/// image's aspect ratio. Exact port of `calculate_grid_dimensions` in
/// shared/puzzle_shapes/puzzle_shapes/edge_grid.py:60-110 — keep both in sync.
enum GridEstimator {
  /// Returns the estimated (rows, cols) grid for `pieceCount` pieces spread
  /// over an image of `imageWidth` × `imageHeight`.
  ///
  /// Mirrors the Python reference candidate-by-candidate: `Int(...)` on a
  /// positive `Double` truncates toward zero just like Python's `int()`, and
  /// candidates are scored in the same (rows-floor, rows-floor+1) ×
  /// (cols-floor, cols-floor+1) order, keeping the *first* maximal-score
  /// candidate (Python's `max()` returns the first maximum) by only
  /// replacing the best on a *strictly greater* score.
  static func estimate(pieceCount: Int, imageWidth: CGFloat, imageHeight: CGFloat) -> (
    rows: Int, cols: Int
  ) {
    guard pieceCount >= 4 else { return (2, 2) }

    let aspectRatio = Double(imageWidth) / Double(imageHeight)
    let targetPieces = Double(pieceCount)
    let colsFloat = (targetPieces * aspectRatio).squareRoot()
    let rowsFloat = (targetPieces / aspectRatio).squareRoot()

    let rowChoices = [max(2, Int(rowsFloat)), max(2, Int(rowsFloat) + 1)]
    let colChoices = [max(2, Int(colsFloat)), max(2, Int(colsFloat) + 1)]

    var bestRows = rowChoices[0]
    var bestCols = colChoices[0]
    var bestScore = -Double.infinity

    for rows in rowChoices {
      for cols in colChoices {
        let count = rows * cols
        let pieceWidth = Double(imageWidth) / Double(cols)
        let pieceHeight = Double(imageHeight) / Double(rows)
        let pieceAspectRatio = pieceWidth / pieceHeight
        let squareness = min(pieceAspectRatio, 1 / pieceAspectRatio)
        let countDiff = abs(Double(count) - targetPieces) / targetPieces
        let score = squareness - countDiff * 0.5
        if score > bestScore {
          bestScore = score
          bestRows = rows
          bestCols = cols
        }
      }
    }

    return (bestRows, bestCols)
  }
}
