import XCTest

@testable import Pussel

/// One (image size, piece count) → expected (rows, cols) case generated from
/// the Python reference implementation.
private struct GridVector {
  let width: Int
  let height: Int
  let pieces: Int
  let expectedRows: Int
  let expectedCols: Int
}

/// Parity tests against `calculate_grid_dimensions` in
/// shared/puzzle_shapes/puzzle_shapes/edge_grid.py:60-110.
final class GridEstimatorTests: XCTestCase {
  func testBelowFourPiecesAlwaysReturnsTwoByTwo() {
    for pieces in [0, 1, 2, 3] {
      let result = GridEstimator.estimate(pieceCount: pieces, imageWidth: 1234, imageHeight: 567)
      XCTAssertEqual(result.rows, 2, "pieces=\(pieces)")
      XCTAssertEqual(result.cols, 2, "pieces=\(pieces)")
    }
  }

  /// 114 vectors generated from the Python reference implementation.
  private static let vectors: [GridVector] = [
    GridVector(width: 1000, height: 1000, pieces: 4, expectedRows: 2, expectedCols: 2),
    GridVector(width: 1000, height: 1000, pieces: 6, expectedRows: 2, expectedCols: 2),
    GridVector(width: 1000, height: 1000, pieces: 9, expectedRows: 3, expectedCols: 3),
    GridVector(width: 1000, height: 1000, pieces: 12, expectedRows: 3, expectedCols: 3),
    GridVector(width: 1000, height: 1000, pieces: 16, expectedRows: 4, expectedCols: 4),
    GridVector(width: 1000, height: 1000, pieces: 20, expectedRows: 4, expectedCols: 4),
    GridVector(width: 1000, height: 1000, pieces: 24, expectedRows: 5, expectedCols: 5),
    GridVector(width: 1000, height: 1000, pieces: 25, expectedRows: 5, expectedCols: 5),
    GridVector(width: 1000, height: 1000, pieces: 30, expectedRows: 5, expectedCols: 5),
    GridVector(width: 1000, height: 1000, pieces: 35, expectedRows: 6, expectedCols: 6),
    GridVector(width: 1000, height: 1000, pieces: 48, expectedRows: 7, expectedCols: 7),
    GridVector(width: 1000, height: 1000, pieces: 54, expectedRows: 7, expectedCols: 7),
    GridVector(width: 1000, height: 1000, pieces: 100, expectedRows: 10, expectedCols: 10),
    GridVector(width: 1000, height: 1000, pieces: 204, expectedRows: 14, expectedCols: 14),
    GridVector(width: 1000, height: 1000, pieces: 300, expectedRows: 17, expectedCols: 17),
    GridVector(width: 1000, height: 1000, pieces: 500, expectedRows: 22, expectedCols: 22),
    GridVector(width: 1000, height: 1000, pieces: 1000, expectedRows: 32, expectedCols: 32),
    GridVector(width: 1000, height: 1000, pieces: 1500, expectedRows: 39, expectedCols: 39),
    GridVector(width: 1000, height: 1000, pieces: 2000, expectedRows: 45, expectedCols: 45),
    GridVector(width: 1200, height: 900, pieces: 4, expectedRows: 2, expectedCols: 2),
    GridVector(width: 1200, height: 900, pieces: 6, expectedRows: 2, expectedCols: 3),
    GridVector(width: 1200, height: 900, pieces: 9, expectedRows: 3, expectedCols: 4),
    GridVector(width: 1200, height: 900, pieces: 12, expectedRows: 3, expectedCols: 4),
    GridVector(width: 1200, height: 900, pieces: 16, expectedRows: 3, expectedCols: 4),
    GridVector(width: 1200, height: 900, pieces: 20, expectedRows: 4, expectedCols: 5),
    GridVector(width: 1200, height: 900, pieces: 24, expectedRows: 4, expectedCols: 6),
    GridVector(width: 1200, height: 900, pieces: 25, expectedRows: 4, expectedCols: 6),
    GridVector(width: 1200, height: 900, pieces: 30, expectedRows: 5, expectedCols: 6),
    GridVector(width: 1200, height: 900, pieces: 35, expectedRows: 5, expectedCols: 7),
    GridVector(width: 1200, height: 900, pieces: 48, expectedRows: 6, expectedCols: 8),
    GridVector(width: 1200, height: 900, pieces: 54, expectedRows: 6, expectedCols: 8),
    GridVector(width: 1200, height: 900, pieces: 100, expectedRows: 9, expectedCols: 12),
    GridVector(width: 1200, height: 900, pieces: 204, expectedRows: 12, expectedCols: 16),
    GridVector(width: 1200, height: 900, pieces: 300, expectedRows: 15, expectedCols: 20),
    GridVector(width: 1200, height: 900, pieces: 500, expectedRows: 19, expectedCols: 26),
    GridVector(width: 1200, height: 900, pieces: 1000, expectedRows: 27, expectedCols: 36),
    GridVector(width: 1200, height: 900, pieces: 1500, expectedRows: 33, expectedCols: 44),
    GridVector(width: 1200, height: 900, pieces: 2000, expectedRows: 39, expectedCols: 52),
    GridVector(width: 1500, height: 1000, pieces: 4, expectedRows: 2, expectedCols: 3),
    GridVector(width: 1500, height: 1000, pieces: 6, expectedRows: 2, expectedCols: 3),
    GridVector(width: 1500, height: 1000, pieces: 9, expectedRows: 2, expectedCols: 3),
    GridVector(width: 1500, height: 1000, pieces: 12, expectedRows: 3, expectedCols: 4),
    GridVector(width: 1500, height: 1000, pieces: 16, expectedRows: 3, expectedCols: 5),
    GridVector(width: 1500, height: 1000, pieces: 20, expectedRows: 4, expectedCols: 6),
    GridVector(width: 1500, height: 1000, pieces: 24, expectedRows: 4, expectedCols: 6),
    GridVector(width: 1500, height: 1000, pieces: 25, expectedRows: 4, expectedCols: 6),
    GridVector(width: 1500, height: 1000, pieces: 30, expectedRows: 4, expectedCols: 6),
    GridVector(width: 1500, height: 1000, pieces: 35, expectedRows: 5, expectedCols: 7),
    GridVector(width: 1500, height: 1000, pieces: 48, expectedRows: 6, expectedCols: 9),
    GridVector(width: 1500, height: 1000, pieces: 54, expectedRows: 6, expectedCols: 9),
    GridVector(width: 1500, height: 1000, pieces: 100, expectedRows: 8, expectedCols: 12),
    GridVector(width: 1500, height: 1000, pieces: 204, expectedRows: 12, expectedCols: 18),
    GridVector(width: 1500, height: 1000, pieces: 300, expectedRows: 14, expectedCols: 21),
    GridVector(width: 1500, height: 1000, pieces: 500, expectedRows: 18, expectedCols: 27),
    GridVector(width: 1500, height: 1000, pieces: 1000, expectedRows: 26, expectedCols: 39),
    GridVector(width: 1500, height: 1000, pieces: 1500, expectedRows: 32, expectedCols: 48),
    GridVector(width: 1500, height: 1000, pieces: 2000, expectedRows: 36, expectedCols: 54),
    GridVector(width: 900, height: 1200, pieces: 4, expectedRows: 2, expectedCols: 2),
    GridVector(width: 900, height: 1200, pieces: 6, expectedRows: 3, expectedCols: 2),
    GridVector(width: 900, height: 1200, pieces: 9, expectedRows: 4, expectedCols: 3),
    GridVector(width: 900, height: 1200, pieces: 12, expectedRows: 4, expectedCols: 3),
    GridVector(width: 900, height: 1200, pieces: 16, expectedRows: 4, expectedCols: 3),
    GridVector(width: 900, height: 1200, pieces: 20, expectedRows: 5, expectedCols: 4),
    GridVector(width: 900, height: 1200, pieces: 24, expectedRows: 6, expectedCols: 4),
    GridVector(width: 900, height: 1200, pieces: 25, expectedRows: 6, expectedCols: 4),
    GridVector(width: 900, height: 1200, pieces: 30, expectedRows: 6, expectedCols: 5),
    GridVector(width: 900, height: 1200, pieces: 35, expectedRows: 7, expectedCols: 5),
    GridVector(width: 900, height: 1200, pieces: 48, expectedRows: 8, expectedCols: 6),
    GridVector(width: 900, height: 1200, pieces: 54, expectedRows: 8, expectedCols: 6),
    GridVector(width: 900, height: 1200, pieces: 100, expectedRows: 12, expectedCols: 9),
    GridVector(width: 900, height: 1200, pieces: 204, expectedRows: 16, expectedCols: 12),
    GridVector(width: 900, height: 1200, pieces: 300, expectedRows: 20, expectedCols: 15),
    GridVector(width: 900, height: 1200, pieces: 500, expectedRows: 26, expectedCols: 19),
    GridVector(width: 900, height: 1200, pieces: 1000, expectedRows: 36, expectedCols: 27),
    GridVector(width: 900, height: 1200, pieces: 1500, expectedRows: 44, expectedCols: 33),
    GridVector(width: 900, height: 1200, pieces: 2000, expectedRows: 52, expectedCols: 39),
    GridVector(width: 1000, height: 1500, pieces: 4, expectedRows: 3, expectedCols: 2),
    GridVector(width: 1000, height: 1500, pieces: 6, expectedRows: 3, expectedCols: 2),
    GridVector(width: 1000, height: 1500, pieces: 9, expectedRows: 3, expectedCols: 2),
    GridVector(width: 1000, height: 1500, pieces: 12, expectedRows: 4, expectedCols: 3),
    GridVector(width: 1000, height: 1500, pieces: 16, expectedRows: 5, expectedCols: 3),
    GridVector(width: 1000, height: 1500, pieces: 20, expectedRows: 6, expectedCols: 4),
    GridVector(width: 1000, height: 1500, pieces: 24, expectedRows: 6, expectedCols: 4),
    GridVector(width: 1000, height: 1500, pieces: 25, expectedRows: 6, expectedCols: 4),
    GridVector(width: 1000, height: 1500, pieces: 30, expectedRows: 6, expectedCols: 4),
    GridVector(width: 1000, height: 1500, pieces: 35, expectedRows: 7, expectedCols: 5),
    GridVector(width: 1000, height: 1500, pieces: 48, expectedRows: 9, expectedCols: 6),
    GridVector(width: 1000, height: 1500, pieces: 54, expectedRows: 9, expectedCols: 6),
    GridVector(width: 1000, height: 1500, pieces: 100, expectedRows: 12, expectedCols: 8),
    GridVector(width: 1000, height: 1500, pieces: 204, expectedRows: 18, expectedCols: 12),
    GridVector(width: 1000, height: 1500, pieces: 300, expectedRows: 21, expectedCols: 14),
    GridVector(width: 1000, height: 1500, pieces: 500, expectedRows: 27, expectedCols: 18),
    GridVector(width: 1000, height: 1500, pieces: 1000, expectedRows: 39, expectedCols: 26),
    GridVector(width: 1000, height: 1500, pieces: 1500, expectedRows: 48, expectedCols: 32),
    GridVector(width: 1000, height: 1500, pieces: 2000, expectedRows: 54, expectedCols: 36),
    GridVector(width: 1600, height: 900, pieces: 4, expectedRows: 2, expectedCols: 3),
    GridVector(width: 1600, height: 900, pieces: 6, expectedRows: 2, expectedCols: 3),
    GridVector(width: 1600, height: 900, pieces: 9, expectedRows: 2, expectedCols: 4),
    GridVector(width: 1600, height: 900, pieces: 12, expectedRows: 3, expectedCols: 5),
    GridVector(width: 1600, height: 900, pieces: 16, expectedRows: 3, expectedCols: 5),
    GridVector(width: 1600, height: 900, pieces: 20, expectedRows: 3, expectedCols: 6),
    GridVector(width: 1600, height: 900, pieces: 24, expectedRows: 4, expectedCols: 7),
    GridVector(width: 1600, height: 900, pieces: 25, expectedRows: 4, expectedCols: 7),
    GridVector(width: 1600, height: 900, pieces: 30, expectedRows: 4, expectedCols: 7),
    GridVector(width: 1600, height: 900, pieces: 35, expectedRows: 4, expectedCols: 7),
    GridVector(width: 1600, height: 900, pieces: 48, expectedRows: 5, expectedCols: 9),
    GridVector(width: 1600, height: 900, pieces: 54, expectedRows: 5, expectedCols: 9),
    GridVector(width: 1600, height: 900, pieces: 100, expectedRows: 8, expectedCols: 14),
    GridVector(width: 1600, height: 900, pieces: 204, expectedRows: 11, expectedCols: 19),
    GridVector(width: 1600, height: 900, pieces: 300, expectedRows: 13, expectedCols: 23),
    GridVector(width: 1600, height: 900, pieces: 500, expectedRows: 17, expectedCols: 30),
    GridVector(width: 1600, height: 900, pieces: 1000, expectedRows: 24, expectedCols: 42),
    GridVector(width: 1600, height: 900, pieces: 1500, expectedRows: 29, expectedCols: 52),
    GridVector(width: 1600, height: 900, pieces: 2000, expectedRows: 34, expectedCols: 60),
  ]

  func testParityAgainstPythonReference() {
    for vector in Self.vectors {
      let result = GridEstimator.estimate(
        pieceCount: vector.pieces, imageWidth: CGFloat(vector.width),
        imageHeight: CGFloat(vector.height))
      let label = "\(vector.width)x\(vector.height) pieces=\(vector.pieces)"
      XCTAssertEqual(result.rows, vector.expectedRows, "rows mismatch for \(label): got \(result)")
      XCTAssertEqual(result.cols, vector.expectedCols, "cols mismatch for \(label): got \(result)")
    }
  }
}
