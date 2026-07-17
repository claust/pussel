import XCTest

@testable import Pussel

final class PieceThumbnailGeometryTests: XCTestCase {
  /// A 2:1 landscape puzzle — deliberately non-square, since that's what
  /// makes a span's two axes mean different lengths.
  private let aspect: CGFloat = 2
  private let tileSide: CGFloat = 100

  // MARK: extent

  func testExtentConvertsHeightByAspectAndLeavesWidthAlone() {
    let size = PieceThumbnailGeometry.extent(
      span: PieceSpan(width: 0.2, height: 0.2), puzzleAspect: aspect)
    // Equal span components are NOT a square piece: on a 2:1 puzzle, 0.2 of
    // the height is half as long as 0.2 of the width.
    XCTAssertEqual(size.width, 0.2, accuracy: 1e-9)
    XCTAssertEqual(size.height, 0.1, accuracy: 1e-9)
  }

  func testSquarePuzzleLeavesBothAxesAlone() {
    let size = PieceThumbnailGeometry.extent(
      span: PieceSpan(width: 0.2, height: 0.3), puzzleAspect: 1)
    XCTAssertEqual(size.width, 0.2, accuracy: 1e-9)
    XCTAssertEqual(size.height, 0.3, accuracy: 1e-9)
  }

  func testNonPositiveAspectIsRejected() {
    XCTAssertEqual(
      PieceThumbnailGeometry.extent(span: PieceSpan(width: 0.2, height: 0.2), puzzleAspect: 0),
      .zero)
  }

  // MARK: maxExtent

  func testMaxExtentTakesLongestAxisAcrossMeasuredPieces() {
    let spans: [PieceSpan?] = [
      PieceSpan(width: 0.2, height: 0.2),  // extent 0.2 x 0.1 -> 0.2
      PieceSpan(width: 0.1, height: 0.6),  // extent 0.1 x 0.3 -> 0.3
    ]
    XCTAssertEqual(
      PieceThumbnailGeometry.maxExtent(spans: spans, puzzleAspect: aspect) ?? 0, 0.3,
      accuracy: 1e-9)
  }

  func testMaxExtentSkipsUnmeasuredPieces() {
    let spans: [PieceSpan?] = [nil, PieceSpan(width: 0.25, height: 0.2), nil]
    XCTAssertEqual(
      PieceThumbnailGeometry.maxExtent(spans: spans, puzzleAspect: aspect) ?? 0, 0.25,
      accuracy: 1e-9)
  }

  func testMaxExtentIsNilWhenNothingIsMeasured() {
    XCTAssertNil(PieceThumbnailGeometry.maxExtent(spans: [nil, nil], puzzleAspect: aspect))
    XCTAssertNil(PieceThumbnailGeometry.maxExtent(spans: [], puzzleAspect: aspect))
  }

  func testMaxExtentIgnoresDegenerateSpans() {
    let spans: [PieceSpan?] = [PieceSpan(width: 0, height: 0), PieceSpan(width: 0.25, height: 0.2)]
    XCTAssertEqual(
      PieceThumbnailGeometry.maxExtent(spans: spans, puzzleAspect: aspect) ?? 0, 0.25,
      accuracy: 1e-9)
  }

  // MARK: size

  func testBiggestPieceFillsTheTileAndSmallerOnesScaleWithIt() {
    let big = PieceSpan(width: 0.1, height: 0.6)  // extent 0.1 x 0.3
    let small = PieceSpan(width: 0.05, height: 0.3)  // extent 0.05 x 0.15 — half of big
    let maxExtent = PieceThumbnailGeometry.maxExtent(spans: [big, small], puzzleAspect: aspect)

    let bigSize = PieceThumbnailGeometry.size(
      span: big, maxExtent: maxExtent, puzzleAspect: aspect, tileSide: tileSide)
    let smallSize = PieceThumbnailGeometry.size(
      span: small, maxExtent: maxExtent, puzzleAspect: aspect, tileSide: tileSide)

    // The piece that set the scale reaches the tile edge on its longest axis,
    // and nothing overflows the square.
    XCTAssertEqual(bigSize?.height ?? 0, tileSide, accuracy: 1e-6)
    XCTAssertEqual(bigSize?.width ?? 0, tileSide / 3, accuracy: 1e-6)
    // Half the piece draws at half the size — the whole point of the shared scale.
    XCTAssertEqual(smallSize?.width ?? 0, (bigSize?.width ?? 0) / 2, accuracy: 1e-6)
    XCTAssertEqual(smallSize?.height ?? 0, (bigSize?.height ?? 0) / 2, accuracy: 1e-6)
  }

  func testNothingOverflowsTheSquareTile() {
    let spans = [
      PieceSpan(width: 0.3, height: 0.1), PieceSpan(width: 0.05, height: 0.5),
      PieceSpan(width: 0.2, height: 0.4),
    ]
    let maxExtent = PieceThumbnailGeometry.maxExtent(spans: spans, puzzleAspect: aspect)
    for span in spans {
      let size = PieceThumbnailGeometry.size(
        span: span, maxExtent: maxExtent, puzzleAspect: aspect, tileSide: tileSide)
      // A quarter turn swaps the frame inside the square, so BOTH axes have
      // to clear the tile side, not just the one that happens to be longer.
      XCTAssertLessThanOrEqual(size?.width ?? .infinity, tileSide + 1e-6)
      XCTAssertLessThanOrEqual(size?.height ?? .infinity, tileSide + 1e-6)
    }
  }

  func testUnmeasuredPieceHasNoSize() {
    XCTAssertNil(
      PieceThumbnailGeometry.size(
        span: nil, maxExtent: 0.3, puzzleAspect: aspect, tileSide: tileSide))
  }

  func testNoScaleToMeasureAgainstHasNoSize() {
    let span = PieceSpan(width: 0.1, height: 0.6)
    XCTAssertNil(
      PieceThumbnailGeometry.size(
        span: span, maxExtent: nil, puzzleAspect: aspect, tileSide: tileSide))
    XCTAssertNil(
      PieceThumbnailGeometry.size(
        span: span, maxExtent: 0, puzzleAspect: aspect, tileSide: tileSide))
  }

  func testZeroTileHasNoSize() {
    XCTAssertNil(
      PieceThumbnailGeometry.size(
        span: PieceSpan(width: 0.1, height: 0.6), maxExtent: 0.3, puzzleAspect: aspect,
        tileSide: 0))
  }
}
