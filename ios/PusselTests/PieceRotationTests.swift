import XCTest

@testable import Pussel

/// `PieceResponse.uprightRotationDegrees` — the correction that turns a piece
/// photo to sit the way the puzzle does. Shared by the piece grid
/// (`PieceQueueView`) and the scanner's gallery strip (`PieceScanView`), so
/// the two can't disagree about which way a piece faces.
final class PieceRotationTests: XCTestCase {

  private func response(rotation: Int) -> PieceResponse {
    PieceResponse(
      position: NormalizedPoint(x: 0.5, y: 0.5),
      positionConfidence: 0.9,
      rotation: rotation,
      rotationConfidence: 0.9,
      cleanedImage: nil,
      pieceSpan: nil,
      gridRow: nil,
      gridCol: nil,
      snappedPosition: nil
    )
  }

  /// The correction undoes the reported rotation, so the two cancel out.
  func testUndoesTheReportedRotation() {
    XCTAssertEqual(response(rotation: 0).uprightRotationDegrees, 0)
    XCTAssertEqual(response(rotation: 90).uprightRotationDegrees, -90)
    XCTAssertEqual(response(rotation: 270).uprightRotationDegrees, 90)
  }

  /// A half turn stays positive: 180 and -180 land in the same place, and the
  /// (-180, 180] normalization keeps the upper bound.
  func testHalfTurnKeepsThePositiveForm() {
    XCTAssertEqual(response(rotation: 180).uprightRotationDegrees, 180)
  }

  /// Every correction is the short way round, so an animating view never
  /// spins three quarter turns to reach a place one turn away.
  func testTakesTheShortWayRound() {
    for rotation in [0, 90, 180, 270] {
      let degrees = response(rotation: rotation).uprightRotationDegrees
      XCTAssertGreaterThan(degrees, -180)
      XCTAssertLessThanOrEqual(degrees, 180)
    }
  }

  /// Rotations that arrive already wrapped past a full turn are equivalent to
  /// their in-range form rather than producing a second, larger correction.
  func testFullTurnsAreEquivalentToTheirInRangeForm() {
    XCTAssertEqual(response(rotation: 360).uprightRotationDegrees, 0)
    XCTAssertEqual(response(rotation: 450).uprightRotationDegrees, -90)
  }
}
