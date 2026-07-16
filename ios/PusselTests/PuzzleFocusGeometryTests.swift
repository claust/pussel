import CoreGraphics
import XCTest

@testable import Pussel

final class PuzzleFocusGeometryTests: XCTestCase {
  private let centre = CGPoint(x: 0.5, y: 0.5)

  func testFocusRectIsCentredOnThePiece() {
    let rect = PuzzleFocusGeometry.focusRect(
      position: CGPoint(x: 0.25, y: 0.8), rows: 20, cols: 20)
    XCTAssertEqual(rect.midX, 0.25, accuracy: 0.0001)
    XCTAssertEqual(rect.midY, 0.8, accuracy: 0.0001)
  }

  func testFineGridFramesThreeCells() {
    // 3 of 30 columns is a tenth of the puzzle — well under the ceiling, so
    // the cell span is what decides the frame.
    let rect = PuzzleFocusGeometry.focusRect(position: centre, rows: 40, cols: 30)
    XCTAssertEqual(rect.width, 0.1, accuracy: 0.0001)
    XCTAssertEqual(rect.height, 0.075, accuracy: 0.0001)
  }

  func testCoarseGridIsCappedSoTheZoomStaysVisible() {
    // The regression this cap exists for: on a 4×3 puzzle, three cells is the
    // entire width, which resolves to fit — opening on a piece would look like
    // the tap did nothing.
    let rect = PuzzleFocusGeometry.focusRect(position: centre, rows: 4, cols: 3)
    XCTAssertEqual(rect.width, PuzzleFocusGeometry.maxFraction, accuracy: 0.0001)
    XCTAssertEqual(rect.height, PuzzleFocusGeometry.maxFraction, accuracy: 0.0001)
  }

  func testFocusRectNeverFramesMoreThanTheCeiling() {
    // Whatever the grid, both axes stay within the ceiling.
    for rows in 1...12 {
      for cols in 1...12 {
        let rect = PuzzleFocusGeometry.focusRect(position: centre, rows: rows, cols: cols)
        XCTAssertLessThanOrEqual(rect.width, PuzzleFocusGeometry.maxFraction + 0.0001)
        XCTAssertLessThanOrEqual(rect.height, PuzzleFocusGeometry.maxFraction + 0.0001)
      }
    }
  }

  func testEdgePieceKeepsFullSizeRectRatherThanBeingClamped() {
    // A rect running off the puzzle is intended: the scroll view pins the
    // offset back. Clamping here would zoom edge pieces further in than
    // middle ones.
    let rect = PuzzleFocusGeometry.focusRect(position: CGPoint(x: 0, y: 1), rows: 40, cols: 30)
    XCTAssertEqual(rect.width, 0.1, accuracy: 0.0001)
    XCTAssertLessThan(rect.minX, 0)
    XCTAssertGreaterThan(rect.maxY, 1)
  }

  func testDegenerateGridFallsBackToTheCeiling() {
    // A zero grid must not divide by zero into an infinite/NaN rect.
    let rect = PuzzleFocusGeometry.focusRect(position: centre, rows: 0, cols: 0)
    XCTAssertEqual(rect.width, PuzzleFocusGeometry.maxFraction, accuracy: 0.0001)
    XCTAssertEqual(rect.height, PuzzleFocusGeometry.maxFraction, accuracy: 0.0001)
  }
}
