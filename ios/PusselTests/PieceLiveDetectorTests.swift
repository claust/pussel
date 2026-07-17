import XCTest
import simd

@testable import Pussel

/// Tests for `PieceLiveDetector`'s pure geometry helpers — the band scoring
/// ported from the backend and the contour smoothing/resampling. The Vision
/// pipeline itself (subject lift + contours) is exercised on-device; these
/// cover the math that turns its contour into the published outline.
final class PieceLiveDetectorTests: XCTestCase {
  // MARK: - bandScore (port of piece_detector._band_score)

  func testBandScoreFullConfidenceInsideBand() {
    XCTAssertEqual(
      PieceLiveDetector.bandScore(
        0.05, hardLow: 0.005, fullLow: 0.01, fullHigh: 0.15, hardHigh: 0.35), 1.0)
  }

  func testBandScoreZeroAtOrBelowHardLow() {
    XCTAssertEqual(
      PieceLiveDetector.bandScore(
        0.005, hardLow: 0.005, fullLow: 0.01, fullHigh: 0.15, hardHigh: 0.35), 0.0)
    XCTAssertEqual(
      PieceLiveDetector.bandScore(
        0.001, hardLow: 0.005, fullLow: 0.01, fullHigh: 0.15, hardHigh: 0.35), 0.0)
  }

  func testBandScoreZeroAtOrAboveHardHigh() {
    XCTAssertEqual(
      PieceLiveDetector.bandScore(
        0.35, hardLow: 0.005, fullLow: 0.01, fullHigh: 0.15, hardHigh: 0.35), 0.0)
    XCTAssertEqual(
      PieceLiveDetector.bandScore(
        0.5, hardLow: 0.005, fullLow: 0.01, fullHigh: 0.15, hardHigh: 0.35), 0.0)
  }

  func testBandScoreTapersLinearlyBelowFullBand() {
    // Halfway between hardLow 0 and fullLow 1 → 0.5, matching the backend.
    XCTAssertEqual(
      PieceLiveDetector.bandScore(0.5, hardLow: 0.0, fullLow: 1.0, fullHigh: 2.0, hardHigh: 3.5),
      0.5,
      accuracy: 1e-9)
  }

  func testBandScoreTapersLinearlyAboveFullBand() {
    // Halfway between fullHigh 2.0 and hardHigh 3.5 → 0.5.
    XCTAssertEqual(
      PieceLiveDetector.bandScore(2.75, hardLow: 0.0, fullLow: 1.0, fullHigh: 2.0, hardHigh: 3.5),
      0.5,
      accuracy: 1e-9)
  }

  // MARK: - resampled

  func testResampledReturnsRequestedCount() {
    let square: [SIMD2<Double>] = [.init(0, 0), .init(1, 0), .init(1, 1), .init(0, 1)]
    XCTAssertEqual(PieceLiveDetector.resampled(square, count: 120).count, 120)
  }

  func testResampledPointsAreArcLengthEquidistant() {
    // A unit square has perimeter 4; 8 samples → one every 0.5 along the
    // outline, starting at the first vertex.
    let square: [SIMD2<Double>] = [.init(0, 0), .init(1, 0), .init(1, 1), .init(0, 1)]
    let result = PieceLiveDetector.resampled(square, count: 8)
    let expected: [SIMD2<Double>] = [
      .init(0, 0), .init(0.5, 0), .init(1, 0), .init(1, 0.5),
      .init(1, 1), .init(0.5, 1), .init(0, 1), .init(0, 0.5),
    ]
    for (point, want) in zip(result, expected) {
      XCTAssertEqual(point.x, want.x, accuracy: 1e-9)
      XCTAssertEqual(point.y, want.y, accuracy: 1e-9)
    }
  }

  func testResampledDegenerateContourReturnsInput() {
    let stacked: [SIMD2<Double>] = [.init(0.5, 0.5), .init(0.5, 0.5), .init(0.5, 0.5)]
    XCTAssertEqual(PieceLiveDetector.resampled(stacked, count: 10), stacked)
  }

  // MARK: - smoothed

  func testSmoothedPreservesCountAndConstantContour() {
    let constant = [SIMD2<Double>](repeating: .init(0.3, 0.7), count: 20)
    let result = PieceLiveDetector.smoothed(constant, window: 5)
    XCTAssertEqual(result.count, 20)
    for point in result {
      XCTAssertEqual(point.x, 0.3, accuracy: 1e-9)
      XCTAssertEqual(point.y, 0.7, accuracy: 1e-9)
    }
  }

  func testSmoothedPullsSpikeTowardNeighbors() {
    // A single outlier vertex on an otherwise flat run gets averaged down.
    var points = [SIMD2<Double>](repeating: .init(0.5, 0.5), count: 20)
    points[10] = .init(0.5, 0.9)
    let result = PieceLiveDetector.smoothed(points, window: 5)
    XCTAssertEqual(result[10].y, 0.58, accuracy: 1e-9)
    XCTAssertEqual(result[0].y, 0.5, accuracy: 1e-9)
  }

  func testSmoothedTinyContourPassesThrough() {
    let triangle: [SIMD2<Double>] = [.init(0, 0), .init(1, 0), .init(0, 1)]
    XCTAssertEqual(PieceLiveDetector.smoothed(triangle, window: 5), triangle)
  }
}
