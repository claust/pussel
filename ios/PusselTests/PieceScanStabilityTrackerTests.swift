import XCTest

@testable import Pussel

/// Unit tests for `PieceScanStabilityTracker`. All calls use injected `Date`
/// values so no wall clock is involved — tests are deterministic regardless of
/// host speed.
final class PieceScanStabilityTrackerTests: XCTestCase {

  // MARK: - Fixtures

  /// A tightly-packed square polygon covering most of [0,1]² — stable by design.
  private let squarePolygon: [NormalizedPoint] = [
    NormalizedPoint(x: 0.1, y: 0.1),
    NormalizedPoint(x: 0.9, y: 0.1),
    NormalizedPoint(x: 0.9, y: 0.9),
    NormalizedPoint(x: 0.1, y: 0.9),
  ]

  /// A bbox-shifted square with IoU ≈ 0.62 against squarePolygon — below the
  /// default 0.8 threshold. Manually: bbox A=[0.1,0.9]², bbox B=[0.2,1.0]²;
  /// intersection=[0.2,0.9]²=0.49, union=0.64+0.64-0.49=0.79, IoU≈0.62.
  private let shiftedPolygon: [NormalizedPoint] = [
    NormalizedPoint(x: 0.2, y: 0.2),
    NormalizedPoint(x: 1.0, y: 0.2),
    NormalizedPoint(x: 1.0, y: 1.0),
    NormalizedPoint(x: 0.2, y: 1.0),
  ]

  /// A polygon nearly identical to squarePolygon — IoU very close to 1.0.
  private let nearlyIdenticalPolygon: [NormalizedPoint] = [
    NormalizedPoint(x: 0.10, y: 0.10),
    NormalizedPoint(x: 0.91, y: 0.10),
    NormalizedPoint(x: 0.91, y: 0.91),
    NormalizedPoint(x: 0.10, y: 0.91),
  ]

  // Shorthand: a lockable state using the stable square polygon.
  private func lockableSquare(confidence: Double = 0.9) -> PiecePreviewState {
    .lockable(polygon: squarePolygon, confidence: confidence)
  }

  // Shorthand: a lockable state using the nearly-identical polygon.
  private func lockableNear(confidence: Double = 0.9) -> PiecePreviewState {
    .lockable(polygon: nearlyIdenticalPolygon, confidence: confidence)
  }

  // MARK: - Happy path: streak fires after minDuration + minSamples

  func testFiresAfterMinDurationAndMinSamples() {
    // minDuration=1.0, minSamples=3: feed 3 stable frames spanning 1.0 s.
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
    // Third sample at exactly t0+1.0 — boundary should fire.
    XCTAssertTrue(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.0)))
  }

  // MARK: - Jumped bbox restarts the streak

  func testJumpedBboxRestartsStreak() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    // Build a near-complete streak (2 of 3 needed samples, 0.9 s elapsed).
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.9)))
    // A frame with IoU ≈ 0.62 (< 0.8) breaks the streak.
    XCTAssertFalse(
      tracker.ingest(
        .lockable(polygon: shiftedPolygon, confidence: 0.9), at: t0.addingTimeInterval(0.95)))
    // After the jump, 2 more stable frames within 0.5 s should NOT fire
    // (new streak only has 2 samples and < 1.0 s elapsed).
    XCTAssertFalse(
      tracker.ingest(
        .lockable(polygon: shiftedPolygon, confidence: 0.9), at: t0.addingTimeInterval(1.2)))
    XCTAssertFalse(
      tracker.ingest(
        .lockable(polygon: shiftedPolygon, confidence: 0.9), at: t0.addingTimeInterval(1.3)))
  }

  // MARK: - .detected resets the streak

  func testDetectedResetsStreak() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
    // A single detected (quality gate not met) breaks the streak.
    XCTAssertFalse(
      tracker.ingest(
        .detected(polygon: squarePolygon, confidence: 0.6), at: t0.addingTimeInterval(0.6)))
    // New streak starts at t0+0.7. Two samples at 0.8 s elapsed is not enough.
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0.addingTimeInterval(0.7)))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.5)))
    // Third sample at t0+0.7+1.0 = t0+1.7 — fires because the new streak now
    // spans exactly 1.0 s and has 3 samples. If the streak had NOT been reset,
    // the very first 3-sample+1.0 s window would have fired much earlier.
    XCTAssertTrue(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.7)))
  }

  // MARK: - .none resets the streak

  func testNoneResetsStreak() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
    XCTAssertFalse(tracker.ingest(.none, at: t0.addingTimeInterval(0.6)))
    // New streak from t0+0.7: 2 samples at t0+1.5 is not enough.
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0.addingTimeInterval(0.7)))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.5)))
    // Third sample completes the new streak (3 samples, 1.0 s elapsed since t0+0.7).
    XCTAssertTrue(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.7)))
  }

  // MARK: - Latch prevents double-fire until reset()

  func testLatchPreventsdoubleFire() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    // Fire once.
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
    XCTAssertTrue(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.0)))

    // Additional lockable frames must NOT fire again.
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.1)))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(5.0)))

    // After reset() the tracker fires again on the next completed streak.
    tracker.reset()
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
    XCTAssertTrue(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.0)))
  }

  // MARK: - Degenerate polygon resets the streak

  func testDegeneratePolygonLessThanThreePointsResetsStreak() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    // A two-point "polygon" collapses to a line — zero-area bbox.
    let twoPoints: [NormalizedPoint] = [
      NormalizedPoint(x: 0.1, y: 0.1),
      NormalizedPoint(x: 0.9, y: 0.1),
    ]
    XCTAssertFalse(
      tracker.ingest(
        .lockable(polygon: twoPoints, confidence: 0.9), at: t0.addingTimeInterval(0.5)))
    // Next stable lockable frame restarts a fresh streak.
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0.addingTimeInterval(0.6)))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.3)))
    // Only 2 samples in the streak (0.6 and 1.3), elapsed = 0.7 s — should NOT fire.
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(1.5)))
  }

  func testZeroAreaBboxPolygonResetsStreak() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    // All points collinear → zero-height bbox.
    let collinear: [NormalizedPoint] = [
      NormalizedPoint(x: 0.1, y: 0.5),
      NormalizedPoint(x: 0.5, y: 0.5),
      NormalizedPoint(x: 0.9, y: 0.5),
    ]
    XCTAssertFalse(
      tracker.ingest(
        .lockable(polygon: collinear, confidence: 0.9), at: t0.addingTimeInterval(0.5)))
    // Streak was reset; a subsequent lockable frame begins a new one.
    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0.addingTimeInterval(0.6)))
  }

  // MARK: - Boundary: exactly at minDuration / minSamples

  func testBoundaryExactlyAtMinDurationAndMinSamples() {
    // minSamples=2, minDuration=0.5 — fire on the second frame at t=0.5.
    var tracker = PieceScanStabilityTracker(minDuration: 0.5, minSamples: 2, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertTrue(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
  }

  func testBoundaryOneFrameShortDoesNotFire() {
    var tracker = PieceScanStabilityTracker(minDuration: 1.0, minSamples: 3, minIoU: 0.8)
    let t0 = Date(timeIntervalSince1970: 0)

    XCTAssertFalse(tracker.ingest(lockableSquare(), at: t0))
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.5)))
    // Two samples, 0.99 s elapsed — short of both thresholds (needs 3 samples AND 1.0 s).
    XCTAssertFalse(tracker.ingest(lockableNear(), at: t0.addingTimeInterval(0.99)))
  }
}
