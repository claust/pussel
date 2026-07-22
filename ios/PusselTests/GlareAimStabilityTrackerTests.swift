import Foundation
import XCTest

@testable import Pussel

final class GlareAimStabilityTrackerTests: XCTestCase {
  private let start = Date(timeIntervalSinceReferenceDate: 1000)

  private func at(_ seconds: TimeInterval) -> Date {
    start.addingTimeInterval(seconds)
  }

  func testFiresAfterSustainedOnTargetDwell() {
    var tracker = GlareAimStabilityTracker()
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0)))
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0.2)))
    XCTAssertTrue(tracker.ingest(onTarget: true, at: at(0.45)))
  }

  func testDwellNeedsElapsedTimeNotJustSamples() {
    var tracker = GlareAimStabilityTracker()
    // A burst of samples with no time passing is not a steady hand.
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0)))
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0)))
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0.1)))
  }

  func testOffTargetBreaksTheStreak() {
    var tracker = GlareAimStabilityTracker()
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0)))
    XCTAssertFalse(tracker.ingest(onTarget: false, at: at(0.2)))
    // The streak restarts here; 0.3 s later is not yet a full dwell.
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0.4)))
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(0.7)))
    XCTAssertTrue(tracker.ingest(onTarget: true, at: at(0.9)))
  }

  func testLatchesUntilReset() {
    var tracker = GlareAimStabilityTracker()
    _ = tracker.ingest(onTarget: true, at: at(0))
    XCTAssertTrue(tracker.ingest(onTarget: true, at: at(0.5)))
    // Still on target — must not fire again for the same step.
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(1.0)))
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(2.0)))
    tracker.reset()
    XCTAssertFalse(tracker.ingest(onTarget: true, at: at(3.0)))
    XCTAssertTrue(tracker.ingest(onTarget: true, at: at(3.5)))
  }
}
