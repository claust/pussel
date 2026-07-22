import XCTest

@testable import Pussel

final class BarcodeStabilityTrackerTests: XCTestCase {
  private let ean = "4005556050093"

  func testFiresExactlyOnceAtRequiredConsecutiveHits() {
    var tracker = BarcodeStabilityTracker(requiredConsecutiveHits: 3)
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertEqual(tracker.ingest(ean), ean)
    // Latched: the same payload continuing does not refire.
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertNil(tracker.ingest(ean))
  }

  func testDifferentPayloadResetsStreak() {
    var tracker = BarcodeStabilityTracker(requiredConsecutiveHits: 3)
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertNil(tracker.ingest(ean))
    // A different code restarts counting — no fire on its first two frames.
    XCTAssertNil(tracker.ingest("4005555006220"))
    XCTAssertNil(tracker.ingest("4005555006220"))
    XCTAssertEqual(tracker.ingest("4005555006220"), "4005555006220")
  }

  func testNilBreaksStreakAndRearms() {
    var tracker = BarcodeStabilityTracker(requiredConsecutiveHits: 3)
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertNil(tracker.ingest(nil))
    // Streak restarted from zero.
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertEqual(tracker.ingest(ean), ean)
  }

  func testNilAfterFireRearmsForTheSamePayload() {
    var tracker = BarcodeStabilityTracker(requiredConsecutiveHits: 2)
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertEqual(tracker.ingest(ean), ean)
    // Barcode leaves the frame, then returns: it may fire again (the
    // controller's phase guard/blacklist decides what to do with it).
    XCTAssertNil(tracker.ingest(nil))
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertEqual(tracker.ingest(ean), ean)
  }

  func testResetRearms() {
    var tracker = BarcodeStabilityTracker(requiredConsecutiveHits: 2)
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertEqual(tracker.ingest(ean), ean)
    tracker.reset()
    XCTAssertNil(tracker.ingest(ean))
    XCTAssertEqual(tracker.ingest(ean), ean)
  }
}
