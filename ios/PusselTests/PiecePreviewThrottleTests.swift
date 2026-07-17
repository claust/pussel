import XCTest

@testable import Pussel

final class PiecePreviewThrottleTests: XCTestCase {
  func testFirstFrameAlwaysSends() {
    let throttle = PiecePreviewThrottle()
    XCTAssertTrue(throttle.shouldSend(now: Date()))
  }

  func testInFlightBlocksAdditionalSendsEvenPastMinInterval() {
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    // Well past minInterval, but still in flight — must not send.
    XCTAssertFalse(throttle.shouldSend(now: t0.addingTimeInterval(10)))
  }

  func testRespectsMinIntervalAfterCompletion() {
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    throttle.markCompleted()
    XCTAssertFalse(throttle.shouldSend(now: t0.addingTimeInterval(0.1)))
  }

  func testResumesOnceMinIntervalElapsesAfterCompletion() {
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    throttle.markCompleted()
    XCTAssertTrue(throttle.shouldSend(now: t0.addingTimeInterval(1)))
  }

  func testExactlyMinIntervalSends() {
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    throttle.markCompleted()
    XCTAssertTrue(
      throttle.shouldSend(now: t0.addingTimeInterval(PiecePreviewThrottle.minInterval)))
  }

  func testJustUnderMinIntervalDoesNotSend() {
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    throttle.markCompleted()
    XCTAssertFalse(
      throttle.shouldSend(now: t0.addingTimeInterval(PiecePreviewThrottle.minInterval - 0.01)))
  }

  func testMarkSentWithoutCompletionKeepsBlockingRegardlessOfElapsedTime() {
    // Simulates a hung/never-completing request: no amount of elapsed time
    // should let a second frame through without an explicit completion.
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    XCTAssertFalse(throttle.shouldSend(now: t0.addingTimeInterval(60)))
  }

  func testCanSendAgainImmediatelyAfterErrorCompletionPastInterval() {
    // markCompleted() doesn't distinguish success from failure — a failed
    // request must free up the throttle exactly like a successful one.
    var throttle = PiecePreviewThrottle()
    let t0 = Date()
    throttle.markSent(at: t0)
    throttle.markCompleted()
    let t1 = t0.addingTimeInterval(PiecePreviewThrottle.minInterval)
    XCTAssertTrue(throttle.shouldSend(now: t1))
    throttle.markSent(at: t1)
    XCTAssertFalse(throttle.shouldSend(now: t1.addingTimeInterval(0.01)))
  }
}
