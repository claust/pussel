import CoreGraphics
import XCTest

@testable import Pussel

// MARK: - Fake consumer

/// Recording `LiveFrameConsumer` whose throttle answer is scripted, so the
/// fan-out's per-consumer gating can be driven directly.
private final class FakeFrameConsumer: LiveFrameConsumer, @unchecked Sendable {
  var accepts = true
  private(set) var submitCount = 0
  private(set) var shouldAcceptCallCount = 0

  func shouldAcceptFrame(now: Date) -> Bool {
    shouldAcceptCallCount += 1
    return accepts
  }

  func submit(cgImage: CGImage, now: Date) {
    submitCount += 1
  }
}

private func frame() -> CGImage {
  let context = CGContext(
    data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
  // Force-unwrapped on purpose: a 4×4 RGBA context cannot fail to allocate,
  // and a nil here should fail the test loudly rather than silently skip.
  // swiftlint:disable:next force_unwrapping
  return context!.makeImage()!
}

// MARK: - Tests

final class FanOutFrameConsumerTests: XCTestCase {
  func testDeliversOneFrameToEveryConsumer() {
    let first = FakeFrameConsumer()
    let second = FakeFrameConsumer()
    let fanOut = FanOutFrameConsumer([first, second])

    fanOut.submit(cgImage: frame(), now: Date())

    XCTAssertEqual(first.submitCount, 1)
    XCTAssertEqual(second.submitCount, 1)
  }

  func testAcceptsWhileAnyConsumerWouldTakeTheFrame() {
    let busy = FakeFrameConsumer()
    busy.accepts = false
    let idle = FakeFrameConsumer()

    XCTAssertTrue(FanOutFrameConsumer([busy, idle]).shouldAcceptFrame(now: Date()))
  }

  func testRejectsOnlyWhenNoConsumerWouldTakeTheFrame() {
    let busy = FakeFrameConsumer()
    busy.accepts = false
    let alsoBusy = FakeFrameConsumer()
    alsoBusy.accepts = false

    XCTAssertFalse(FanOutFrameConsumer([busy, alsoBusy]).shouldAcceptFrame(now: Date()))
  }

  /// A consumer mid-analysis sits the frame out; the other still gets it —
  /// the barcode scanner must not stall on the guide tracker, or vice versa.
  func testBusyConsumerIsSkippedWithoutBlockingTheOther() {
    let busy = FakeFrameConsumer()
    busy.accepts = false
    let idle = FakeFrameConsumer()
    let fanOut = FanOutFrameConsumer([busy, idle])

    fanOut.submit(cgImage: frame(), now: Date())

    XCTAssertEqual(busy.submitCount, 0)
    XCTAssertEqual(idle.submitCount, 1)
  }
}
