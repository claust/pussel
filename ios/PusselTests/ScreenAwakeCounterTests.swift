import UIKit
import XCTest

@testable import Pussel

/// The idle timer is only observable on a real device (the Simulator never
/// dims), so these tests pin the ref-counting instead: the screen stays awake
/// for exactly as long as at least one viewfinder is on screen.
@MainActor
final class ScreenAwakeCounterTests: XCTestCase {
  private var counter = ScreenAwakeCounter()

  override func setUp() {
    super.setUp()
    // A fresh counter per test — the app uses a shared one, and leaked state
    // would make these pass or fail depending on ordering.
    counter = ScreenAwakeCounter()
    UIApplication.shared.isIdleTimerDisabled = false
  }

  override func tearDown() {
    UIApplication.shared.isIdleTimerDisabled = false
    super.tearDown()
  }

  func testAcquireDisablesIdleTimer() {
    counter.acquire()
    XCTAssertTrue(UIApplication.shared.isIdleTimerDisabled)
  }

  func testBalancedReleaseReenablesIdleTimer() {
    counter.acquire()
    counter.release()
    XCTAssertEqual(counter.holders, 0)
    XCTAssertFalse(UIApplication.shared.isIdleTimerDisabled)
  }

  /// The case the counter exists for: SwiftUI appears the next view before it
  /// disappears the last one, so the overlapping release must not drop the lock.
  func testOverlappingHoldersKeepScreenAwakeUntilLastRelease() {
    counter.acquire()
    counter.acquire()
    counter.release()
    XCTAssertTrue(UIApplication.shared.isIdleTimerDisabled, "still one viewfinder on screen")

    counter.release()
    XCTAssertFalse(UIApplication.shared.isIdleTimerDisabled)
  }

  /// An unbalanced release must not push the count negative — that would make
  /// the next acquire/release pair leave the idle timer disabled forever.
  func testUnbalancedReleaseDoesNotGoNegative() {
    counter.release()
    XCTAssertEqual(counter.holders, 0)

    counter.acquire()
    counter.release()
    XCTAssertFalse(UIApplication.shared.isIdleTimerDisabled)
  }
}
