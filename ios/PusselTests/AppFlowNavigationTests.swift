import XCTest

@testable import Pussel

/// Covers the back chevron's state machine (`AppFlowStore.canGoBack` /
/// `goBack`), which `AppFlowView` renders as one bar item kept mounted across
/// every phase — so the button's availability is decided here rather than by
/// whichever phase view happens to be on screen.
@MainActor
final class AppFlowNavigationTests: XCTestCase {
  private func candidate() -> TrimCandidate {
    .wholeImage(jpeg: Data("not-a-real-jpeg".utf8), source: .library)
  }

  private func session() -> SolveSession {
    SolveSession(
      name: "Test", puzzleId: "p1", trimmedJPEG: Data(), targetPieceCount: 100, rows: 10, cols: 10)
  }

  func testHomeScreenHasNothingToGoBackTo() {
    let flow = AppFlowStore()
    XCTAssertFalse(flow.canGoBack)
  }

  func testGoBackLeavesASolveSession() {
    let flow = AppFlowStore()
    flow.phase = .solving(session())
    XCTAssertTrue(flow.canGoBack)

    flow.goBack()
    guard case .capturePuzzle = flow.phase else {
      return XCTFail("expected the capture phase, got \(flow.phase)")
    }
  }

  /// The back chevron is live from the moment the phase flips, with no
  /// intermediate state in which it is mounted but inert — the window a tap
  /// used to fall into.
  func testGoBackIsAvailableImmediatelyAfterOpeningASession() {
    let flow = AppFlowStore()
    flow.phase = .solving(session())
    flow.goBack()
    flow.phase = .solving(session())

    XCTAssertTrue(flow.canGoBack)
    flow.goBack()
    guard case .capturePuzzle = flow.phase else {
      return XCTFail("expected the capture phase, got \(flow.phase)")
    }
  }

  func testGoBackFromConfirmTrimReturnsHome() {
    let flow = AppFlowStore()
    flow.phase = .confirmTrim(candidate())
    XCTAssertTrue(flow.canGoBack)

    flow.goBack()
    guard case .capturePuzzle = flow.phase else {
      return XCTFail("expected the capture phase, got \(flow.phase)")
    }
  }

  /// Leaving mid-upload would strand the session being created, so the chevron
  /// is withheld there — and `goBack` honours that on its own, not just by the
  /// button being dimmed.
  func testConfirmTrimHidesBackWhileUploading() {
    let flow = AppFlowStore()
    flow.phase = .confirmTrim(candidate())
    flow.isBusy = true
    XCTAssertFalse(flow.canGoBack)

    flow.goBack()
    guard case .confirmTrim = flow.phase else {
      return XCTFail("expected to stay on confirm-trim, got \(flow.phase)")
    }
  }

  /// A solve session is never mid-upload, so the chevron stays available.
  func testSolvingKeepsBackAvailable() {
    let flow = AppFlowStore()
    flow.phase = .solving(session())
    flow.isBusy = true
    XCTAssertTrue(flow.canGoBack)
  }

  func testGoBackClearsWizardState() {
    let flow = AppFlowStore()
    flow.phase = .solving(session())
    flow.errorMessage = "boom"
    flow.pendingRetake = .camera

    flow.goBack()
    XCTAssertNil(flow.errorMessage)
    XCTAssertNil(flow.pendingRetake)
  }
}
