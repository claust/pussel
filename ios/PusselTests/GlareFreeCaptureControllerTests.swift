import UIKit
import XCTest

@testable import Pussel

/// Tests for the five-shot glare-free capture state machine, with capture
/// and compose injected — no camera or Vision involved.
@MainActor
final class GlareFreeCaptureControllerTests: XCTestCase {
  private func solidImage() -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: CGSize(width: 4, height: 4), format: format)
      .image { context in
        UIColor.red.setFill()
        context.fill(CGRect(x: 0, y: 0, width: 4, height: 4))
      }
  }

  /// A tracker update whose offset steers the given step's anchor exactly
  /// onto the screen center.
  private func onTargetUpdate(step: Int) -> GlareGuideUpdate {
    let anchor = GlareFreeCaptureController.steps[step].anchor
    return GlareGuideUpdate(
      offset: CGSize(width: 0.5 - anchor.x, height: 0.5 - anchor.y), frameAspect: 0.75)
  }

  /// Waits for the controller's async auto-capture task to move the phase
  /// past the given step.
  private func waitForAdvance(
    of controller: GlareFreeCaptureController, past step: Int
  ) async {
    for _ in 0..<200 {
      if controller.phase != .capturing(step: step) { return }
      await Task.yield()
    }
  }

  func testAdvancesThroughAllStepsThenComposes() async {
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() },
      compose: { reference, others, _ in
        GlareFreeComposer.Composite(image: reference, alignedFrameCount: others.count)
      })
    for step in 0..<GlareFreeCaptureController.steps.count {
      XCTAssertEqual(controller.phase, .capturing(step: step))
      XCTAssertEqual(controller.capturedCount, step)
      await controller.captureShot()
    }
    XCTAssertEqual(controller.phase, .done)
    // The fake compose reports one aligned frame per non-reference shot,
    // proving the reference/others split reached it intact.
    XCTAssertEqual(
      controller.composite?.alignedFrameCount, GlareFreeCaptureController.steps.count - 1)
  }

  func testCenterShotBecomesTrackingReference() async {
    let reference = solidImage()
    let controller = GlareFreeCaptureController(
      capture: { reference }, compose: { _, _, _ in nil })
    XCTAssertNil(controller.referenceShot)
    await controller.captureShot()
    XCTAssertIdentical(controller.referenceShot, reference)
  }

  func testAimDwellAutoCapturesCornerStep() async {
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() }, compose: { _, _, _ in nil })
    await controller.captureShot()
    XCTAssertEqual(controller.phase, .capturing(step: 1))

    let start = Date()
    // One on-target measurement is not enough — the dwell needs time.
    controller.ingestGuide(onTargetUpdate(step: 1), at: start)
    XCTAssertEqual(controller.phase, .capturing(step: 1))
    // Comfortably past minDuration — Date arithmetic can land a rounding
    // error short of an exact threshold.
    controller.ingestGuide(
      onTargetUpdate(step: 1),
      at: start.addingTimeInterval(GlareAimStabilityTracker.minDuration + 0.1))
    await waitForAdvance(of: controller, past: 1)
    XCTAssertEqual(controller.phase, .capturing(step: 2))
    XCTAssertEqual(controller.capturedCount, 2)
    // The advance re-armed the aim: the new step starts without a guide
    // fix, so no stale dot can instantly re-fire.
    XCTAssertNil(controller.guide)
  }

  func testOffTargetDwellDoesNotCapture() async {
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() }, compose: { _, _, _ in nil })
    await controller.captureShot()

    // The dot sits at its resting anchor position (offset zero) — well
    // outside the center ring.
    let resting = GlareGuideUpdate(offset: .zero, frameAspect: 0.75)
    let start = Date()
    for tick in 0..<10 {
      controller.ingestGuide(resting, at: start.addingTimeInterval(Double(tick) * 0.2))
    }
    XCTAssertEqual(controller.phase, .capturing(step: 1))
    XCTAssertEqual(controller.capturedCount, 1)
  }

  func testGuideIsIgnoredWhileAimingTheReferenceShot() {
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() }, compose: { _, _, _ in nil })
    controller.ingestGuide(onTargetUpdate(step: 1))
    XCTAssertNil(controller.guide)
    XCTAssertEqual(controller.phase, .capturing(step: 0))
  }

  func testDotPositionTracksAnchorPlusOffset() async throws {
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() }, compose: { _, _, _ in nil })
    // While aiming the reference shot the dot marks the screen center.
    XCTAssertEqual(controller.dotUnitPosition, CGPoint(x: 0.5, y: 0.5))
    await controller.captureShot()
    // No fix yet — no dot.
    XCTAssertNil(controller.dotUnitPosition)
    controller.ingestGuide(
      GlareGuideUpdate(offset: CGSize(width: 0.1, height: -0.05), frameAspect: 0.75))
    let anchor = GlareFreeCaptureController.steps[1].anchor
    let dot = try XCTUnwrap(controller.dotUnitPosition)
    XCTAssertEqual(dot.x, anchor.x + 0.1, accuracy: 1e-9)
    XCTAssertEqual(dot.y, anchor.y - 0.05, accuracy: 1e-9)
  }

  func testComposeReceivesTheExpectedShifts() async throws {
    var receivedShifts: [CGSize?] = []
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() },
      compose: { reference, _, shifts in
        receivedShifts = shifts
        return GlareFreeComposer.Composite(image: reference, alignedFrameCount: 4)
      })
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    XCTAssertEqual(receivedShifts.count, GlareFreeCaptureController.steps.count - 1)
    // Each corner shot is expected to have moved the content by its
    // anchor's offset from the center — e.g. the top-left anchor at
    // (0.25, 0.32) means the content moved (+0.25, +0.18).
    let first = try XCTUnwrap(receivedShifts.first ?? nil)
    XCTAssertEqual(first.width, 0.25, accuracy: 1e-9)
    XCTAssertEqual(first.height, 0.18, accuracy: 1e-9)
  }

  func testNilCaptureFailsTheStep() async {
    let controller = GlareFreeCaptureController(
      capture: { nil },
      compose: { _, _, _ in
        XCTFail("compose should not run after a failed capture")
        return nil
      })
    await controller.captureShot()
    guard case .failed = controller.phase else {
      return XCTFail("expected .failed, got \(controller.phase)")
    }
  }

  func testRestartAfterFailureBeginsANewSequence() async {
    let controller = GlareFreeCaptureController(capture: { nil }, compose: { _, _, _ in nil })
    await controller.captureShot()
    controller.restart()
    XCTAssertEqual(controller.phase, .capturing(step: 0))
    XCTAssertEqual(controller.capturedCount, 0)
    XCTAssertNil(controller.composite)
    XCTAssertNil(controller.referenceShot)
  }

  func testNilComposeFallsBackToReferenceShot() async {
    let controller = GlareFreeCaptureController(
      capture: { self.solidImage() }, compose: { _, _, _ in nil })
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    XCTAssertEqual(controller.phase, .done)
    // The degraded composite is the reference shot, flagged by a zero
    // aligned-frame count so the view can tell the user.
    XCTAssertEqual(controller.composite?.alignedFrameCount, 0)
    XCTAssertNotNil(controller.composite?.image)
  }
}
