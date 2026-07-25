import UIKit
import XCTest
import simd

@testable import Pussel

/// A stand-in for `GlareFreeComposer.Session`: records the burst the
/// controller drives it with instead of registering anything, so the state
/// machine can be tested without Vision.
private final class RecordingComposeSession: GlareFreeComposeSession {
  struct Frame {
    let image: UIImage
    let expectedShift: CGSize?
    let geometry: PlaneCaptureGeometry?
  }

  let reference: UIImage
  let referenceGeometry: PlaneCaptureGeometry?
  /// The corner frames handed over so far, in the order they arrived.
  private(set) var frames: [Frame] = []
  private(set) var finishCount = 0
  private(set) var isCancelled = false
  /// Models a composition that failed outright — `finish` returning nil.
  var fails = false

  init(reference: UIImage, geometry: PlaneCaptureGeometry?) {
    self.reference = reference
    self.referenceGeometry = geometry
  }

  func addFrame(_ image: UIImage, expectedShift: CGSize?, geometry: PlaneCaptureGeometry?) {
    frames.append(Frame(image: image, expectedShift: expectedShift, geometry: geometry))
  }

  func finish() async -> GlareFreeComposer.Composite? {
    finishCount += 1
    guard !fails else { return nil }
    // One aligned frame per corner handed over, so a test can tell from the
    // composite alone that the burst reached the session intact.
    return GlareFreeComposer.Composite(image: reference, alignedFrameCount: frames.count)
  }

  func cancel() { isCancelled = true }
}

/// Keeps hold of the stand-in session the controller opens for its
/// reference shot, which only exists once that shot has been taken.
private final class SessionSpy {
  var session: RecordingComposeSession?

  /// The controller's session factory, wired to this spy.
  func factory() -> (UIImage, PlaneCaptureGeometry?) async -> (any GlareFreeComposeSession)? {
    { [self] reference, geometry in
      let opened = RecordingComposeSession(reference: reference, geometry: geometry)
      session = opened
      return opened
    }
  }
}

/// Tests for the five-shot glare-free capture state machine, with capture
/// and the compositing session injected — no camera or Vision involved.
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
    let spy = SessionSpy()
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: self.solidImage()) }, session: spy.factory())
    for step in 0..<GlareFreeCaptureController.steps.count {
      XCTAssertEqual(controller.phase, .capturing(step: step))
      XCTAssertEqual(controller.capturedCount, step)
      await controller.captureShot()
    }
    XCTAssertEqual(controller.phase, .done)
    // The stand-in session reports one aligned frame per corner shot,
    // proving the reference/corner split reached it intact.
    XCTAssertEqual(
      controller.composite?.alignedFrameCount, GlareFreeCaptureController.steps.count - 1)
  }

  func testCenterShotBecomesTrackingReference() async {
    let reference = solidImage()
    let spy = SessionSpy()
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: reference) }, session: spy.factory())
    XCTAssertNil(controller.referenceShot)
    await controller.captureShot()
    XCTAssertIdentical(controller.referenceShot, reference)
    // ...and opens the composition, which every corner registers onto.
    XCTAssertIdentical(spy.session?.reference, reference)
  }

  func testAimDwellAutoCapturesCornerStep() async {
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: self.solidImage()) }, session: SessionSpy().factory())
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
      capture: { GlareFreeShot(image: self.solidImage()) }, session: SessionSpy().factory())
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
      capture: { GlareFreeShot(image: self.solidImage()) }, session: SessionSpy().factory())
    controller.ingestGuide(onTargetUpdate(step: 1))
    XCTAssertNil(controller.guide)
    XCTAssertEqual(controller.phase, .capturing(step: 0))
  }

  func testDotPositionTracksAnchorPlusOffset() async throws {
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: self.solidImage()) }, session: SessionSpy().factory())
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

  func testEachCornerIsHandedOverWithItsExpectedShift() async throws {
    let spy = SessionSpy()
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: self.solidImage()) }, session: spy.factory())
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    let frames = try XCTUnwrap(spy.session?.frames)
    XCTAssertEqual(frames.count, GlareFreeCaptureController.steps.count - 1)
    // Each corner shot is expected to have moved the content by its
    // anchor's offset from the center — e.g. the top-left anchor at
    // (0.25, 0.32) means the content moved (+0.25, +0.18).
    let first = try XCTUnwrap(frames.first?.expectedShift)
    XCTAssertEqual(first.width, 0.25, accuracy: 1e-9)
    XCTAssertEqual(first.height, 0.18, accuracy: 1e-9)
  }

  func testTheSessionGetsTheReferenceGeometryAndEachCornersOwn() async throws {
    var shotIndex = 0
    let spy = SessionSpy()
    let controller = GlareFreeCaptureController(
      capture: {
        defer { shotIndex += 1 }
        // Geometry on the reference and the second corner only: the
        // reference's own opens the session (it is the one that decides the
        // rectification), and each corner's must travel with that corner
        // rather than being shifted onto a neighbour.
        let carries = shotIndex == 0 || shotIndex == 2
        return GlareFreeShot(
          image: self.solidImage(),
          geometry: carries ? self.geometry(planeHeight: Float(shotIndex) - 1) : nil)
      },
      session: spy.factory())
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    let session = try XCTUnwrap(spy.session)
    XCTAssertEqual(try XCTUnwrap(session.referenceGeometry).planeHeight, -1)
    XCTAssertEqual(session.frames.count, GlareFreeCaptureController.steps.count - 1)
    XCTAssertNil(session.frames[0].geometry)
    XCTAssertEqual(try XCTUnwrap(session.frames[1].geometry).planeHeight, 1)
    XCTAssertNil(session.frames[2].geometry)
    XCTAssertNil(session.frames[3].geometry)
  }

  func testCornersRegisterWhileTheBurstIsStillBeingShot() async {
    let spy = SessionSpy()
    /// How many corners were already registering as each shot was taken;
    /// -1 while there is no session at all.
    var registeringAtShot: [Int] = []
    let controller = GlareFreeCaptureController(
      capture: {
        registeringAtShot.append(spy.session.map { $0.frames.count } ?? -1)
        return GlareFreeShot(image: self.solidImage())
      },
      session: spy.factory())
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    // The center shot opens the composition, and every corner joins it as
    // it lands: the last shot is taken with three registrations already
    // under way. This is the whole point of the session — that work used to
    // start only once the burst was complete, under the spinner.
    XCTAssertEqual(registeringAtShot, [-1, 0, 1, 2, 3])
    // And the wait itself happens exactly once, at the end.
    XCTAssertEqual(spy.session?.finishCount, 1)
  }

  private func geometry(planeHeight: Float) -> PlaneCaptureGeometry {
    PlaneCaptureGeometry(
      cameraTransform: matrix_identity_float4x4, intrinsics: matrix_identity_float3x3,
      imageSize: CGSize(width: 4, height: 4), planeHeight: planeHeight)
  }

  func testNilCaptureFailsTheStep() async {
    let controller = GlareFreeCaptureController(
      capture: { nil },
      session: { _, _ in
        XCTFail("no composition should open after a failed capture")
        return nil
      })
    await controller.captureShot()
    guard case .failed = controller.phase else {
      return XCTFail("expected .failed, got \(controller.phase)")
    }
  }

  func testRestartAfterFailureBeginsANewSequence() async {
    let controller = GlareFreeCaptureController(
      capture: { nil }, session: { _, _ in nil })
    await controller.captureShot()
    controller.restart()
    XCTAssertEqual(controller.phase, .capturing(step: 0))
    XCTAssertEqual(controller.capturedCount, 0)
    XCTAssertNil(controller.composite)
    XCTAssertNil(controller.referenceShot)
  }

  func testRestartCancelsTheCompositionInFlight() async throws {
    let spy = SessionSpy()
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: self.solidImage()) }, session: spy.factory())
    await controller.captureShot()
    await controller.captureShot()
    let abandoned = try XCTUnwrap(spy.session)
    XCTAssertEqual(abandoned.frames.count, 1)
    controller.restart()
    // The frame already registering belongs to a reference shot the next
    // burst won't share, so its result is worth nothing — drop it rather
    // than letting it run on.
    XCTAssertTrue(abandoned.isCancelled)
    XCTAssertEqual(abandoned.finishCount, 0)
  }

  func testFailedCompositionFallsBackToReferenceShot() async {
    let spy = SessionSpy()
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: self.solidImage()) },
      session: { [spy] reference, geometry in
        let opened = RecordingComposeSession(reference: reference, geometry: geometry)
        opened.fails = true
        spy.session = opened
        return opened
      })
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    XCTAssertEqual(controller.phase, .done)
    // The degraded composite is the reference shot, flagged by a zero
    // aligned-frame count so the view can tell the user.
    XCTAssertEqual(controller.composite?.alignedFrameCount, 0)
    XCTAssertNotNil(controller.composite?.image)
  }

  func testAnUnusableReferenceShotStillFinishesTheFlow() async {
    // No session at all — the reference shot was too broken to prepare
    // one. The burst still has to end somewhere the user can act on, which
    // is the same degraded single-photo result as a failed composition.
    let reference = solidImage()
    let controller = GlareFreeCaptureController(
      capture: { GlareFreeShot(image: reference) }, session: { _, _ in nil })
    for _ in GlareFreeCaptureController.steps.indices {
      await controller.captureShot()
    }
    XCTAssertEqual(controller.phase, .done)
    XCTAssertEqual(controller.composite?.alignedFrameCount, 0)
    XCTAssertIdentical(controller.composite?.image, reference)
  }
}
