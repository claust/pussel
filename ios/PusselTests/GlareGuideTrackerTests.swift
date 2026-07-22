import UIKit
import XCTest

@testable import Pussel

/// Tests for the guide tracker's measurement core, running the real Vision
/// translational registration on synthetic frames — the same seeded random
/// test card the composer tests use (a periodic pattern would false-lock
/// registration at identity). The measured offsets pin the tracker's sign
/// convention: content shifted right/down must report a positive offset in
/// top-left unit coordinates.
final class GlareGuideTrackerTests: XCTestCase {
  private let cardSize = 512

  private func testCard() -> UIImage {
    var state: UInt64 = 0x9E37_79B9_7F4A_7C15
    func next() -> CGFloat {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return CGFloat(state >> 33) / CGFloat(UInt32.max >> 1)
    }
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(
      size: CGSize(width: cardSize, height: cardSize), format: format
    ).image { context in
      UIColor(white: 0.4, alpha: 1).setFill()
      context.fill(CGRect(x: 0, y: 0, width: cardSize, height: cardSize))
      for _ in 0..<300 {
        let width = 10 + next() * 70
        let height = 10 + next() * 70
        let x = next() * (CGFloat(cardSize) - width)
        let y = next() * (CGFloat(cardSize) - height)
        UIColor(
          red: 0.15 + 0.55 * next(),
          green: 0.15 + 0.55 * next(),
          blue: 0.15 + 0.55 * next(),
          alpha: 1
        ).setFill()
        context.fill(CGRect(x: x, y: y, width: width, height: height))
      }
    }
  }

  /// The card drawn shifted by `translation` (top-left UIKit coordinates)
  /// over a gray backdrop — one simulated live frame.
  private func frame(base: UIImage, translation: CGSize) throws -> CGImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let image = UIGraphicsImageRenderer(
      size: CGSize(width: cardSize, height: cardSize), format: format
    ).image { context in
      UIColor(white: 0.45, alpha: 1).setFill()
      context.fill(CGRect(x: 0, y: 0, width: cardSize, height: cardSize))
      base.draw(at: CGPoint(x: translation.width, y: translation.height))
    }
    return try XCTUnwrap(image.cgImage)
  }

  private func referenceProxy(of reference: CGImage) throws -> CGImage {
    try XCTUnwrap(GlareGuideTracker.trackingProxy(of: reference))
  }

  func testMeasuresContentShiftInTopLeftUnits() throws {
    let base = testCard()
    let reference = try frame(base: base, translation: .zero)
    let live = try frame(base: base, translation: CGSize(width: 51, height: -38))

    let (update, translation) = GlareGuideTracker.measure(
      frame: live, referenceProxy: try referenceProxy(of: reference))
    let offset = try XCTUnwrap(update.offset, "shifted frame should still track")
    // Content moved 51 px right and 38 px up on a 512 px frame.
    XCTAssertEqual(offset.width, 51.0 / 512.0, accuracy: 0.02)
    XCTAssertEqual(offset.height, -38.0 / 512.0, accuracy: 0.02)
    XCTAssertEqual(update.frameAspect, 1)
    XCTAssertNotNil(translation, "a verified fix should be reusable as the next prior")
  }

  func testPriorSeededMeasurementFindsTheSameShift() throws {
    let base = testCard()
    let reference = try frame(base: base, translation: .zero)
    let live = try frame(base: base, translation: CGSize(width: 51, height: -38))
    let proxy = try referenceProxy(of: reference)

    // Seed with the roughly correct prior (as the previous frame's fix or
    // the step's expectation would): the residual path must converge to
    // the same answer as direct registration.
    let roughPrior = GlareGuideTracker.translationVector(
      ofUnitOffset: CGSize(width: 48.0 / 512.0, height: -34.0 / 512.0),
      proxySize: CGSize(width: proxy.width, height: proxy.height))
    let (update, _) = GlareGuideTracker.measure(
      frame: live, referenceProxy: proxy, priors: [roughPrior])
    let offset = try XCTUnwrap(update.offset)
    XCTAssertEqual(offset.width, 51.0 / 512.0, accuracy: 0.02)
    XCTAssertEqual(offset.height, -38.0 / 512.0, accuracy: 0.02)
  }

  func testBadPriorFallsThroughToDirectRegistration() throws {
    let base = testCard()
    let reference = try frame(base: base, translation: .zero)
    let live = try frame(base: base, translation: CGSize(width: 51, height: -38))
    let proxy = try referenceProxy(of: reference)

    // A wildly wrong prior fails verification; the trailing .zero
    // hypothesis (direct registration) must still find the shift.
    let badPrior = CGVector(dx: 150, dy: -120)
    let (update, _) = GlareGuideTracker.measure(
      frame: live, referenceProxy: proxy, priors: [badPrior, .zero])
    let offset = try XCTUnwrap(update.offset)
    XCTAssertEqual(offset.width, 51.0 / 512.0, accuracy: 0.02)
    XCTAssertEqual(offset.height, -38.0 / 512.0, accuracy: 0.02)
  }

  func testIdenticalFrameMeasuresZeroOffset() throws {
    let base = testCard()
    let reference = try frame(base: base, translation: .zero)
    let (update, _) = GlareGuideTracker.measure(
      frame: reference, referenceProxy: try referenceProxy(of: reference))
    let offset = try XCTUnwrap(update.offset)
    XCTAssertEqual(offset.width, 0, accuracy: 0.01)
    XCTAssertEqual(offset.height, 0, accuracy: 0.01)
  }

  func testUnrelatedSceneReportsTrackingLost() throws {
    let base = testCard()
    let reference = try frame(base: base, translation: .zero)
    // A featureless frame: whatever translation Vision returns, the
    // verification difference stays huge and the tracker must say "lost"
    // rather than hand the UI a confident wrong offset.
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let blank = UIGraphicsImageRenderer(
      size: CGSize(width: cardSize, height: cardSize), format: format
    ).image { context in
      UIColor(white: 0.2, alpha: 1).setFill()
      context.fill(CGRect(x: 0, y: 0, width: cardSize, height: cardSize))
    }
    let (update, translation) = GlareGuideTracker.measure(
      frame: try XCTUnwrap(blank.cgImage), referenceProxy: try referenceProxy(of: reference))
    XCTAssertNil(update.offset)
    XCTAssertNil(translation)
  }

  func testRotatedReferencePhotoStillTracks() throws {
    let base = testCard()
    let upright = try frame(base: base, translation: .zero)
    // A device photo: the pixel bitmap is stored rotated, with the EXIF
    // orientation tag saying how to display it upright. The reference path
    // must bake that orientation before registering against live frames,
    // which arrive physically upright.
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let rotatedBitmap = UIGraphicsImageRenderer(
      size: CGSize(width: cardSize, height: cardSize), format: format
    ).image { context in
      let half = CGFloat(cardSize) / 2
      context.cgContext.translateBy(x: half, y: half)
      context.cgContext.rotate(by: .pi / 2)
      context.cgContext.translateBy(x: -half, y: -half)
      UIImage(cgImage: upright).draw(at: .zero)
    }
    let photo = UIImage(
      cgImage: try XCTUnwrap(rotatedBitmap.cgImage), scale: 1, orientation: .left)
    // Sanity: the tagged photo displays as the upright card again. If this
    // fails the fixture is wrong, not the tracker.
    let displayed = UIGraphicsImageRenderer(size: photo.size, format: format).image { _ in
      photo.draw(at: .zero)
    }
    XCTAssertEqual(
      GlareGuideTracker.measure(
        frame: try XCTUnwrap(displayed.cgImage),
        referenceProxy: try referenceProxy(of: upright)
      ).0.offset.map { hypot($0.width, $0.height) } ?? 1, 0, accuracy: 0.01,
      "fixture: tagged photo should display as the original card")

    let proxy = try XCTUnwrap(
      GlareGuideTracker.referenceProxy(of: photo),
      "reference proxy should build from a tagged photo")
    let (update, _) = GlareGuideTracker.measure(frame: upright, referenceProxy: proxy)
    let offset = try XCTUnwrap(
      update.offset, "an upright live frame must track against a rotated reference photo")
    XCTAssertEqual(offset.width, 0, accuracy: 0.01)
    XCTAssertEqual(offset.height, 0, accuracy: 0.01)
  }

  @MainActor
  func testSetReferenceEnablesFrameAcceptance() throws {
    let tracker = GlareGuideTracker()
    XCTAssertFalse(tracker.shouldAcceptFrame(now: Date()))
    tracker.setReference(testCard())
    XCTAssertTrue(tracker.shouldAcceptFrame(now: Date()))
    tracker.clearReference()
    XCTAssertFalse(tracker.shouldAcceptFrame(now: Date()))
  }

  /// The tracker doubles as the Simulator/E2E `GlareGuideSource` — the
  /// protocol lifecycle must drive the same reference plumbing.
  @MainActor
  func testGuideSourceConformanceDrivesTheReferenceLifecycle() throws {
    let tracker = GlareGuideTracker()
    let source: any GlareGuideSource = tracker
    source.beginGuiding(reference: testCard())
    XCTAssertTrue(tracker.shouldAcceptFrame(now: Date()))
    source.setActiveStep(1)
    source.setActiveStep(nil)
    source.stopGuiding()
    XCTAssertFalse(tracker.shouldAcceptFrame(now: Date()))
  }
}
