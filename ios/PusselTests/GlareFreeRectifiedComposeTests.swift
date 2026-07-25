import CoreGraphics
import UIKit
import XCTest
import simd

/// End-to-end check that `GlareFreeComposer.compose` uses a burst's camera
/// geometry to straighten the composite.
///
/// The input is synthesized the way a tilted phone would actually see it: a
/// rectangle of known size, lying on a known plane, projected through a
/// known camera. So the answer is not "the output looks less skewed" but a
/// number — the rectangle's true 3:2 aspect, which only a correct
/// rectification can recover from a trapezoid.
@testable import Pussel

final class GlareFreeRectifiedComposeTests: XCTestCase {
  /// A small synthetic camera: 504×378 landscape sensor (so the upright still
  /// is 378×504), 200 px focal length, principal point centered. Deliberately
  /// tiny — two of these tests run the real Vision registration, whose cost is
  /// quadratic in frame size, and the rectified output is a fixed 2048 px long
  /// side regardless, so shrinking the input costs no measurement precision.
  private let sensorResolution = CGSize(width: 504, height: 378)

  private var uprightSize: CGSize {
    PlaneRectification.uprightSize(captureResolution: sensorResolution)
  }

  /// The rectangle on the table: 48 cm across by 32 cm deep, centered on
  /// the plane origin — about as much of the 0.5 m viewing distance's
  /// footprint as stays in frame at a 25° tilt. Its 1.5 aspect is what the
  /// composite must recover.
  private let cardHalfWidth: Float = 0.24
  private let cardHalfDepth: Float = 0.16
  private let cardAspect: CGFloat = 1.5

  /// A camera `tiltDegrees` back from straight down, orbited so it keeps
  /// looking at the plane origin from 0.5 m away. Mirrors the pose helper
  /// in `PlaneRectificationTests` — the still is rotated 90° clockwise into
  /// portrait, so the camera's x axis is world +z and its y axis world +x.
  private func geometry(tiltDegrees: Float) -> PlaneCaptureGeometry {
    let sensor = simd_float3x3(rows: [
      SIMD3<Float>(200, 0, Float(sensorResolution.width) / 2),
      SIMD3<Float>(0, 200, Float(sensorResolution.height) / 2),
      SIMD3<Float>(0, 0, 1),
    ])
    let lookingDown = simd_float3x3(
      SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0))
    let tilt = simd_float3x3(simd_quatf(angle: tiltDegrees * .pi / 180, axis: SIMD3(1, 0, 0)))
    let rotation = tilt * lookingDown
    var transform = matrix_identity_float4x4
    transform.columns.0 = SIMD4(rotation.columns.0, 0)
    transform.columns.1 = SIMD4(rotation.columns.1, 0)
    transform.columns.2 = SIMD4(rotation.columns.2, 0)
    transform.columns.3 = SIMD4(tilt * SIMD3<Float>(0, 0.5, 0), 1)
    let intrinsics = PlaneRectification.uprightIntrinsics(
      sensor: sensor, sensorResolution: sensorResolution,
      captureResolution: sensorResolution)
    return PlaneCaptureGeometry(
      cameraTransform: transform, intrinsics: intrinsics ?? matrix_identity_float3x3,
      imageSize: uprightSize, planeHeight: 0)
  }

  /// The shot that camera would take of the card: the card's four plane
  /// corners projected into the frame and filled dark on white.
  private func shot(
    of geometry: PlaneCaptureGeometry, background: UIColor = .white
  ) -> UIImage {
    let toImage = PlaneRectification.planeToImage(geometry)
    let corners = [
      SIMD2<Float>(-cardHalfWidth, -cardHalfDepth), SIMD2<Float>(cardHalfWidth, -cardHalfDepth),
      SIMD2<Float>(cardHalfWidth, cardHalfDepth), SIMD2<Float>(-cardHalfWidth, cardHalfDepth),
    ].map { corner -> CGPoint in
      let mapped = toImage * SIMD3<Float>(corner.x, corner.y, 1)
      return CGPoint(x: CGFloat(mapped.x / mapped.z), y: CGFloat(mapped.y / mapped.z))
    }
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: uprightSize, format: format).image { context in
      background.setFill()
      context.fill(CGRect(origin: .zero, size: uprightSize))
      UIColor(white: 0.1, alpha: 1).setFill()
      context.cgContext.beginPath()
      context.cgContext.move(to: corners[0])
      for corner in corners.dropFirst() { context.cgContext.addLine(to: corner) }
      context.cgContext.closePath()
      context.cgContext.fillPath()
    }
  }

  /// The card's horizontal extent on one scanline, in pixels. `fraction`
  /// is measured across the card's own bounds rather than the frame's — the
  /// card covers only part of the picture, and rows outside it carry no
  /// width to compare.
  private func cardWidth(
    in gray: GrayImage, bounds: CGRect, atFraction fraction: CGFloat
  ) -> CGFloat? {
    let row = Int((bounds.minY + bounds.height * fraction).rounded())
    guard row >= 0, row < gray.height else { return nil }
    var first: Int?
    var last: Int?
    for column in 0..<gray.width where gray.value(column, row) < 128 {
      if first == nil { first = column }
      last = column
    }
    guard let first, let last, last > first else { return nil }
    return CGFloat(last - first)
  }

  private func cardBounds(in gray: GrayImage) -> CGRect? {
    var minX = gray.width
    var maxX = -1
    var minY = gray.height
    var maxY = -1
    for row in 0..<gray.height {
      for column in 0..<gray.width where gray.value(column, row) < 128 {
        minX = min(minX, column)
        maxX = max(maxX, column)
        minY = min(minY, row)
        maxY = max(maxY, row)
      }
    }
    guard maxX > minX, maxY > minY else { return nil }
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }

  func testGeometryStraightensAnObliqueShotIntoATrueRectangle() throws {
    let geometry = geometry(tiltDegrees: 25)
    let oblique = shot(of: geometry)

    // Control: without geometry the composer can only pass the shot
    // through, so the card stays a trapezoid — sampled a tenth in from each
    // end, its near and far widths still differ by about 19%.
    let asShot = try XCTUnwrap(
      GlareFreeComposer.compose(reference: oblique, others: [])?.image)
    let asShotGray = try XCTUnwrap(GrayImage(asShot))
    let asShotBounds = try XCTUnwrap(cardBounds(in: asShotGray))
    let narrow = try XCTUnwrap(
      cardWidth(in: asShotGray, bounds: asShotBounds, atFraction: 0.1))
    let wide = try XCTUnwrap(cardWidth(in: asShotGray, bounds: asShotBounds, atFraction: 0.9))
    XCTAssertGreaterThan(abs(wide - narrow) / narrow, 0.15)

    // With it, the same pixels come out square-on.
    let rectified = try XCTUnwrap(
      GlareFreeComposer.compose(reference: oblique, others: [], geometries: [geometry])?.image)
    let gray = try XCTUnwrap(GrayImage(rectified))
    let bounds = try XCTUnwrap(cardBounds(in: gray))
    let top = try XCTUnwrap(cardWidth(in: gray, bounds: bounds, atFraction: 0.1))
    let bottom = try XCTUnwrap(cardWidth(in: gray, bounds: bounds, atFraction: 0.9))
    XCTAssertEqual(bottom / top, 1, accuracy: 0.02)

    // And at its true proportions — the measurement the backend's crop
    // inherits, and the thing a mere de-skew would still get wrong.
    XCTAssertEqual(bounds.width / bounds.height, cardAspect, accuracy: 0.05)
  }

  func testASquareOnShotSurvivesRectificationUnchanged() throws {
    let geometry = geometry(tiltDegrees: 0)
    let square = shot(of: geometry)
    let rectified = try XCTUnwrap(
      GlareFreeComposer.compose(reference: square, others: [], geometries: [geometry])?.image)
    let gray = try XCTUnwrap(GrayImage(rectified))
    let bounds = try XCTUnwrap(cardBounds(in: gray))
    // Nothing to undo, so nothing may be introduced either: a rectification
    // that quietly rotated or mirrored a square-on shot would still pass
    // the oblique test above.
    XCTAssertEqual(bounds.width / bounds.height, cardAspect, accuracy: 0.05)
    XCTAssertEqual(bounds.midX / CGFloat(gray.width), 0.5, accuracy: 0.02)
    XCTAssertEqual(bounds.midY / CGFloat(gray.height), 0.5, accuracy: 0.02)
  }

  func testOnlyTheReferencesGeometryIsUsed() throws {
    let geometry = geometry(tiltDegrees: 25)
    let oblique = shot(of: geometry)
    let reference = try XCTUnwrap(
      GlareFreeComposer.compose(reference: oblique, others: [], geometries: [geometry])?.image)

    // A corner shot whose geometry is nonsense — a pose 40 m up on the far
    // side of the room. Rectifying the extra frames by their own geometry
    // (the design this replaced) would drag this one somewhere unrecognisable
    // and ghost the composite; registering them as shot ignores it entirely.
    //
    // This is the regression that matters here. ARKit reports the plane of
    // the *table*, but the puzzle sits above it, so a per-frame table warp
    // leaves each shot's subject displaced by its own parallax. Registration
    // has to see the frames as taken.
    var absurd = geometry
    absurd.cameraTransform.columns.3 = SIMD4<Float>(9, 40, 9, 1)
    absurd.planeHeight = 7
    let withCorner = try XCTUnwrap(
      GlareFreeComposer.compose(
        reference: oblique, others: [oblique], geometries: [geometry, absurd])?.image)

    XCTAssertEqual(withCorner.size, reference.size)
    let gray = try XCTUnwrap(GrayImage(withCorner))
    let bounds = try XCTUnwrap(cardBounds(in: gray))
    XCTAssertEqual(bounds.width / bounds.height, cardAspect, accuracy: 0.05)
  }

  func testTheCompositeIsRectifiedAfterFramesAreFused() throws {
    let geometry = geometry(tiltDegrees: 25)
    let oblique = shot(of: geometry)
    // Two real frames, so healing runs rather than the reference-only exit:
    // the rectification must still land, and land last.
    let result = try XCTUnwrap(
      GlareFreeComposer.compose(
        reference: oblique, others: [oblique], geometries: [geometry, geometry]))
    XCTAssertNotEqual(result.image.size, uprightSize, "the output should be the rectified space")
    let gray = try XCTUnwrap(GrayImage(result.image))
    let bounds = try XCTUnwrap(cardBounds(in: gray))
    XCTAssertEqual(bounds.width / bounds.height, cardAspect, accuracy: 0.05)
  }

  func testTheRectifiedOutputIsAllRealPixels() throws {
    // The card is drawn on a mid-grey background here, not white, so any
    // uncovered wedge the rectification left behind would show up as a
    // bright band no matter what colour it was filled with.
    let geometry = geometry(tiltDegrees: 25)
    let backdrop = UIColor(white: 0.5, alpha: 1)
    let oblique = shot(of: geometry, background: backdrop)
    let rectified = try XCTUnwrap(
      GlareFreeComposer.compose(reference: oblique, others: [], geometries: [geometry])?.image)
    let gray = try XCTUnwrap(GrayImage(rectified))

    // Every pixel is either backdrop or card — nothing near white, and
    // nothing near black either. Filling the wedges (which is what this
    // replaced) put a long straight frame-spanning edge into the composite,
    // and the backend's line detector preferred it to the puzzle.
    var brightest: UInt8 = 0
    for row in 0..<gray.height {
      for column in 0..<gray.width {
        brightest = max(brightest, gray.value(column, row))
      }
    }
    XCTAssertLessThan(brightest, 200, "an uncovered wedge survived into the output")

    // ...and the crop is still the puzzle's true shape, not a sliver.
    let bounds = try XCTUnwrap(cardBounds(in: gray))
    XCTAssertEqual(bounds.width / bounds.height, cardAspect, accuracy: 0.05)
    XCTAssertGreaterThan(bounds.width / CGFloat(gray.width), 0.5, "cropped too aggressively")
  }

  func testGeometryWithoutAUsableReferenceFallsBackToTheShotAsTaken() throws {
    let geometry = geometry(tiltDegrees: 25)
    let oblique = shot(of: geometry)
    // The reference's own geometry is what defines the output space, so a
    // burst missing it must degrade to the unrectified pipeline rather than
    // rectify the corner shots into a space nothing else shares.
    let result = try XCTUnwrap(
      GlareFreeComposer.compose(reference: oblique, others: [], geometries: [nil])?.image)
    XCTAssertEqual(result.size, uprightSize)
  }
}

/// An 8-bit grayscale copy of an image, for counting pixels in tests.
private struct GrayImage {
  let width: Int
  let height: Int
  private let pixels: [UInt8]

  init?(_ image: UIImage) {
    guard let cgImage = image.cgImage else { return nil }
    width = cgImage.width
    height = cgImage.height
    var buffer = [UInt8](repeating: 0, count: width * height)
    guard
      let context = CGContext(
        data: &buffer, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
        space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
    else { return nil }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    pixels = buffer
  }

  /// Top-left-origin lookup — `CGContext` draws bottom-up, so the row index
  /// is flipped here rather than at every call site.
  func value(_ column: Int, _ row: Int) -> UInt8 {
    pixels[(height - 1 - row) * width + column]
  }
}
