import CoreGraphics
import XCTest
import simd

@testable import Pussel

/// Tests for the plane rectifier's geometry: the sensor-to-upright
/// intrinsics restatement, the plane↔image homography, and the shared
/// top-down output space the glare-free burst is warped into. No ARKit
/// involved — cameras here are synthesized from known poses, so every
/// expectation is a closed-form answer rather than a recorded one.
final class PlaneRectificationTests: XCTestCase {
  // A plausible rear-camera still: 4032×3024 landscape sensor, square
  // pixels, principal point at the center.
  private let sensorResolution = CGSize(width: 4032, height: 3024)
  private let focal: Float = 1500

  private var sensorIntrinsics: simd_float3x3 {
    simd_float3x3(rows: [
      SIMD3<Float>(focal, 0, Float(sensorResolution.width) / 2),
      SIMD3<Float>(0, focal, Float(sensorResolution.height) / 2),
      SIMD3<Float>(0, 0, 1),
    ])
  }

  /// A camera `height` meters above the plane origin, aimed at it, tilted
  /// `tiltDegrees` back from straight down about the world x axis.
  ///
  /// The untilted pose looks along −y with the photo's top pointing toward
  /// −z, which is how a phone held flat over a table sees it. Because the
  /// still is rotated 90° clockwise into portrait, that costs the camera
  /// basis its own quarter turn: the sensor's +x runs down the portrait
  /// frame, so the camera's x axis is world +z and its y axis world +x.
  private func geometry(height: Float, tiltDegrees: Float) -> PlaneCaptureGeometry {
    let lookingDown = simd_float3x3(
      SIMD3<Float>(0, 0, 1), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0))
    let tilt = simd_float3x3(simd_quatf(angle: tiltDegrees * .pi / 180, axis: SIMD3(1, 0, 0)))
    let rotation = tilt * lookingDown
    // Orbit the position with the tilt so the camera keeps looking at the
    // plane origin from `height` away.
    let position = tilt * SIMD3<Float>(0, height, 0)
    var transform = matrix_identity_float4x4
    transform.columns.0 = SIMD4(rotation.columns.0, 0)
    transform.columns.1 = SIMD4(rotation.columns.1, 0)
    transform.columns.2 = SIMD4(rotation.columns.2, 0)
    transform.columns.3 = SIMD4(position, 1)
    let upright = PlaneRectification.uprightIntrinsics(
      sensor: sensorIntrinsics, sensorResolution: sensorResolution,
      captureResolution: sensorResolution)
    return PlaneCaptureGeometry(
      cameraTransform: transform, intrinsics: upright ?? matrix_identity_float3x3,
      imageSize: PlaneRectification.uprightSize(captureResolution: sensorResolution),
      planeHeight: 0)
  }

  private func project(_ plane: SIMD2<Float>, through matrix: simd_float3x3) -> CGPoint {
    let mapped = matrix * SIMD3<Float>(plane.x, plane.y, 1)
    return CGPoint(x: CGFloat(mapped.x / mapped.z), y: CGFloat(mapped.y / mapped.z))
  }

  // MARK: - Intrinsics

  func testUprightIntrinsicsRotateThePrincipalPointIntoPortrait() throws {
    let upright = try XCTUnwrap(
      PlaneRectification.uprightIntrinsics(
        sensor: sensorIntrinsics, sensorResolution: sensorResolution,
        captureResolution: sensorResolution))
    // The optical axis is the one ray whose image position is known
    // outright, so it pins the rotation: a landscape principal point
    // (2016, 1512) lands at (3024 − 1512, 2016) once rotated 90° clockwise.
    let axis = upright * SIMD3<Float>(0, 0, 1)
    XCTAssertEqual(axis.x / axis.z, 1512, accuracy: 0.01)
    XCTAssertEqual(axis.y / axis.z, 2016, accuracy: 0.01)
  }

  func testUprightIntrinsicsScaleToTheCaptureResolution() throws {
    // ARKit's intrinsics describe the video format, not the high-resolution
    // still — a capture at twice the width must double the focal length.
    let doubled = CGSize(width: 8064, height: 6048)
    let upright = try XCTUnwrap(
      PlaneRectification.uprightIntrinsics(
        sensor: sensorIntrinsics, sensorResolution: sensorResolution,
        captureResolution: doubled))
    let axis = upright * SIMD3<Float>(0, 0, 1)
    XCTAssertEqual(axis.x / axis.z, 3024, accuracy: 0.01)
    XCTAssertEqual(axis.y / axis.z, 4032, accuracy: 0.01)
  }

  func testUprightIntrinsicsRejectDegenerateSizes() {
    XCTAssertNil(
      PlaneRectification.uprightIntrinsics(
        sensor: sensorIntrinsics, sensorResolution: .zero,
        captureResolution: sensorResolution))
  }

  func testUprightSizeSwapsTheAxes() {
    XCTAssertEqual(
      PlaneRectification.uprightSize(captureResolution: sensorResolution),
      CGSize(width: 3024, height: 4032))
  }

  // MARK: - Plane ↔ image

  func testPlaneOriginProjectsToTheImageCenterWhenAimedAtIt() {
    let matrix = PlaneRectification.planeToImage(geometry(height: 0.6, tiltDegrees: 0))
    let center = project(SIMD2(0, 0), through: matrix)
    XCTAssertEqual(center.x, 1512, accuracy: 0.01)
    XCTAssertEqual(center.y, 2016, accuracy: 0.01)
  }

  func testTheImageUpDirectionRunsAwayFromTheCamera() {
    // Held flat with the photo's top toward −z, a point farther along −z
    // must appear higher up the portrait frame than the center does.
    let matrix = PlaneRectification.planeToImage(geometry(height: 0.6, tiltDegrees: 0))
    let away = project(SIMD2(0, -0.1), through: matrix)
    let right = project(SIMD2(0.1, 0), through: matrix)
    XCTAssertLessThan(away.y, 2016)
    XCTAssertEqual(away.x, 1512, accuracy: 0.01)
    XCTAssertGreaterThan(right.x, 1512)
    XCTAssertEqual(right.y, 2016, accuracy: 0.01)
  }

  func testPlanePointInvertsTheProjection() throws {
    let geometry = geometry(height: 0.7, tiltDegrees: 22)
    let matrix = PlaneRectification.planeToImage(geometry)
    let inverse = matrix.inverse
    let source = SIMD2<Float>(0.13, -0.09)
    let pixel = project(source, through: matrix)
    let round = try XCTUnwrap(
      PlaneRectification.planePoint(ofPixel: pixel, imageToPlane: inverse))
    XCTAssertEqual(round.x, source.x, accuracy: 1e-4)
    XCTAssertEqual(round.y, source.y, accuracy: 1e-4)
  }

  func testPlanePointRejectsRaysThatMissThePlane() {
    // Tilted 80° back from straight down, the top of the frame is above the
    // horizon and its ray never meets the table.
    let inverse = PlaneRectification.planeToImage(geometry(height: 0.6, tiltDegrees: 80)).inverse
    XCTAssertNil(
      PlaneRectification.planePoint(ofPixel: CGPoint(x: 1512, y: 0), imageToPlane: inverse))
  }

  // MARK: - Output space

  func testSquareOnShotRectifiesToAPlainRescale() throws {
    let geometry = geometry(height: 0.6, tiltDegrees: 0)
    let space = try XCTUnwrap(
      PlaneRectification.outputSpace(reference: geometry, maxDimension: 2048))
    let warp = try XCTUnwrap(PlaneRectification.rectification(of: geometry, into: space))
    // Nothing to undo: the photo is already square-on, so the warp may only
    // scale and shift, never rotate, shear, or foreshorten.
    XCTAssertEqual(warp[1][0], 0, accuracy: 1e-6)
    XCTAssertEqual(warp[0][1], 0, accuracy: 1e-6)
    XCTAssertEqual(warp[0][2], 0, accuracy: 1e-9)
    XCTAssertEqual(warp[1][2], 0, accuracy: 1e-9)
    XCTAssertGreaterThan(warp[0][0], 0)
    XCTAssertGreaterThan(warp[1][1], 0)
    XCTAssertEqual(warp[0][0], warp[1][1], accuracy: 1e-4)
    // ...and the frame keeps its shape and its long side.
    XCTAssertEqual(space.size.width / space.size.height, 3024.0 / 4032.0, accuracy: 1e-3)
    XCTAssertEqual(max(space.size.width, space.size.height), 2048, accuracy: 1)
  }

  func testTiltedShotRectifiesASquareBackToASquare() throws {
    let geometry = geometry(height: 0.6, tiltDegrees: 28)
    let toImage = PlaneRectification.planeToImage(geometry)
    let space = try XCTUnwrap(
      PlaneRectification.outputSpace(reference: geometry, maxDimension: 2048))
    let warp = try XCTUnwrap(PlaneRectification.rectification(of: geometry, into: space))

    // A 20 cm square lying on the plane, in the order TL, TR, BR, BL as the
    // untilted camera would see it (photo up is −z).
    let half: Float = 0.1
    let corners = [
      SIMD2<Float>(-half, -half), SIMD2<Float>(half, -half),
      SIMD2<Float>(half, half), SIMD2<Float>(-half, half),
    ]
    let inImage = corners.map { project($0, through: toImage) }
    let inOutput = inImage.map { pixel -> CGPoint in
      let mapped = warp * SIMD3<Float>(Float(pixel.x), Float(pixel.y), 1)
      return CGPoint(x: CGFloat(mapped.x / mapped.z), y: CGFloat(mapped.y / mapped.z))
    }

    func side(_ from: CGPoint, _ to: CGPoint) -> CGFloat { hypot(to.x - from.x, to.y - from.y) }
    // The photo really is oblique: its top and bottom edges differ in
    // length, which is the distortion the backend cannot crop away.
    let imageTop = side(inImage[0], inImage[1])
    let imageBottom = side(inImage[3], inImage[2])
    XCTAssertGreaterThan(abs(imageTop - imageBottom) / imageBottom, 0.1)

    // Rectified, all four sides match and the corners are right angles.
    let sides = [
      side(inOutput[0], inOutput[1]), side(inOutput[1], inOutput[2]),
      side(inOutput[2], inOutput[3]), side(inOutput[3], inOutput[0]),
    ]
    let mean = sides.reduce(0, +) / 4
    XCTAssertGreaterThan(mean, 1)
    for length in sides {
      XCTAssertEqual(length / mean, 1, accuracy: 0.002)
    }
    let across = CGVector(dx: inOutput[1].x - inOutput[0].x, dy: inOutput[1].y - inOutput[0].y)
    let down = CGVector(dx: inOutput[3].x - inOutput[0].x, dy: inOutput[3].y - inOutput[0].y)
    XCTAssertEqual((across.dx * down.dx + across.dy * down.dy) / (mean * mean), 0, accuracy: 0.005)
  }

  func testRectifiedOutputKeepsThePhotosOrientation() throws {
    let geometry = geometry(height: 0.6, tiltDegrees: 28)
    let toImage = PlaneRectification.planeToImage(geometry)
    let space = try XCTUnwrap(
      PlaneRectification.outputSpace(reference: geometry, maxDimension: 2048))
    let warp = try XCTUnwrap(PlaneRectification.rectification(of: geometry, into: space))
    // Up in the photo (−z on the plane) must stay up in the output, and
    // right must stay right — a rectified composite the user sees rotated
    // or mirrored would be a regression, not a fix.
    func warped(_ plane: SIMD2<Float>) -> CGPoint {
      project(plane, through: warp * toImage)
    }
    let origin = warped(SIMD2(0, 0))
    XCTAssertLessThan(warped(SIMD2(0, -0.1)).y, origin.y)
    XCTAssertGreaterThan(warped(SIMD2(0.1, 0)).x, origin.x)
  }

  func testOutputSpaceRefusesAShotWithTheHorizonInFrame() {
    XCTAssertNil(
      PlaneRectification.outputSpace(
        reference: geometry(height: 0.6, tiltDegrees: 80), maxDimension: 2048))
  }

  func testOutputSpaceRefusesADegenerateRequest() {
    XCTAssertNil(
      PlaneRectification.outputSpace(
        reference: geometry(height: 0.6, tiltDegrees: 0), maxDimension: 0))
  }

  // MARK: - Coordinate plumbing

  func testLowerLeftOriginFlipsBothEnds() {
    // A top-left-origin identity, restated for Core Image's lower-left
    // origin between a 100-tall source and a 40-tall destination: a source
    // point 10 above its bottom sits 90 from the top, which is 90 from the
    // destination's top, i.e. −50 from its bottom.
    let flipped = PlaneRectification.lowerLeftOrigin(
      matrix_identity_float3x3, sourceHeight: 100, destinationHeight: 40)
    let mapped = flipped * SIMD3<Float>(7, 10, 1)
    XCTAssertEqual(mapped.x / mapped.z, 7, accuracy: 1e-5)
    XCTAssertEqual(mapped.y / mapped.z, -50, accuracy: 1e-5)
  }

  // MARK: - Frame-to-frame seeds

  func testRelativeCarriesAPlanePointBetweenTwoShots() throws {
    // Two shots of the same table from different angles. Anything on the
    // plane must land where the second camera actually sees it — that is
    // what makes this usable as a registration seed.
    let first = geometry(height: 0.6, tiltDegrees: 8)
    let second = geometry(height: 0.55, tiltDegrees: 26)
    let warp = try XCTUnwrap(PlaneRectification.relative(from: first, to: second))
    for point in [SIMD2<Float>(0, 0), SIMD2(0.12, -0.08), SIMD2(-0.05, 0.11)] {
      let inFirst = project(point, through: PlaneRectification.planeToImage(first))
      let inSecond = project(point, through: PlaneRectification.planeToImage(second))
      let mapped = warp * SIMD3<Float>(Float(inFirst.x), Float(inFirst.y), 1)
      XCTAssertEqual(CGFloat(mapped.x / mapped.z), inSecond.x, accuracy: 0.5)
      XCTAssertEqual(CGFloat(mapped.y / mapped.z), inSecond.y, accuracy: 0.5)
    }
  }

  func testRelativeBetweenAShotAndItselfIsIdentity() throws {
    let only = geometry(height: 0.6, tiltDegrees: 17)
    let warp = try XCTUnwrap(PlaneRectification.relative(from: only, to: only))
    let point = SIMD3<Float>(900, 1700, 1)
    let mapped = warp * point
    XCTAssertEqual(mapped.x / mapped.z, point.x, accuracy: 0.5)
    XCTAssertEqual(mapped.y / mapped.z, point.y, accuracy: 0.5)
  }

  func testPlaneSeedIsExpressedInProxyLowerLeftPixels() throws {
    // The seed Vision is handed lives in a downscaled, bottom-up space, and
    // getting either wrong fails silently — the other seed hypotheses would
    // rescue it and only the convergence rate would suffer. So convert a
    // known plane point by hand and require the seed to agree.
    let floating = geometry(height: 0.6, tiltDegrees: 26)
    let reference = geometry(height: 0.6, tiltDegrees: 4)
    let proxy = CGRect(x: 0, y: 0, width: 768, height: 1024)
    let seed = try XCTUnwrap(
      GlareFreeComposer.planeSeed(
        floating: floating, reference: reference, proxyExtent: proxy))

    let onPlane = SIMD2<Float>(0.07, -0.06)
    let scale = proxy.width / floating.imageSize.width
    func proxyLowerLeft(_ full: CGPoint) -> CGPoint {
      CGPoint(x: full.x * scale, y: proxy.height - full.y * scale)
    }
    let from = proxyLowerLeft(project(onPlane, through: PlaneRectification.planeToImage(floating)))
    let to = proxyLowerLeft(project(onPlane, through: PlaneRectification.planeToImage(reference)))
    let mapped = seed * SIMD3<Float>(Float(from.x), Float(from.y), 1)
    XCTAssertEqual(CGFloat(mapped.x / mapped.z), to.x, accuracy: 0.5)
    XCTAssertEqual(CGFloat(mapped.y / mapped.z), to.y, accuracy: 0.5)
  }

  func testScalingDestinationMovesOnlyTheOutput() {
    let warp = simd_float3x3(diagonal: SIMD3<Float>(1, 1, 1))
    let scaled = PlaneRectification.scalingDestination(warp, by: 0.25)
    let mapped = scaled * SIMD3<Float>(100, 200, 1)
    XCTAssertEqual(mapped.x / mapped.z, 25, accuracy: 1e-4)
    XCTAssertEqual(mapped.y / mapped.z, 50, accuracy: 1e-4)
  }

  func testScalingSourceUndoesADownscale() {
    let warp = simd_float3x3(diagonal: SIMD3<Float>(2, 2, 1))
    let scaled = PlaneRectification.scalingSource(warp, by: 0.5)
    // A pixel at 50 in the half-size copy is 100 in the original, which the
    // original warp sends to 200.
    let mapped = scaled * SIMD3<Float>(50, 50, 1)
    XCTAssertEqual(mapped.x / mapped.z, 200, accuracy: 1e-4)
    XCTAssertEqual(mapped.y / mapped.z, 200, accuracy: 1e-4)
  }
}
