import XCTest
import simd

@testable import Pussel

/// Tests for the AR guide source's pure geometry — the quad interpolation
/// that places the corner targets on the puzzle's plane, and the
/// front-of-camera test that keeps behind-the-camera projections (which
/// come out mirrored) off the screen. No `ARSession` involved.
final class ARGuideGeometryTests: XCTestCase {
  /// A 2×2 m quad on the floor plane, in raycast screen-corner order:
  /// top left, top right, bottom right, bottom left.
  private let quad: [SIMD3<Float>] = [
    SIMD3(-1, 0, -1),
    SIMD3(1, 0, -1),
    SIMD3(1, 0, 1),
    SIMD3(-1, 0, 1),
  ]

  private func assertEqual(
    _ point: SIMD3<Float>, _ expected: SIMD3<Float>, file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(point.x, expected.x, accuracy: 1e-5, file: file, line: line)
    XCTAssertEqual(point.y, expected.y, accuracy: 1e-5, file: file, line: line)
    XCTAssertEqual(point.z, expected.z, accuracy: 1e-5, file: file, line: line)
  }

  func testInterpolationHitsTheQuadCorners() {
    assertEqual(ARGuideGeometry.interpolated(quad: quad, at: CGPoint(x: 0, y: 0)), quad[0])
    assertEqual(ARGuideGeometry.interpolated(quad: quad, at: CGPoint(x: 1, y: 0)), quad[1])
    assertEqual(ARGuideGeometry.interpolated(quad: quad, at: CGPoint(x: 1, y: 1)), quad[2])
    assertEqual(ARGuideGeometry.interpolated(quad: quad, at: CGPoint(x: 0, y: 1)), quad[3])
  }

  func testInterpolationOfCenterAndAnchor() {
    assertEqual(
      ARGuideGeometry.interpolated(quad: quad, at: CGPoint(x: 0.5, y: 0.5)), SIMD3(0, 0, 0))
    // The top-left step anchor: a quarter across, a third down.
    assertEqual(
      ARGuideGeometry.interpolated(quad: quad, at: CGPoint(x: 0.25, y: 0.32)),
      SIMD3(-0.5, 0, -0.36))
  }

  func testInterpolationOfANonPlanarQuadStaysBilinear() {
    // Lift one corner: interpolation along the top edge must follow it.
    var lifted = quad
    lifted[1].y = 2
    let point = ARGuideGeometry.interpolated(quad: lifted, at: CGPoint(x: 0.75, y: 0))
    XCTAssertEqual(point.y, 1.5, accuracy: 1e-5)
  }

  func testIsInFrontUsesTheCameraLookDirection() {
    // The identity camera sits at the origin looking down −z.
    let identity = matrix_identity_float4x4
    XCTAssertTrue(ARGuideGeometry.isInFront(ofCameraAt: identity, point: SIMD3(0, 0, -1)))
    XCTAssertFalse(ARGuideGeometry.isInFront(ofCameraAt: identity, point: SIMD3(0, 0, 1)))
  }

  func testIsInFrontFollowsTheCameraTransform() {
    // Turn the camera 180° about y: it now looks toward +z, so the same
    // world points swap sides.
    let about = simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
    var transform = simd_float4x4(about)
    transform.columns.3 = SIMD4(0, 0, -3, 1)
    XCTAssertTrue(ARGuideGeometry.isInFront(ofCameraAt: transform, point: SIMD3(0, 0, 1)))
    XCTAssertFalse(ARGuideGeometry.isInFront(ofCameraAt: transform, point: SIMD3(0, 0, -5)))
  }
}
