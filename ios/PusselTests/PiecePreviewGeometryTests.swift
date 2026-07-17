import XCTest

@testable import Pussel

/// The streamed frames are upright portrait (PieceCameraSession sets
/// `videoRotationAngle = 90` on the video data output connection), so the
/// backend's polygon is normalized against an image that looks exactly like
/// the preview: no rotation step exists, and the overlay mapping is a single
/// centered aspect-fill onto the view bounds — the same fit the
/// `.resizeAspectFill` preview layer applies to the same-aspect feed.
final class PiecePreviewGeometryTests: XCTestCase {
  private func assertViewPoint(
    _ actual: CGPoint, _ expected: CGPoint, accuracy: CGFloat = 0.0001,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertEqual(actual.x, expected.x, accuracy: accuracy, file: file, line: line)
    XCTAssertEqual(actual.y, expected.y, accuracy: accuracy, file: file, line: line)
  }

  // MARK: aspect-fill crop direction

  func testAspectFillWideFrameCropsHorizontally() {
    // Frame 100x50 into a 100x100 view: fill scale = max(1, 2) = 2, so the
    // 200x100 scaled frame overflows the width and is centered (originX =
    // -50), while the height fills exactly (originY = 0).
    let frameSize = CGSize(width: 100, height: 50)
    let viewBounds = CGSize(width: 100, height: 100)
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0, y: 0), frameSize: frameSize, viewBounds: viewBounds),
      CGPoint(x: -50, y: 0))
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 1, y: 1), frameSize: frameSize, viewBounds: viewBounds),
      CGPoint(x: 150, y: 100))
    // The frame center always lands at the view center under aspect-fill.
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0.5, y: 0.5), frameSize: frameSize,
        viewBounds: viewBounds),
      CGPoint(x: 50, y: 50))
  }

  func testAspectFillTallFrameCropsVertically() {
    // Frame 50x100 into a 100x100 view: fill scale = max(2, 1) = 2, so the
    // 100x200 scaled frame overflows the height and is centered (originY =
    // -50), while the width fills exactly (originX = 0).
    let frameSize = CGSize(width: 50, height: 100)
    let viewBounds = CGSize(width: 100, height: 100)
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0, y: 0), frameSize: frameSize, viewBounds: viewBounds),
      CGPoint(x: 0, y: -50))
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 1, y: 1), frameSize: frameSize, viewBounds: viewBounds),
      CGPoint(x: 100, y: 150))
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0.5, y: 0.5), frameSize: frameSize,
        viewBounds: viewBounds),
      CGPoint(x: 50, y: 50))
  }

  func testAspectFillMatchingAspectRatioHasNoCrop() {
    // A frame whose aspect ratio matches the view fills without cropping:
    // both axes scale by the same factor and origin stays at (0, 0).
    let frameSize = CGSize(width: 100, height: 100)
    let viewBounds = CGSize(width: 200, height: 200)
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0, y: 0), frameSize: frameSize, viewBounds: viewBounds),
      CGPoint(x: 0, y: 0))
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 1, y: 1), frameSize: frameSize, viewBounds: viewBounds),
      CGPoint(x: 200, y: 200))
  }

  // MARK: the device case — a portrait 3:4 frame in a taller full-screen view

  func testPortraitFrameInTallerViewTopLeftStaysTopLeft() {
    // The real shape of the problem on a phone: the streamed frame is
    // upright portrait 3:4 (e.g. 360x480 after downscale) and the overlay
    // view is a taller full-screen portrait (e.g. 390x844). Fill scale =
    // max(390/360, 844/480) = 844/480, so the width overflows and is
    // cropped equally left/right; the height fills exactly. A polygon
    // point at the frame's visual top-left must land in the view's
    // top-left region: x slightly negative (just inside the cropped
    // margin), y exactly 0 — NOT rotated to another corner and NOT
    // vertically offset.
    let frameSize = CGSize(width: 360, height: 480)
    let viewBounds = CGSize(width: 390, height: 844)
    let scale = 844.0 / 480.0  // = max(390/360, 844/480) ≈ 1.7583
    let scaledWidth = 360 * scale
    let originX = (390 - scaledWidth) / 2

    let topLeft = PiecePreviewGeometry.viewPoint(
      fromFramePoint: NormalizedPoint(x: 0, y: 0), frameSize: frameSize, viewBounds: viewBounds)
    assertViewPoint(topLeft, CGPoint(x: originX, y: 0), accuracy: 0.001)
    // Top-left region: at or left of the view's left edge, exactly at its
    // top — no rotation, no vertical offset.
    XCTAssertLessThanOrEqual(topLeft.x, 0)
    XCTAssertEqual(topLeft.y, 0, accuracy: 0.001)

    // A point just inside the frame (10% in from its left, 10% down) must
    // land in the view's upper-left quadrant.
    let nearTopLeft = PiecePreviewGeometry.viewPoint(
      fromFramePoint: NormalizedPoint(x: 0.1, y: 0.1), frameSize: frameSize,
      viewBounds: viewBounds)
    XCTAssertGreaterThan(nearTopLeft.x, topLeft.x)
    XCTAssertLessThan(nearTopLeft.x, viewBounds.width / 2)
    XCTAssertEqual(nearTopLeft.y, 84.4, accuracy: 0.001)
    XCTAssertLessThan(nearTopLeft.y, viewBounds.height / 2)

    // And the frame center still lands at the view center.
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0.5, y: 0.5), frameSize: frameSize,
        viewBounds: viewBounds),
      CGPoint(x: 195, y: 422))
  }

  // MARK: degenerate input

  func testAspectFillZeroFrameSizeStretchesOntoBounds() {
    // A degenerate frame size can't drive a fill ratio, so the point is
    // stretched directly onto the bounds rather than dividing by zero.
    let viewBounds = CGSize(width: 100, height: 200)
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 0.5, y: 0.5), frameSize: .zero, viewBounds: viewBounds),
      CGPoint(x: 50, y: 100))
    assertViewPoint(
      PiecePreviewGeometry.viewPoint(
        fromFramePoint: NormalizedPoint(x: 1, y: 1), frameSize: .zero, viewBounds: viewBounds),
      CGPoint(x: 100, y: 200))
  }

  // MARK: polygon mapping

  func testViewPolygonMapsEachPointInOrder() {
    let polygon = [
      NormalizedPoint(x: 0, y: 0),
      NormalizedPoint(x: 1, y: 0),
      NormalizedPoint(x: 0.5, y: 0.5),
    ]
    let frameSize = CGSize(width: 100, height: 100)
    let viewBounds = CGSize(width: 100, height: 100)
    let result = PiecePreviewGeometry.viewPolygon(
      fromFramePolygon: polygon, frameSize: frameSize, viewBounds: viewBounds)
    XCTAssertEqual(result.count, 3)
    assertViewPoint(result[0], CGPoint(x: 0, y: 0))
    assertViewPoint(result[1], CGPoint(x: 100, y: 0))
    assertViewPoint(result[2], CGPoint(x: 50, y: 50))
  }

  func testViewPolygonEmptyMapsToEmpty() {
    XCTAssertEqual(
      PiecePreviewGeometry.viewPolygon(
        fromFramePolygon: [], frameSize: CGSize(width: 10, height: 10),
        viewBounds: CGSize(width: 10, height: 10)),
      [])
  }
}
