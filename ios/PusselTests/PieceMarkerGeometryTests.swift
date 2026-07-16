import XCTest

@testable import Pussel

final class PieceMarkerGeometryTests: XCTestCase {
  private let canvas = CGSize(width: 1000, height: 500)
  private let cellSize = CGSize(width: 100, height: 100)

  // MARK: span present

  func testSpanPresentIgnoresImageSizeAndUsesCanvasFraction() {
    let span = PieceSpan(width: 0.34, height: 0.25)
    let size = PieceMarkerGeometry.size(
      span: span, imageSize: CGSize(width: 999, height: 1), canvas: canvas, cellSize: cellSize,
      rotation: 0)
    XCTAssertEqual(size, CGSize(width: canvas.width * 0.34, height: canvas.height * 0.25))
  }

  func testSpanPresentIgnoresRotationSwap() {
    let span = PieceSpan(width: 0.34, height: 0.25)
    let size0 = PieceMarkerGeometry.size(
      span: span, imageSize: .zero, canvas: canvas, cellSize: cellSize, rotation: 0)
    let size90 = PieceMarkerGeometry.size(
      span: span, imageSize: .zero, canvas: canvas, cellSize: cellSize, rotation: 90)
    // Unlike the fallback path, a span-present marker's pre-rotation frame
    // never swaps — `.rotationEffect` alone handles orientation.
    XCTAssertEqual(size0, size90)
  }

  // MARK: null-span fallback

  func testFallbackWideImageMinDimensionRendersAtTarget() {
    // 200x50 image: width-cells = 2, height-cells = 0.5 -> smaller is height.
    let size = PieceMarkerGeometry.size(
      span: nil, imageSize: CGSize(width: 200, height: 50), canvas: canvas, cellSize: cellSize,
      rotation: 0)
    // scale = 1.1 / 0.5 = 2.2 -> width = 200*2.2 = 440, height = 50*2.2 = 110
    XCTAssertEqual(size.width, 440, accuracy: 0.001)
    XCTAssertEqual(size.height, 110, accuracy: 0.001)
    // The smaller cell-relative dimension (height here) lands at exactly
    // fallbackMinCellSpan cells.
    XCTAssertEqual(
      size.height / cellSize.height, PieceMarkerGeometry.fallbackMinCellSpan, accuracy: 0.001)
  }

  func testFallbackTallImageMinDimensionRendersAtTarget() {
    // 50x200 image: width-cells = 0.5, height-cells = 2 -> smaller is width.
    let size = PieceMarkerGeometry.size(
      span: nil, imageSize: CGSize(width: 50, height: 200), canvas: canvas, cellSize: cellSize,
      rotation: 0)
    // scale = 1.1 / 0.5 = 2.2 -> width = 50*2.2 = 110, height = 200*2.2 = 440
    XCTAssertEqual(size.width, 110, accuracy: 0.001)
    XCTAssertEqual(size.height, 440, accuracy: 0.001)
    XCTAssertEqual(
      size.width / cellSize.width, PieceMarkerGeometry.fallbackMinCellSpan, accuracy: 0.001)
  }

  func testFallbackSquareImageBothDimensionsRenderAtTarget() {
    let size = PieceMarkerGeometry.size(
      span: nil, imageSize: CGSize(width: 80, height: 80), canvas: canvas, cellSize: cellSize,
      rotation: 0)
    // width-cells = height-cells = 0.8 -> scale = 1.1/0.8 = 1.375
    XCTAssertEqual(size.width, 80 * 1.375, accuracy: 0.001)
    XCTAssertEqual(size.height, 80 * 1.375, accuracy: 0.001)
    XCTAssertEqual(size.width, size.height, accuracy: 0.001)
  }

  func testFallbackNonSquareCellsScaleAxesIndependently() {
    // A non-square cell (2x wide) still yields correct per-axis scaling.
    let wideCellSize = CGSize(width: 200, height: 100)
    let size = PieceMarkerGeometry.size(
      span: nil, imageSize: CGSize(width: 100, height: 100), canvas: canvas, cellSize: wideCellSize,
      rotation: 0)
    // width-cells = 100/200 = 0.5, height-cells = 100/100 = 1.0 -> smaller
    // is width-cells, scale = 1.1/0.5 = 2.2.
    XCTAssertEqual(size.width, 220, accuracy: 0.001)
    XCTAssertEqual(size.height, 220, accuracy: 0.001)
  }

  func test90DegreeRotationSwapsFallbackDimensions() {
    let imageSize = CGSize(width: 200, height: 50)
    let unrotated = PieceMarkerGeometry.size(
      span: nil, imageSize: imageSize, canvas: canvas, cellSize: cellSize, rotation: 0)
    let rotated90 = PieceMarkerGeometry.size(
      span: nil, imageSize: imageSize, canvas: canvas, cellSize: cellSize, rotation: 90)
    let rotated270 = PieceMarkerGeometry.size(
      span: nil, imageSize: imageSize, canvas: canvas, cellSize: cellSize, rotation: 270)
    let rotated180 = PieceMarkerGeometry.size(
      span: nil, imageSize: imageSize, canvas: canvas, cellSize: cellSize, rotation: 180)

    XCTAssertEqual(rotated90.width, unrotated.height, accuracy: 0.001)
    XCTAssertEqual(rotated90.height, unrotated.width, accuracy: 0.001)
    XCTAssertEqual(rotated270.width, unrotated.height, accuracy: 0.001)
    XCTAssertEqual(rotated270.height, unrotated.width, accuracy: 0.001)
    // A half turn doesn't swap axes.
    XCTAssertEqual(rotated180, unrotated)
  }

  func testFallbackZeroSizedImageFallsBackToBareCell() {
    let size = PieceMarkerGeometry.size(
      span: nil, imageSize: .zero, canvas: canvas, cellSize: cellSize, rotation: 0)
    XCTAssertEqual(
      size,
      CGSize(
        width: cellSize.width * PieceMarkerGeometry.fallbackMinCellSpan,
        height: cellSize.height * PieceMarkerGeometry.fallbackMinCellSpan))
  }
}
