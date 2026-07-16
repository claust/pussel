import UIKit
import XCTest

@testable import Pussel

final class ImageUtilitiesTests: XCTestCase {
  /// A solid-colour image at an exact pixel size (scale 1) for dimension checks.
  private func makeImage(width: Int, height: Int) -> UIImage {
    let size = CGSize(width: width, height: height)
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: size, format: format).image { context in
      UIColor.red.setFill()
      context.fill(CGRect(origin: .zero, size: size))
    }
  }

  // MARK: rotated(_:quarterTurns:)

  func testRotatedNormalizesFullTurnToIdentity() {
    let image = makeImage(width: 8, height: 4)
    // 4 quarter-turns (and 0) normalize to no rotation: same instance back.
    XCTAssertIdentical(ImageUtilities.rotated(image, quarterTurns: 0), image)
    XCTAssertIdentical(ImageUtilities.rotated(image, quarterTurns: 4), image)
    XCTAssertIdentical(ImageUtilities.rotated(image, quarterTurns: -4), image)
  }

  func testRotatedQuarterTurnSwapsReportedSize() {
    let image = makeImage(width: 8, height: 4)
    let turned = ImageUtilities.rotated(image, quarterTurns: 1)
    XCTAssertEqual(turned.size, CGSize(width: 4, height: 8))
  }

  func testRotatedHalfTurnKeepsSize() {
    let image = makeImage(width: 8, height: 4)
    let turned = ImageUtilities.rotated(image, quarterTurns: 2)
    XCTAssertEqual(turned.size, CGSize(width: 8, height: 4))
  }

  func testRotatedNegativeTurnNormalizes() {
    let image = makeImage(width: 8, height: 4)
    // -1 (≡ 3) is still a 90° turn, so the reported size swaps.
    let turned = ImageUtilities.rotated(image, quarterTurns: -1)
    XCTAssertEqual(turned.size, CGSize(width: 4, height: 8))
  }

  // MARK: perspectiveCorrected(from:corners:maxDimension:)

  /// A quad covering the whole image, so the correction is a no-op crop and
  /// only the sizing behaviour is under test.
  private func fullFrameCorners() -> QuadCorners {
    QuadCorners(
      topLeft: NormalizedPoint(x: 0, y: 0),
      topRight: NormalizedPoint(x: 1, y: 0),
      bottomRight: NormalizedPoint(x: 1, y: 1),
      bottomLeft: NormalizedPoint(x: 0, y: 1))
  }

  private func correctedSize(
    _ image: UIImage, corners: QuadCorners, maxDimension: CGFloat
  ) throws -> CGSize {
    let corrected = try XCTUnwrap(
      ImageUtilities.perspectiveCorrected(
        from: image, corners: corners, maxDimension: maxDimension))
    return CGSize(
      width: corrected.size.width * corrected.scale,
      height: corrected.size.height * corrected.scale)
  }

  func testPerspectiveCorrectedFullFrameKeepsSizeUnderCap() throws {
    let size = try correctedSize(
      makeImage(width: 800, height: 400), corners: fullFrameCorners(), maxDimension: 3000)
    // Comfortably under the cap, so the whole frame survives at 1:1 rather
    // than being upscaled to meet it.
    XCTAssertEqual(size, CGSize(width: 800, height: 400))
  }

  func testPerspectiveCorrectedDownscalesLongSideToCap() throws {
    let size = try correctedSize(
      makeImage(width: 800, height: 400), corners: fullFrameCorners(), maxDimension: 200)
    XCTAssertEqual(size.width, 200, accuracy: 1)
    XCTAssertEqual(size.height, 100, accuracy: 1)
  }

  func testPerspectiveCorrectedCropsToQuad() throws {
    // The left half of an 800×400 image: a 400×400 square, so the result must
    // carry the quad's aspect ratio, not the source photo's.
    let corners = QuadCorners(
      topLeft: NormalizedPoint(x: 0, y: 0),
      topRight: NormalizedPoint(x: 0.5, y: 0),
      bottomRight: NormalizedPoint(x: 0.5, y: 1),
      bottomLeft: NormalizedPoint(x: 0, y: 1))
    let size = try correctedSize(
      makeImage(width: 800, height: 400), corners: corners, maxDimension: 3000)
    XCTAssertEqual(size.width, 400, accuracy: 1)
    XCTAssertEqual(size.height, 400, accuracy: 1)
  }

  func testPerspectiveCorrectedStraightensSkewedQuadToLongestEdges() throws {
    // A quad whose top edge spans half the width and whose bottom edge spans
    // all of it — a picture shot at an angle. Each output side takes the
    // longer of its two source edges, so this straightens to the full 800
    // wide rather than the foreshortened 400.
    let corners = QuadCorners(
      topLeft: NormalizedPoint(x: 0.25, y: 0),
      topRight: NormalizedPoint(x: 0.75, y: 0),
      bottomRight: NormalizedPoint(x: 1, y: 1),
      bottomLeft: NormalizedPoint(x: 0, y: 1))
    let size = try correctedSize(
      makeImage(width: 800, height: 400), corners: corners, maxDimension: 3000)
    XCTAssertEqual(size.width, 800, accuracy: 1)
  }

  func testPerspectiveCorrectedDegenerateQuadReturnsNil() {
    // Every corner in the same place: no region to straighten.
    let point = NormalizedPoint(x: 0.5, y: 0.5)
    let corners = QuadCorners(
      topLeft: point, topRight: point, bottomRight: point, bottomLeft: point)
    XCTAssertNil(
      ImageUtilities.perspectiveCorrected(
        from: makeImage(width: 800, height: 400), corners: corners))
  }

  func testPerspectiveCorrectedFeedsRotationWithoutAReEncode() throws {
    // The zoom copy's pipeline (AppModel.acceptTrim): warp, rotate, encode
    // once. `perspectiveCorrected` returning an image is what lets the
    // rotation happen before the single lossy pass.
    let corrected = try XCTUnwrap(
      ImageUtilities.perspectiveCorrected(
        from: makeImage(width: 800, height: 400), corners: fullFrameCorners(),
        maxDimension: 3000))
    let turned = ImageUtilities.rotated(corrected, quarterTurns: 1)
    XCTAssertEqual(turned.size.width * turned.scale, 400, accuracy: 1)
    XCTAssertEqual(turned.size.height * turned.scale, 800, accuracy: 1)
  }

  // MARK: rotatedJPEG(from:quarterTurns:)

  func testRotatedJPEGZeroTurnsReturnsInputUnchanged() throws {
    let data = try XCTUnwrap(makeImage(width: 8, height: 4).jpegData(compressionQuality: 0.9))
    XCTAssertEqual(ImageUtilities.rotatedJPEG(from: data, quarterTurns: 0), data)
    XCTAssertEqual(ImageUtilities.rotatedJPEG(from: data, quarterTurns: 4), data)
  }

  func testRotatedJPEGInvalidDataZeroTurnsReturnsInput() {
    let bogus = Data([0x00, 0x01, 0x02])
    XCTAssertEqual(ImageUtilities.rotatedJPEG(from: bogus, quarterTurns: 0), bogus)
  }

  func testRotatedJPEGInvalidDataWithRotationReturnsNil() {
    // A requested rotation that can't be applied must fail loudly (nil)
    // rather than silently returning the unrotated bytes.
    let bogus = Data([0x00, 0x01, 0x02])
    XCTAssertNil(ImageUtilities.rotatedJPEG(from: bogus, quarterTurns: 1))
  }

  func testRotatedJPEGSwapsDimensionsOnQuarterTurn() throws {
    let data = try XCTUnwrap(makeImage(width: 8, height: 4).jpegData(compressionQuality: 0.9))
    let rotatedData = try XCTUnwrap(ImageUtilities.rotatedJPEG(from: data, quarterTurns: 1))
    let decoded = try XCTUnwrap(UIImage(data: rotatedData))
    XCTAssertEqual(decoded.size.width * decoded.scale, 4)
    XCTAssertEqual(decoded.size.height * decoded.scale, 8)
  }

  // MARK: alphaBoundingBox(of:) / croppedToAlphaBounds(_:) / alphaTrimmedPNG(from:)

  /// A 20x20 transparent image with a 10x10 opaque red square centered at
  /// (5, 5)...(14, 14) — mirrors the backend's cleaned piece PNGs, which
  /// have a transparent margin around the actual piece.
  private func makeImageWithTransparentBorder() -> UIImage {
    let size = CGSize(width: 20, height: 20)
    let format = UIGraphicsImageRendererFormat()
    format.opaque = false
    format.scale = 1
    return UIGraphicsImageRenderer(size: size, format: format).image { context in
      UIColor.clear.setFill()
      context.fill(CGRect(origin: .zero, size: size))
      UIColor.red.setFill()
      context.fill(CGRect(x: 5, y: 5, width: 10, height: 10))
    }
  }

  private func makeFullyTransparentImage(width: Int = 10, height: Int = 10) -> UIImage {
    let size = CGSize(width: width, height: height)
    let format = UIGraphicsImageRendererFormat()
    format.opaque = false
    format.scale = 1
    return UIGraphicsImageRenderer(size: size, format: format).image { context in
      UIColor.clear.setFill()
      context.fill(CGRect(origin: .zero, size: size))
    }
  }

  func testAlphaBoundingBoxFindsOpaqueRegion() throws {
    let bbox = try XCTUnwrap(ImageUtilities.alphaBoundingBox(of: makeImageWithTransparentBorder()))
    XCTAssertEqual(bbox, CGRect(x: 5, y: 5, width: 10, height: 10))
  }

  /// Opaque BLACK content (all color bytes 0, only alpha 255) placed
  /// off-center in a non-square canvas: fails if the buffer scan reads a
  /// color byte instead of alpha (channel-order regression) or if rows are
  /// vertically flipped relative to CGImage pixel space.
  func testAlphaBoundingBoxScansAlphaChannelNotColor() throws {
    let size = CGSize(width: 20, height: 30)
    let format = UIGraphicsImageRendererFormat()
    format.opaque = false
    format.scale = 1
    let image = UIGraphicsImageRenderer(size: size, format: format).image { context in
      UIColor.clear.setFill()
      context.fill(CGRect(origin: .zero, size: size))
      UIColor.black.setFill()
      context.fill(CGRect(x: 2, y: 3, width: 5, height: 7))
    }
    let bbox = try XCTUnwrap(ImageUtilities.alphaBoundingBox(of: image))
    XCTAssertEqual(bbox, CGRect(x: 2, y: 3, width: 5, height: 7))
  }

  func testAlphaBoundingBoxNilForFullyTransparentImage() {
    XCTAssertNil(ImageUtilities.alphaBoundingBox(of: makeFullyTransparentImage()))
  }

  func testAlphaBoundingBoxCoversWholeImageWhenFullyOpaque() throws {
    let image = makeImage(width: 8, height: 4)
    let bbox = try XCTUnwrap(ImageUtilities.alphaBoundingBox(of: image))
    XCTAssertEqual(bbox, CGRect(x: 0, y: 0, width: 8, height: 4))
  }

  func testCroppedToAlphaBoundsTrimsTransparentMargin() {
    let cropped = ImageUtilities.croppedToAlphaBounds(makeImageWithTransparentBorder())
    XCTAssertEqual(cropped.size, CGSize(width: 10, height: 10))
  }

  func testCroppedToAlphaBoundsReturnsSameInstanceWhenFullyOpaque() {
    let image = makeImage(width: 8, height: 4)
    XCTAssertIdentical(ImageUtilities.croppedToAlphaBounds(image), image)
  }

  func testAlphaTrimmedPNGTrimsTransparentMargin() throws {
    let data = try XCTUnwrap(makeImageWithTransparentBorder().pngData())
    let trimmed = ImageUtilities.alphaTrimmedPNG(from: data)
    let decoded = try XCTUnwrap(UIImage(data: trimmed))
    XCTAssertEqual(decoded.size, CGSize(width: 10, height: 10))
  }

  /// Regression: when the backend's cleaned PNG replaces `displayImage`
  /// mid-session (SolveSession.process), the overlay's pre-trimmed copy must
  /// refresh too — it used to keep the raw capture's (untrimmable) bytes.
  func testCaptureEntryRetrimsWhenDisplayImageChanges() throws {
    var entry = CaptureEntry(jpeg: Data("fake jpeg".utf8))
    XCTAssertEqual(entry.trimmedDisplayImage, Data("fake jpeg".utf8))

    let cleaned = try XCTUnwrap(makeImageWithTransparentBorder().pngData())
    entry.displayImage = cleaned

    let decoded = try XCTUnwrap(UIImage(data: entry.trimmedDisplayImage))
    XCTAssertEqual(decoded.size, CGSize(width: 10, height: 10))
  }

  func testAlphaTrimmedPNGReturnsInputUnchangedForNonPNGData() {
    let data = Data("not a png".utf8)
    XCTAssertEqual(ImageUtilities.alphaTrimmedPNG(from: data), data)
  }

  func testAlphaTrimmedPNGReturnsInputUnchangedWhenFullyOpaque() throws {
    let size = CGSize(width: 8, height: 4)
    let format = UIGraphicsImageRendererFormat()
    format.opaque = false
    format.scale = 1
    let image = UIGraphicsImageRenderer(size: size, format: format).image { context in
      UIColor.red.setFill()
      context.fill(CGRect(origin: .zero, size: size))
    }
    let data = try XCTUnwrap(image.pngData())
    XCTAssertEqual(ImageUtilities.alphaTrimmedPNG(from: data), data)
  }
}
