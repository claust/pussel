import CoreGraphics
import XCTest

@testable import Pussel

/// End-to-end tests for `BarcodeDetector` against synthetically rendered
/// EAN-13 codes — the renderer below draws the real L/G/R module pattern, so
/// these run the actual Vision request (revision 1, classic CV, available in
/// the test host) rather than mocking it. The rotated cases are the
/// regression tests for the vertical-barcode bug: Vision decodes a rotated
/// EAN-13 fine, but reports a narrow-and-tall bounding box, which the old
/// width-only size gate rejected.
final class BarcodeDetectorTests: XCTestCase {
  /// The Frozen II 2x12 box code used across the barcode tests.
  private let payload = "4005556050093"

  private let detector = BarcodeDetector()

  func testDetectsHorizontalBarcode() throws {
    let frame = try XCTUnwrap(
      Self.renderFrame(payload: payload, rotatedDegrees: 0, barcodeFraction: 0.6))
    let detection = try detector.detect(cgImage: frame)
    XCTAssertEqual(detection?.payload, payload)
  }

  func testDetectsRotatedBarcodes() throws {
    for degrees in [90, 180, 270] {
      let frame = try XCTUnwrap(
        Self.renderFrame(payload: payload, rotatedDegrees: degrees, barcodeFraction: 0.6))
      let detection = try detector.detect(cgImage: frame)
      XCTAssertEqual(detection?.payload, payload, "no detection at \(degrees) degrees")
    }
  }

  func testFarAwayBarcodeRejectedBySizeGate() throws {
    // Well below minBoundingBoxLongSide in both orientations: decodable in a
    // clean synthetic render, but too small to trust from a real camera.
    for degrees in [0, 90] {
      let frame = try XCTUnwrap(
        Self.renderFrame(payload: payload, rotatedDegrees: degrees, barcodeFraction: 0.1))
      XCTAssertNil(try detector.detect(cgImage: frame), "size gate passed at \(degrees) degrees")
    }
  }

  // MARK: - EAN-13 rendering

  /// Standard EAN-13 left-odd (L) digit patterns; R is L complemented, G is
  /// R reversed.
  private static let lCodes = [
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
  ]
  private static let rCodes = lCodes.map { String($0.map { $0 == "0" ? "1" : "0" }) }
  private static let gCodes = rCodes.map { String($0.reversed()) }
  /// First-digit parity patterns for the left half.
  private static let parities = [
    "LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
    "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL",
  ]

  /// The 95-module bar pattern (1 = dark) for a 13-digit payload.
  private static func modules(for payload: String) -> String {
    let digits = payload.compactMap { $0.wholeNumberValue }
    precondition(digits.count == 13, "payload must be 13 digits")
    var bits = "101"
    for (index, parity) in parities[digits[0]].enumerated() {
      let digit = digits[1 + index]
      bits += parity == "L" ? lCodes[digit] : gCodes[digit]
    }
    bits += "01010"
    for digit in digits[7...] {
      bits += rCodes[digit]
    }
    return bits + "101"
  }

  /// Renders `payload` as an EAN-13 centered in a portrait 810x1080 frame
  /// (matching the downscaled camera frame's shape), rotated by
  /// `rotatedDegrees`, with the barcode's bar-to-bar axis spanning
  /// `barcodeFraction` of the frame width before rotation.
  private static func renderFrame(
    payload: String, rotatedDegrees: Int, barcodeFraction: CGFloat
  ) -> CGImage? {
    let frameWidth = 810
    let frameHeight = 1080
    guard
      let context = CGContext(
        data: nil, width: frameWidth, height: frameHeight, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return nil }
    // A light non-white background so the barcode's quiet zone is distinct.
    context.setFillColor(red: 0.9, green: 0.9, blue: 0.85, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: frameWidth, height: frameHeight))

    let bits = modules(for: payload)
    let barcodeLength = barcodeFraction * CGFloat(frameWidth)
    let moduleWidth = barcodeLength / CGFloat(bits.count)
    let barcodeHeight = barcodeLength * 0.4

    context.saveGState()
    context.translateBy(x: CGFloat(frameWidth) / 2, y: CGFloat(frameHeight) / 2)
    context.rotate(by: CGFloat(rotatedDegrees) * .pi / 180)
    context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
    context.fill(
      CGRect(
        x: -barcodeLength / 2 - 10 * moduleWidth, y: -barcodeHeight / 2 - 8,
        width: barcodeLength + 20 * moduleWidth, height: barcodeHeight + 16))
    context.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
    for (index, bit) in bits.enumerated() where bit == "1" {
      context.fill(
        CGRect(
          x: -barcodeLength / 2 + CGFloat(index) * moduleWidth, y: -barcodeHeight / 2,
          width: moduleWidth, height: barcodeHeight))
    }
    context.restoreGState()
    return context.makeImage()
  }
}
