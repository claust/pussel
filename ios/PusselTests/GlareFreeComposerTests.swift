import UIKit
import XCTest
import simd

@testable import Pussel

/// Tests for the glare-free burst composer, running the real Vision
/// registration + Core Image min-composite pipeline on synthetic frames
/// with known glare. The frames shift by ~40 px, so a warp applied in the
/// wrong direction (an 80 px error) samples entirely different texture and
/// fails loudly — the test pins the registration convention, not just
/// "something blended".
final class GlareFreeComposerTests: XCTestCase {
  private let cardSize = 512

  /// Deterministic test card: seeded random mid-brightness rectangles.
  /// Deliberately non-periodic — a regular grid nearly matches itself at
  /// cell offsets, which can pull intensity-based registration into a
  /// false identity alignment. All colors sit well below white so glare is
  /// always the brightest thing in a frame.
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
      // Uniform patches over the two sampled points, so the assertions
      // compare solid color against solid color rather than landing on a
      // random rectangle edge blurred by warp resampling.
      UIColor(red: 0.30, green: 0.55, blue: 0.25, alpha: 1).setFill()
      context.fill(CGRect(x: 140, y: 140, width: 40, height: 40))
      UIColor(red: 0.25, green: 0.40, blue: 0.55, alpha: 1).setFill()
      context.fill(CGRect(x: 76, y: 268, width: 40, height: 40))
    }
  }

  /// One simulated shot: the card drawn shifted by `translation` over a
  /// gray backdrop, with an optional white "glare" disc burned in at
  /// `glareCenter` (canvas coordinates, top-left origin).
  private func frame(
    base: UIImage, translation: CGSize, glareCenter: CGPoint?
  ) -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(
      size: CGSize(width: cardSize, height: cardSize), format: format
    ).image { context in
      UIColor(white: 0.5, alpha: 1).setFill()
      context.fill(CGRect(x: 0, y: 0, width: cardSize, height: cardSize))
      base.draw(at: CGPoint(x: translation.width, y: translation.height))
      if let glareCenter {
        UIColor.white.setFill()
        context.cgContext.fillEllipse(
          in: CGRect(x: glareCenter.x - 48, y: glareCenter.y - 48, width: 96, height: 96))
      }
    }
  }

  /// The RGB of the pixel at `point` (top-left-origin pixel coordinates).
  private func pixel(of image: UIImage, at point: CGPoint) -> [UInt8] {
    guard let cgImage = image.cgImage else {
      XCTFail("image has no CGImage")
      return [0, 0, 0]
    }
    var rgba = [UInt8](repeating: 0, count: 4)
    guard
      let context = CGContext(
        data: &rgba,
        width: 1,
        height: 1,
        bitsPerComponent: 8,
        bytesPerRow: 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          | CGBitmapInfo.byteOrder32Big.rawValue)
    else {
      XCTFail("could not create pixel context")
      return [0, 0, 0]
    }
    // Draw so the wanted pixel lands at (0,0): offset by -point, flipped to
    // CG's lower-left origin via the height term.
    context.draw(
      cgImage,
      in: CGRect(
        x: -point.x, y: point.y - CGFloat(cgImage.height) + 1,
        width: CGFloat(cgImage.width), height: CGFloat(cgImage.height)))
    return Array(rgba[0..<3])
  }

  private func assertClose(
    _ actual: [UInt8], _ expected: [UInt8], tolerance: Int, _ message: String,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    for (got, want) in zip(actual, expected) {
      XCTAssertEqual(
        Int(got), Int(want), accuracy: tolerance, "\(message) — got \(actual), want \(expected)",
        file: file, line: line)
    }
  }

  func testComposeRemovesReferenceGlare() throws {
    let base = testCard()
    let referenceGlare = CGPoint(x: 160, y: 160)
    let reference = frame(base: base, translation: .zero, glareCenter: referenceGlare)
    let others = [
      frame(
        base: base, translation: CGSize(width: 42, height: -38),
        glareCenter: CGPoint(x: 352, y: 160)),
      frame(
        base: base, translation: CGSize(width: -40, height: 36),
        glareCenter: CGPoint(x: 160, y: 352)),
      frame(
        base: base, translation: CGSize(width: 38, height: 40),
        glareCenter: CGPoint(x: 352, y: 352)),
      frame(
        base: base, translation: CGSize(width: -36, height: -42),
        glareCenter: CGPoint(x: 256, y: 96)),
    ]

    guard let result = GlareFreeComposer.compose(reference: reference, others: others) else {
      return XCTFail("compose returned nil")
    }
    XCTAssertEqual(
      result.alignedFrameCount, others.count,
      "all synthetic frames should register onto the reference")
    XCTAssertEqual(result.image.size, reference.size)

    // The reference's glare disc was pure white; after compositing it must
    // show the card's own cell color again.
    let healed = pixel(of: result.image, at: referenceGlare)
    let want = pixel(of: base, at: referenceGlare)
    assertClose(healed, want, tolerance: 40, "glare spot should heal to the card color")
    XCTAssertLessThan(
      Int(healed[0]) + Int(healed[1]) + Int(healed[2]), 3 * 220,
      "glare spot should no longer be near-white")

    // A spot that was never glared keeps its color — compositing must not
    // disturb clean areas.
    let clean = CGPoint(x: 96, y: 288)
    assertClose(
      pixel(of: result.image, at: clean), pixel(of: base, at: clean), tolerance: 40,
      "clean area should keep the card color")
  }

  func testSeedMatrixUndoesTheExpectedContentShift() {
    let extent = CGRect(x: 0, y: 0, width: 512, height: 512)
    let seed = GlareFreeComposer.seedMatrix(
      expectedShift: CGSize(width: 0.1, height: 0.05), proxyExtent: extent)
    // Content shifted 10% right / 5% down (top-left units) puts the
    // feature that sat at lower-left-origin (256, 256) in the reference at
    // (307.2, 230.4) in the floating frame; the seed warp (floating →
    // reference) must map it back. Pinned here because a wrong-signed seed
    // fails silently — the other seed hypotheses would rescue it.
    let mapped = seed * SIMD3<Float>(307.2, 230.4, 1)
    XCTAssertEqual(CGFloat(mapped.x / mapped.z), 256, accuracy: 0.01)
    XCTAssertEqual(CGFloat(mapped.y / mapped.z), 256, accuracy: 0.01)
  }

  func testComposeWithNoOthersReturnsReference() throws {
    let base = testCard()
    guard let result = GlareFreeComposer.compose(reference: base, others: []) else {
      return XCTFail("compose returned nil")
    }
    XCTAssertEqual(result.alignedFrameCount, 0)
    XCTAssertEqual(result.image.size, base.size)
    let center = CGPoint(x: 256, y: 256)
    assertClose(
      pixel(of: result.image, at: center), pixel(of: base, at: center), tolerance: 8,
      "a no-op composite should reproduce the reference")
  }
}
