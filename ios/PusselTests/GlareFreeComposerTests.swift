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
    let shots: [(translation: CGSize, glare: CGPoint)] = [
      (CGSize(width: 42, height: -38), CGPoint(x: 352, y: 160)),
      (CGSize(width: -40, height: 36), CGPoint(x: 160, y: 352)),
      (CGSize(width: 38, height: 40), CGPoint(x: 352, y: 352)),
      (CGSize(width: -36, height: -42), CGPoint(x: 256, y: 96)),
    ]
    let others = shots.map { frame(base: base, translation: $0.translation, glareCenter: $0.glare) }
    // Seed with each frame's true content shift, as the capture controller
    // does in production (the guided flow always knows where each shot was
    // steered). Unseeded registration is measurably machine-dependent —
    // Vision's basin differs across runners — and is not the shipped path.
    let expectedShifts = shots.map { shot -> CGSize? in
      CGSize(
        width: shot.translation.width / CGFloat(cardSize),
        height: shot.translation.height / CGFloat(cardSize))
    }

    guard
      let result = GlareFreeComposer.compose(
        reference: reference, others: others, expectedShifts: expectedShifts)
    else {
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

  // MARK: - Masked healing: pure math

  /// A uniform `GrayMap` (every covered pixel the same gray level) with the
  /// given size — the masked-healing math's tests build small synthetic
  /// grids rather than driving the full Vision pipeline, since gain/
  /// darkening/mask thresholding is pure array arithmetic (see
  /// `GlareFreeComposer.swift`'s "Masked healing" section).
  private func grayMap(
    width: Int, height: Int, value: Float, coverage: Bool = true
  ) -> GlareFreeComposer.GrayMap {
    GlareFreeComposer.GrayMap(
      width: width, height: height,
      values: [Float](repeating: value, count: width * height),
      coverage: [Bool](repeating: coverage, count: width * height))
  }

  func testGainFactorCorrectsAUniformlyDarkenedFrame() {
    let reference = grayMap(width: 4, height: 4, value: 160)
    let warped = grayMap(width: 4, height: 4, value: 80)
    let gain = GlareFreeComposer.gainFactor(
      reference: reference, warped: warped, range: GlareFreeComposer.gainGrayRange)
    XCTAssertEqual(gain, 2.0, accuracy: 0.01, "a uniformly halved frame should read gain ~2x")
  }

  func testGainFactorIgnoresUncoveredAndOutOfRangePixels() {
    // Only the covered, mid-tone pixel (index 0) should count: index 1 is
    // uncovered, index 2 is near-black (below `gainGrayRange`), index 3 is
    // near-white (glare/saturation) -- all three would corrupt the ratio if
    // wrongly included.
    var warped = grayMap(width: 4, height: 1, value: 0)
    warped.values = [50, 50, 5, 250]
    warped.coverage = [true, false, true, true]
    var reference = grayMap(width: 4, height: 1, value: 0)
    reference.values = [100, 100, 100, 100]
    let gain = GlareFreeComposer.gainFactor(
      reference: reference, warped: warped, range: GlareFreeComposer.gainGrayRange)
    XCTAssertEqual(gain, 2.0, accuracy: 0.01)
  }

  func testGainFactorDefaultsToOneWhenNoPixelQualifies() {
    let reference = grayMap(width: 2, height: 2, value: 100)
    let warped = grayMap(width: 2, height: 2, value: 100, coverage: false)
    let gain = GlareFreeComposer.gainFactor(
      reference: reference, warped: warped, range: GlareFreeComposer.gainGrayRange)
    XCTAssertEqual(gain, 1.0, "no covered pixel qualifies -- gain should default to no-op")
  }

  func testRobustDarkeningMedianOfThreeIgnoresOneOutlierDarkFrame() {
    // Two frames agree the pixel is still bright (195, 190 -- close to the
    // reference's 200, i.e. barely glared if at all); one frame reads it
    // much darker (50), as if a moving shadow or a registration hiccup hit
    // just that frame. Round 6's all-of-3 vote picks the brightest covered
    // reading (195) -- the two bright frames veto the one dark outlier, so
    // this pixel must NOT read as a strong glare candidate -- the "star
    // erasure" case: a single darker sample must not be enough to darken
    // the mask, or fine bright detail near it would get eaten. (Round 5's
    // median-of-3 reached the same "no mask" answer here too -- 190, still
    // under threshold -- but only by chance of these particular values;
    // the vote's veto is exact by construction, not a coincidence of the
    // sample: see `robustDarkening`'s docs for why 2-of-3 median agreement
    // wasn't a safe enough bar in practice.)
    let reference = grayMap(width: 1, height: 1, value: 200)
    let frames = [
      grayMap(width: 1, height: 1, value: 195),
      grayMap(width: 1, height: 1, value: 190),
      grayMap(width: 1, height: 1, value: 50),
    ]
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    XCTAssertEqual(darkening.count, 1)
    XCTAssertEqual(
      darkening[0], 5, accuracy: 0.01,
      "the all-of-3 vote picks the brightest covered reading (195) -- darkening should read 200-195=5"
    )
  }

  func testRobustDarkeningGenuineGlareAcrossAllFramesStillDarkens() {
    // All three covering frames agree the true surface is much darker than
    // the glared reference -- this IS what the mask should catch.
    let reference = grayMap(width: 1, height: 1, value: 220)
    let frames = [
      grayMap(width: 1, height: 1, value: 120),
      grayMap(width: 1, height: 1, value: 130),
      grayMap(width: 1, height: 1, value: 125),
    ]
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    XCTAssertGreaterThan(
      darkening[0], GlareFreeComposer.maskDarkeningThreshold,
      "three frames agreeing on a much darker true surface must clear the darkening threshold")
  }

  func testRobustDarkeningTwoCoveringFramesUsesMaxNotAverage() {
    // Exactly 2 covering frames: one still glared (200, close to the
    // reference), one showing the true dark surface (100). The max (200)
    // should be used, not the average (150) -- an average would understate
    // darkening at a genuinely glared pixel where only one of two frames
    // happened to see through the glare.
    let reference = grayMap(width: 1, height: 1, value: 210)
    let frames = [
      grayMap(width: 1, height: 1, value: 200),
      grayMap(width: 1, height: 1, value: 100),
    ]
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    XCTAssertEqual(
      darkening[0], 10, accuracy: 0.01, "max(200, 100) = 200 -- darkening should read 210-200=10")
  }

  func testRobustDarkeningFewerThanTwoCoveringFramesReadsZero() {
    let reference = grayMap(width: 1, height: 1, value: 220)
    let oneFrame = [grayMap(width: 1, height: 1, value: 50)]
    let darkening = GlareFreeComposer.robustDarkening(
      reference: reference, correctedFrames: oneFrame)
    XCTAssertEqual(
      darkening[0], 0, "a single covering frame has no meaningful robust estimate -- no mask")
  }

  func testRobustDarkeningFourCoveringFramesToleratesOneDissentingBrightFrame() {
    // Three of four covering frames agree the true surface is much darker
    // than the reference; the fourth happens to share the reference's own
    // glare (still bright). The 3-of-4 vote must still darken -- a single
    // dissenting frame (e.g. a corner shot that coincidentally glares in
    // the same spot as the reference) shouldn't be enough to veto genuine,
    // widely-agreed glare.
    let reference = grayMap(width: 1, height: 1, value: 220)
    let frames = [
      grayMap(width: 1, height: 1, value: 120),
      grayMap(width: 1, height: 1, value: 130),
      grayMap(width: 1, height: 1, value: 125),
      grayMap(width: 1, height: 1, value: 210),
    ]
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    XCTAssertEqual(
      darkening[0], 90, accuracy: 0.01,
      "the second-brightest of {210, 130, 125, 120} is 130 -- darkening should read 220-130=90")
  }

  func testRobustDarkeningFourCoveringFramesOnlyTwoDarkStaysUnmasked() {
    // Only 2 of 4 covering frames read darker; the vote requires 3 of 4
    // (the second-brightest), so this must NOT register as darkening --
    // 2-of-4 agreement is exactly the coincidental-carpet-texture scenario
    // the vote was tightened to reject (see the benchmark README's "Mask
    // signal tuning history").
    let reference = grayMap(width: 1, height: 1, value: 200)
    let frames = [
      grayMap(width: 1, height: 1, value: 195),
      grayMap(width: 1, height: 1, value: 190),
      grayMap(width: 1, height: 1, value: 60),
      grayMap(width: 1, height: 1, value: 55),
    ]
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    XCTAssertLessThan(
      darkening[0], GlareFreeComposer.maskDarkeningThreshold,
      "the second-brightest of {195, 190, 60, 55} is 190 -- darkening of 10 must stay well under threshold"
    )
  }

  /// Builds a synthetic scene for `glareAlpha`: a `size`x`size` reference
  /// grid at `gray.bright` everywhere, with a `discRadius`-pixel disc
  /// centered at `discCenter` that reads `gray.dark` in every one of
  /// `frameCount` covering frames -- consistent darkening across every
  /// frame, exactly what real, moving glare looks like once it's healed
  /// (see `robustDarkening`'s docs on why that's the signal the median can
  /// trust).
  private func discFixture(
    size: Int, discCenter: (x: Int, y: Int), discRadius: Int, gray: (bright: Float, dark: Float),
    frameCount: Int
  ) -> (reference: GlareFreeComposer.GrayMap, frames: [GlareFreeComposer.GrayMap]) {
    func makeMap(disc value: Float, background: Float) -> GlareFreeComposer.GrayMap {
      var values = [Float](repeating: background, count: size * size)
      for y in 0..<size {
        for x in 0..<size {
          let dx = x - discCenter.x
          let dy = y - discCenter.y
          if dx * dx + dy * dy <= discRadius * discRadius {
            values[y * size + x] = value
          }
        }
      }
      return GlareFreeComposer.GrayMap(
        width: size, height: size, values: values,
        coverage: [Bool](repeating: true, count: size * size))
    }
    let reference = makeMap(disc: gray.bright, background: gray.bright)
    let frames = (0..<frameCount).map { _ in makeMap(disc: gray.dark, background: gray.bright) }
    return (reference, frames)
  }

  private var testMaskTuning: GlareFreeComposer.MaskTuning {
    GlareFreeComposer.MaskTuning(
      darkeningThreshold: GlareFreeComposer.maskDarkeningThreshold,
      brightnessFloor: GlareFreeComposer.maskBrightnessFloor, blurSigma: 1.5, dilateRadius: 4,
      featherSigma: 2)
  }

  func testGlareAlphaSyntheticDiscIsOneInsideAndZeroFarOutside() {
    let size = 80
    let (reference, frames) = discFixture(
      size: size, discCenter: (40, 40), discRadius: 10, gray: (bright: 230, dark: 130),
      frameCount: 3)
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    let alpha = GlareFreeComposer.glareAlpha(
      reference: reference, darkeningRobust: darkening, tuning: testMaskTuning)

    XCTAssertGreaterThan(
      alpha[40 * size + 40], 0.9, "disc center should be fully in the healed mask")
    // Far corner: outside the disc, its dilation, and its feather radius.
    XCTAssertLessThan(
      alpha[2 * size + 2], 0.05, "far outside the disc, the mask should stay essentially off")
  }

  func testGlareAlphaBrightnessFloorExcludesADarkReferenceRegion() {
    // A region that's already dark in the reference (a moving hand shadow,
    // not glare) but reads even darker in every covering frame must NOT
    // enter the mask, regardless of how much it darkens -- glare heals
    // FROM a bright, washed-out reference pixel; a shadow doesn't start
    // bright to begin with.
    let size = 40
    let (reference, frames) = discFixture(
      size: size, discCenter: (20, 20), discRadius: 8, gray: (bright: 90, dark: 20), frameCount: 3)
    let darkening = GlareFreeComposer.robustDarkening(reference: reference, correctedFrames: frames)
    let alpha = GlareFreeComposer.glareAlpha(
      reference: reference, darkeningRobust: darkening, tuning: testMaskTuning)
    XCTAssertLessThan(
      alpha[20 * size + 20], 0.05,
      "reference gray 90 is below maskBrightnessFloor (110) -- the shadow must stay unmasked")
  }

  // MARK: - Masked healing: full compose() regression

  /// `testCard()`'s own texture (proven to register reliably -- see
  /// `testComposeRemovesReferenceGlare`) with small bright specks burned in
  /// on top -- the starfield scenario that motivated the whole
  /// masked-healing port (see the benchmark README's real-dump findings):
  /// a plain global min-composite
  /// under 1-3px residual misalignment lets a neighboring frame's darker
  /// local-background pixel consistently win a speck's exact location,
  /// erasing it. A flat, nearly textureless backdrop would be closer to a
  /// real starfield box, but Vision's intensity-based registration needs
  /// actual gradient structure to lock onto (mirrors why `testCard()` is
  /// deliberately richly textured); layering specks onto that same texture
  /// keeps registration realistic while still exercising the erasure bug.
  private func testCardWithSpecks(_ base: UIImage, speckCenters: [CGPoint]) -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(
      size: CGSize(width: cardSize, height: cardSize), format: format
    ).image { context in
      base.draw(at: .zero)
      UIColor.white.setFill()
      for center in speckCenters {
        context.cgContext.fillEllipse(
          in: CGRect(x: center.x - 2, y: center.y - 2, width: 4, height: 4))
      }
    }
  }

  func testComposeSurvivesBrightSpecksUnderSubPixelMisalignment() throws {
    let base = testCard()
    let speckCenters = (0..<12).map { index -> CGPoint in
      let row = index / 4
      let column = index % 4
      return CGPoint(x: 96 + column * 96, y: 96 + row * 96)
    }
    let speckled = testCardWithSpecks(base, speckCenters: speckCenters)
    let reference = frame(base: speckled, translation: .zero, glareCenter: nil)
    // Each corner frame is offset by the guided flow's usual few dozen
    // pixels (so registration has real work to do) plus a deliberate
    // sub-pixel nudge on top -- residual misalignment at roughly the scale
    // that survives even a converged registration, which is exactly what
    // erased stars in the old global min-composite (see the benchmark
    // README's real-dump findings).
    let shots: [CGSize] = [
      CGSize(width: 42, height: -38),
      CGSize(width: -40, height: 36),
      CGSize(width: 38, height: 40),
      CGSize(width: -36, height: -42),
    ]
    let residual: [CGSize] = [
      CGSize(width: 1.4, height: -1.1), CGSize(width: -1.2, height: 1.3),
      CGSize(width: 1.1, height: 1.2), CGSize(width: -1.3, height: -1.4),
    ]
    let others = zip(shots, residual).map { shot, nudge -> UIImage in
      frame(
        base: speckled,
        translation: CGSize(width: shot.width + nudge.width, height: shot.height + nudge.height),
        glareCenter: nil)
    }
    let expectedShifts = shots.map { shot -> CGSize? in
      CGSize(width: shot.width / CGFloat(cardSize), height: shot.height / CGFloat(cardSize))
    }

    guard
      let result = GlareFreeComposer.compose(
        reference: reference, others: others, expectedShifts: expectedShifts)
    else {
      return XCTFail("compose returned nil")
    }
    XCTAssertEqual(
      result.alignedFrameCount, others.count, "all four synthetic frames should register")

    var survivedCount = 0
    for center in speckCenters {
      let composited = pixel(of: result.image, at: center)
      let untouched = pixel(of: reference, at: center)
      let compositedBrightness =
        (CGFloat(composited[0]) + CGFloat(composited[1]) + CGFloat(composited[2])) / 3
      let untouchedBrightness =
        (CGFloat(untouched[0]) + CGFloat(untouched[1]) + CGFloat(untouched[2])) / 3
      // A fully-erased speck reads close to the local card texture instead
      // of white; a speck that merely softened under resampling stays much
      // closer to its original brightness. This is the regression check:
      // the OLD plain min-composite erased roughly half the stars in
      // the benchmark's real captures (retention_ratio_excl_healed 0.525) --
      // masked healing should keep nearly all of them here, since none of
      // these specks are ever actually glared.
      if compositedBrightness > untouchedBrightness - 60 { survivedCount += 1 }
    }
    XCTAssertGreaterThanOrEqual(
      survivedCount, Int(Double(speckCenters.count) * 0.9),
      "at least 90% of specks should survive sub-pixel misalignment, matching the masked-healing target"
    )
  }
}
