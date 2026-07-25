import CoreImage
import CoreImage.CIFilterBuiltins
import UIKit
import Vision
import simd

/// Fuses a burst of overview shots taken from slightly different camera
/// positions into one glare-free composite — the technique behind Google's
/// PhotoScan app.
///
/// Glare is specular: it moves across the print as the camera moves, while
/// the artwork's own diffuse color stays put. So a patch that is washed out
/// in one shot is clean in another taken a hand-width away. Each extra shot
/// is registered onto the reference with a homography — exact for a flat
/// puzzle.
///
/// A plain global min-composite (darkest pixel wins) has three failure
/// modes validated against real device captures (`scripts/stitch_quality/`,
/// exp29): it erases fine bright detail (stars) under 1-3px residual
/// misalignment, smears background micro-parallax a homography can't fit
/// (e.g. carpet), and lets an exposure-mismatched frame's global darkness
/// win pixels that have nothing to do with glare. The masked-healing
/// pipeline below (ported from `scripts/stitch_quality/stitch.py --mode
/// masked`; see that directory's `README.md`) fixes all three:
/// per-frame photometric gain compensation before the min-composite, and
/// confining the min-composite's effect to a feathered mask built from a
/// *robust* (vote-of-covering-frames) darkening estimate, so pristine
/// reference pixels survive everywhere the mask doesn't fire.
///
/// Registration is unchanged and still runs on Core Image/Vision
/// (`VNHomographicImageRegistrationRequest` for the homography,
/// `CIPerspectiveTransform` for the warp). The gain/mask/blend math (see
/// "Masked healing" below) instead runs as plain Swift array arithmetic
/// over sRGB-encoded pixel buffers rather than through Core Image's own
/// compositing filters, which blend in a *linear* working color space —
/// the brightness thresholds this recipe is tuned around
/// (`maskDarkeningThreshold`, `maskBrightnessFloor`, `gainGrayRange`) are
/// measured directly off sRGB-encoded JPEG bytes in the Python reference,
/// and would mean something subtly different if run through a filter that
/// silently linearizes first.
enum GlareFreeComposer {
  /// Long-side cap frames are downscaled to before compositing. Bounds
  /// Core Image work on ~12 MP captures while staying above
  /// `ImageUtilities.normalizedJPEG`'s 1920 upload cap, so the composite
  /// loses nothing the backend would ever see.
  static let workingMaxDimension: CGFloat = 2048

  /// Long-side cap for the registration proxies (see
  /// `registrationProxy`). Registration doesn't need compositing
  /// resolution — halving it quarters Vision's work per pass, and the
  /// found warp is rescaled onto the working frames.
  static let proxyMaxDimension: CGFloat = 1024

  // MARK: - Masked healing tunables (scripts/stitch_quality/stitch.py
  // --mode masked; see that directory's README.md)

  /// Long-side size (pixels) the glare-mask math (gain compensation,
  /// robust darkening, threshold/dilate/feather) runs at. This is plain
  /// CPU array work, not GPU Core Image, so a coarser scale keeps it cheap
  /// (roughly 1 MP across up to five buffers) without materially changing
  /// the mask's shape — the mask only ever ends up as a blend WEIGHT,
  /// upscaled back to `workingMaxDimension` before the final blend touches
  /// any actual pixel color, which is where resolution genuinely matters.
  /// (Verified this doesn't change the mask's *accuracy*, not just its
  /// cost: running the same math at full `workingMaxDimension` resolution
  /// on a real capture with a heavily-textured carpet background produced
  /// statistically the same result, just ~5x slower — see
  /// `GlareFreeMaskedHealing.swift`'s `healedComposite` docs.)
  static let maskMaxDimension: CGFloat = 1024

  /// Gray range (0-255) a pixel must fall in, on both the reference and a
  /// warped corner frame, to contribute to that frame's photometric gain
  /// estimate. Mirrors `stitch.py`'s `GAIN_GRAY_LOW`/`GAIN_GRAY_HIGH`:
  /// excludes near-black (noisy ratio, possible shadow) and near-white
  /// (glare/saturation, exactly what compositing is trying to fix) pixels
  /// from the estimate.
  static let gainGrayRange: ClosedRange<Float> = 30...220

  /// The remaining masked-healing constants are tuned (and documented) at
  /// `stitch.py`'s own `WORKING_LONG_SIDE` (2048), the same scale as
  /// `workingMaxDimension` here, so `healedComposite` uses them unscaled
  /// for anything it computes at working resolution, and scales them down
  /// by `maskMaxDimension / workingMaxDimension` for the mask math that
  /// runs at the coarser `maskMaxDimension` scale instead.
  ///
  /// Gaussian blur sigma applied to the robust darkening map before
  /// thresholding, to suppress resample/JPEG noise. Mirrors `stitch.py`'s
  /// `MASK_DARKENING_BLUR_SIGMA`.
  static let maskDarkeningBlurSigma: Float = 3.0
  /// Per-pixel (blurred) darkening above which a pixel is candidate glare,
  /// measured on the vote-of-covering-frames darkening estimate (see
  /// `robustDarkening`) rather than a plain min-based signal, which
  /// carries a large, glare-unrelated noise floor on high-variance
  /// textures like carpet — see the benchmark README's "Mask signal tuning
  /// history" for the tuning story.
  /// Mirrors `stitch.py`'s `MASK_DARKENING_THRESHOLD`.
  static let maskDarkeningThreshold: Float = 30.0
  /// Reference gray level below which a pixel is excluded from the glare
  /// mask regardless of darkening — keeps moving hand shadows (dark in
  /// the reference already) out: glare heals FROM a bright, washed-out
  /// reference pixel, a shadow doesn't. Mirrors `stitch.py`'s
  /// `MASK_BRIGHTNESS_FLOOR`.
  static let maskBrightnessFloor: Float = 110.0
  /// Structuring-element radius (pixels) the raw glare mask is dilated by,
  /// so the blended region comfortably covers a glare patch's soft,
  /// gradually-darkening rim, not just its thresholded core. Mirrors
  /// `stitch.py`'s `MASK_DILATE_RADIUS_PX`.
  static let maskDilateRadius: Float = 15.0
  /// Gaussian blur sigma used to feather the dilated mask into a smooth
  /// [0, 1] alpha, avoiding a hard seam at the blend boundary. Mirrors
  /// `stitch.py`'s `MASK_FEATHER_SIGMA`.
  static let maskFeatherSigma: Float = 8.0

  struct Composite {
    let image: UIImage
    /// How many of the extra frames actually registered and joined the
    /// composite. 0 means the result is just the reference shot.
    let alignedFrameCount: Int
  }

  // Not `private`: `GlareFreeMaskedHealing.swift`'s extension (a separate
  // file, to keep this type's own body under SwiftLint's length ceiling)
  // reuses the same `CIContext` for its own `createCGImage` calls.
  static let context = CIContext()

  /// Builds the glare-free composite. Synchronous and heavy (a few seconds
  /// for five frames) — call it off the main actor.
  ///
  /// `expectedShifts` optionally carries, per extra frame, the content
  /// displacement the guided capture flow expects (unit coordinates,
  /// top-left origin — the shot was taken with a known puzzle anchor
  /// steered to the screen center, so the shift is the anchor's offset
  /// from center). Each becomes an extra registration seed hypothesis;
  /// nil entries and a nil array mean "no expectation".
  ///
  /// Frames that fail to register (Vision throws, or refinement never
  /// converges) are skipped rather than failing the whole composite: a
  /// glare-free result from three frames still beats the single reference
  /// shot. Returns nil only when the reference itself is unusable or the
  /// final render fails.
  static func compose(
    reference: UIImage, others: [UIImage], expectedShifts: [CGSize?]? = nil
  ) -> Composite? {
    guard let referenceCG = downscaledUpright(reference),
      let referenceProxies = registrationProxies(of: referenceCG)
    else { return nil }
    let extent = CGRect(x: 0, y: 0, width: referenceCG.width, height: referenceCG.height)
    let referenceImage = CIImage(cgImage: referenceCG)

    // Registration is unchanged from the plain min-composite pipeline: each
    // frame that registers and passes `alignmentError`'s verification gate
    // (inside `homography`/`refined`) contributes its warped image here,
    // UNFILLED (transparent, not white, outside its own footprint) —
    // filling gaps is now `healedComposite`'s job, which needs to tell
    // "white because glare-free" apart from "white because unfilled".
    var verifiedFrames: [CIImage] = []
    for (index, other) in others.enumerated() {
      let expectedShift = expectedShifts.flatMap { $0.indices.contains(index) ? $0[index] : nil }
      guard let otherCG = downscaledUpright(other),
        let warp = homography(
          mapping: otherCG, ontoReferenceProxies: referenceProxies,
          expectedShift: expectedShift),
        let warped = warpedImage(otherCG, by: warp)
      else { continue }
      verifiedFrames.append(warped.cropped(to: extent))
    }

    guard !verifiedFrames.isEmpty else {
      guard let outputCG = context.createCGImage(referenceImage, from: extent) else { return nil }
      return Composite(image: UIImage(cgImage: outputCG), alignedFrameCount: 0)
    }

    guard
      let healed = healedComposite(
        reference: referenceImage, verifiedFrames: verifiedFrames, extent: extent)
    else {
      // The masked-healing math failed outright (an unexpected Core
      // Graphics allocation failure, say) — fall back to the reference
      // alone rather than losing the whole capture. `alignedFrameCount`
      // reports 0, not how many frames registered: no frame contributed
      // to this output, and 0 is what the capture view keys its
      // "used the center photo as-is" degradation message on.
      guard let outputCG = context.createCGImage(referenceImage, from: extent) else { return nil }
      return Composite(image: UIImage(cgImage: outputCG), alignedFrameCount: 0)
    }
    return Composite(image: healed, alignedFrameCount: verifiedFrames.count)
  }

  // MARK: - Registration

  /// Homographic refinement passes for
  /// `homography(mapping:ontoReferenceProxies:)`. The translational
  /// bootstrap lands within a few pixels, after which one or two passes
  /// converge; anything still unconverged after 6 is treated as failed.
  private static let maxRegistrationPasses = 6
  /// A refinement step that moves the frame's central region by less than
  /// this many proxy pixels counts as converged.
  private static let convergenceThreshold: CGFloat = 3
  /// Blur radii for the coarse and fine registration proxies, as fractions
  /// of the proxy's long side so behavior is resolution-independent
  /// (1024-long proxies blur at 16 and 4).
  private static let coarseBlurFraction: CGFloat = 1.0 / 64.0
  private static let fineBlurFraction: CGFloat = 1.0 / 256.0
  /// Acceptance cap for `alignmentError`: mean absolute difference (in
  /// linear RGB) between the reference proxy and the warped floating proxy
  /// over the frame's central half. Well-aligned synthetic pairs measure
  /// ~0.012 (resampling noise plus capped glare rims); a stuck-at-identity
  /// misregistration measures ~0.075.
  private static let maxAlignmentError: CGFloat = 0.04

  /// The two registration proxies of one frame: heavily blurred for the
  /// coarse passes (a wider convergence basin), lightly blurred for the
  /// final sub-pixel refinement.
  private struct RegistrationProxies {
    let coarse: CGImage
    let fine: CGImage
  }

  /// The homography mapping `floating`'s pixels into the reference's
  /// working-resolution space, or nil when registration fails or never
  /// converges.
  ///
  /// Vision's homographic registration is an intensity-based aligner, not
  /// a feature matcher: it is sub-pixel exact for small displacements but
  /// erratic for larger ones, and bright specular patches — the very
  /// thing this composer exists to remove — are moving outliers that can
  /// pull it off entirely. Four measures make it dependable here:
  ///
  /// 1. **Highlight-capped proxies** — registration runs on copies whose
  ///    brightness is clamped to flat gray (`registrationProxies`), so
  ///    glare interiors carry no gradients for the aligner to chase.
  /// 2. **Translational bootstrap** — a translation-only registration on
  ///    the strongly blurred proxies seeds the warp. Its search is far
  ///    more constrained, so it lands within a few pixels even for
  ///    displacements where the homographic aligner diverges.
  /// 3. **Homographic refinement** — register, warp by the estimate,
  ///    re-register on the lightly blurred proxies; each pass corrects
  ///    what remains, including the perspective the bootstrap can't model.
  /// 4. **Alignment verification** — "the step got small" also describes
  ///    an aligner that has given up, so a converged warp must additionally
  ///    pass a direct image comparison (`alignmentError`) before it is
  ///    trusted.
  ///
  /// A pair that never converges, or whose converged warp fails
  /// verification, is reported as failed rather than smearing a misaligned
  /// frame into the composite.
  private static func homography(
    mapping floating: CGImage, ontoReferenceProxies reference: RegistrationProxies,
    expectedShift: CGSize? = nil
  ) -> matrix_float3x3? {
    guard let floatingProxies = registrationProxies(of: floating) else { return nil }
    let proxyExtent = CGRect(
      x: 0, y: 0, width: reference.coarse.width, height: reference.coarse.height)
    // Seed hypotheses, best first: the translational bootstrap usually
    // lands within a few pixels, but a dominant specular patch can hijack
    // its correlation entirely (it is the brightest structure in the
    // frame even after capping), so the guided flow's geometric
    // expectation (when it has one) and identity — right whenever the
    // user barely moved between shots — back it up. The verification gate
    // decides; a bad seed costs a few wasted passes, never a bad warp.
    var seeds: [matrix_float3x3] = [matrix_identity_float3x3]
    if let expectedShift {
      seeds.insert(seedMatrix(expectedShift: expectedShift, proxyExtent: proxyExtent), at: 0)
    }
    if let bootstrap = translation(
      mapping: floatingProxies.coarse, ontoReference: reference.coarse)
    {
      seeds.insert(bootstrap, at: 0)
    }
    for seed in seeds {
      guard
        let refined = refined(
          from: seed, floatingProxies: floatingProxies, reference: reference,
          proxyExtent: proxyExtent)
      else { continue }
      return rescaled(
        refined, fromProxy: proxyExtent,
        toWorking: CGRect(x: 0, y: 0, width: floating.width, height: floating.height))
    }
    return nil
  }

  /// Runs the homographic refinement loop from `seed` until convergence,
  /// returning the verified proxy-space warp — or nil when refinement
  /// never converges or the converged warp fails verification.
  private static func refined(
    from seed: matrix_float3x3, floatingProxies: RegistrationProxies,
    reference: RegistrationProxies, proxyExtent: CGRect
  ) -> matrix_float3x3? {
    var total = seed
    for _ in 1...maxRegistrationPasses {
      guard
        let current = rendered(floatingProxies.fine, warpedBy: total, extent: proxyExtent),
        let step = homographyOnce(mapping: current, ontoReference: reference.fine)
      else { return nil }
      // `total` is applied first, then `step` corrects what remains —
      // matrix product in application order.
      total = step * total
      let disp = displacement(of: step, in: proxyExtent)
      if disp < convergenceThreshold {
        let error = alignmentError(
          of: total, floatingFine: floatingProxies.fine,
          referenceFine: reference.fine, extent: proxyExtent)
        guard let error, error < maxAlignmentError else { return nil }
        return total
      }
    }
    return nil
  }

  /// One Vision homographic registration pass: the warp aligning
  /// `floating` onto `reference`, as far as Vision's search range reaches.
  private static func homographyOnce(
    mapping floating: CGImage, ontoReference reference: CGImage
  ) -> matrix_float3x3? {
    let request = VNHomographicImageRegistrationRequest(targetedCGImage: floating, options: [:])
    let handler = VNImageRequestHandler(cgImage: reference, options: [:])
    guard (try? handler.perform([request])) != nil else { return nil }
    return request.results?.first?.warpTransform
  }

  /// The registration seed for a frame whose content is expected to sit
  /// `expectedShift` away from the reference (unit coordinates, top-left
  /// origin): the warp mapping floating pixels back onto the reference is
  /// the inverse translation, expressed in the proxies' lower-left pixel
  /// space (which flips the y term's sign along with negating it).
  /// Internal so tests can pin the sign conventions — a wrong-signed seed
  /// would fail silently, rescued by the other hypotheses.
  static func seedMatrix(expectedShift: CGSize, proxyExtent: CGRect) -> matrix_float3x3 {
    var matrix = matrix_identity_float3x3
    matrix.columns.2 = SIMD3<Float>(
      Float(-expectedShift.width * proxyExtent.width),
      Float(expectedShift.height * proxyExtent.height),
      1)
    return matrix
  }

  /// Vision translation-only registration of `floating` onto `reference`,
  /// as a 3×3 matrix so it can seed the homographic refinement.
  private static func translation(
    mapping floating: CGImage, ontoReference reference: CGImage
  ) -> matrix_float3x3? {
    let request = VNTranslationalImageRegistrationRequest(targetedCGImage: floating, options: [:])
    let handler = VNImageRequestHandler(cgImage: reference, options: [:])
    guard (try? handler.perform([request])) != nil,
      let transform = request.results?.first?.alignmentTransform
    else { return nil }
    var matrix = matrix_identity_float3x3
    matrix.columns.2 = SIMD3<Float>(Float(transform.tx), Float(transform.ty), 1)
    return matrix
  }

  /// Builds the coarse and fine registration proxies of `image`:
  /// downscaled to `proxyMaxDimension`, brightness capped at flat gray
  /// (per-channel minimum against a constant — glare interiors flatten to
  /// featureless gray), then Gaussian-blurred at the stage's radius.
  private static func registrationProxies(of image: CGImage) -> RegistrationProxies? {
    let width = CGFloat(image.width)
    let height = CGFloat(image.height)
    guard width > 0, height > 0 else { return nil }
    let scale = min(1, proxyMaxDimension / max(width, height))
    let extent = CGRect(
      x: 0, y: 0, width: (width * scale).rounded(.down), height: (height * scale).rounded(.down))
    let downscaled = CIImage(cgImage: image)
      .transformed(by: CGAffineTransform(scaleX: extent.width / width, y: extent.height / height))
    let cap = CIImage(color: CIColor(red: 0.55, green: 0.55, blue: 0.55)).cropped(to: extent)
    let minimum = CIFilter.minimumCompositing()
    minimum.inputImage = downscaled.cropped(to: extent)
    minimum.backgroundImage = cap
    guard let capped = minimum.outputImage else { return nil }

    func blurred(radius: CGFloat) -> CGImage? {
      let blur = CIFilter.gaussianBlur()
      blur.inputImage = capped.clampedToExtent()
      blur.radius = Float(radius)
      guard let output = blur.outputImage else { return nil }
      return context.createCGImage(output.cropped(to: extent), from: extent)
    }
    let longSide = max(extent.width, extent.height)
    guard let coarse = blurred(radius: longSide * coarseBlurFraction),
      let fine = blurred(radius: longSide * fineBlurFraction)
    else { return nil }
    return RegistrationProxies(coarse: coarse, fine: fine)
  }

  /// Verifies a candidate warp by comparing images directly: the mean
  /// absolute difference (linear RGB) between the reference fine proxy and
  /// the floating fine proxy warped by `warp`, over the central half of
  /// the frame. Read in linear space because Core Image differences in its
  /// linear working space, and re-encoding a small average to sRGB would
  /// inflate it several-fold.
  private static func alignmentError(
    of warp: matrix_float3x3, floatingFine: CGImage, referenceFine: CGImage, extent: CGRect
  ) -> CGFloat? {
    guard let warped = rendered(floatingFine, warpedBy: warp, extent: extent) else { return nil }
    let difference = CIFilter.differenceBlendMode()
    difference.inputImage = CIImage(cgImage: warped)
    difference.backgroundImage = CIImage(cgImage: referenceFine)
    guard let diffImage = difference.outputImage else { return nil }
    let average = CIFilter.areaAverage()
    average.inputImage = diffImage
    average.extent = extent.insetBy(dx: extent.width * 0.25, dy: extent.height * 0.25)
    guard let output = average.outputImage,
      let linear = CGColorSpace(name: CGColorSpace.linearSRGB)
    else { return nil }
    var pixel = [UInt8](repeating: 0, count: 4)
    context.render(
      output, toBitmap: &pixel, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
      format: .RGBA8, colorSpace: linear)
    return (CGFloat(pixel[0]) + CGFloat(pixel[1]) + CGFloat(pixel[2])) / 3 / 255
  }

  /// Rescales a homography found at proxy resolution onto the working
  /// resolution: conjugation by the proxy↔working scale, so the returned
  /// matrix maps working-space points to working-space points.
  private static func rescaled(
    _ warp: matrix_float3x3, fromProxy proxy: CGRect, toWorking working: CGRect
  ) -> matrix_float3x3 {
    let scaleX = Float(proxy.width / working.width)
    let scaleY = Float(proxy.height / working.height)
    let toProxy = matrix_float3x3(diagonal: SIMD3<Float>(scaleX, scaleY, 1))
    let toWorking = matrix_float3x3(diagonal: SIMD3<Float>(1 / scaleX, 1 / scaleY, 1))
    return toWorking * warp * toProxy
  }

  /// How far `warp` moves the frame's central region: the maximum
  /// displacement of the four quarter-inset points. Measured away from the
  /// corners because a near-converged step can still carry projective
  /// noise that only amplifies at the very edges.
  private static func displacement(of warp: matrix_float3x3, in extent: CGRect) -> CGFloat {
    var worst: CGFloat = 0
    for unitX in [0.25, 0.75] {
      for unitY in [0.25, 0.75] {
        let x = extent.width * unitX
        let y = extent.height * unitY
        let projected = warp * SIMD3<Float>(Float(x), Float(y), 1)
        guard abs(projected.z) > .ulpOfOne else { return .infinity }
        worst = max(
          worst,
          hypot(CGFloat(projected.x / projected.z) - x, CGFloat(projected.y / projected.z) - y))
      }
    }
    return worst
  }

  /// Renders `image` warped by `warp` over neutral gray, cropped to
  /// `extent` — the input to the next refinement pass. Gray, not white:
  /// this render is registered against, and a bright border would hand the
  /// aligner false structure to latch onto.
  private static func rendered(
    _ image: CGImage, warpedBy warp: matrix_float3x3, extent: CGRect
  ) -> CGImage? {
    guard let warped = warpedImage(image, by: warp) else { return nil }
    let gray = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: extent)
    return context.createCGImage(warped.composited(over: gray).cropped(to: extent), from: extent)
  }

  /// Applies `warp` with Core Image's perspective transform. A homography
  /// is fully determined by where it sends four points, so mapping the
  /// image's corners through the matrix and handing them to
  /// `CIPerspectiveTransform` reproduces exactly the warp Vision computed.
  /// Vision's registration transforms are expressed in the same
  /// lower-left-origin pixel space Core Image renders in, so the corners
  /// map straight through.
  static func warpedImage(_ image: CGImage, by warp: matrix_float3x3) -> CIImage? {
    let width = CGFloat(image.width)
    let height = CGFloat(image.height)
    func mapped(_ x: CGFloat, _ y: CGFloat) -> CGPoint? {
      let projected = warp * SIMD3<Float>(Float(x), Float(y), 1)
      guard abs(projected.z) > .ulpOfOne else { return nil }
      return CGPoint(
        x: CGFloat(projected.x / projected.z), y: CGFloat(projected.y / projected.z))
    }
    guard let bottomLeft = mapped(0, 0),
      let bottomRight = mapped(width, 0),
      let topLeft = mapped(0, height),
      let topRight = mapped(width, height)
    else { return nil }
    let filter = CIFilter.perspectiveTransform()
    filter.inputImage = CIImage(cgImage: image)
    filter.topLeft = topLeft
    filter.topRight = topRight
    filter.bottomRight = bottomRight
    filter.bottomLeft = bottomLeft
    return filter.outputImage
  }

  /// Bakes EXIF orientation into the pixels and downscales so the long side
  /// is at most `workingMaxDimension` — the same normalization for every
  /// frame, so registration compares like with like.
  private static func downscaledUpright(_ image: UIImage) -> CGImage? {
    let pixelSize = CGSize(
      width: image.size.width * image.scale, height: image.size.height * image.scale)
    let longSide = max(pixelSize.width, pixelSize.height)
    guard longSide > 0 else { return nil }
    let ratio = min(1, workingMaxDimension / longSide)
    let targetSize = CGSize(
      width: (pixelSize.width * ratio).rounded(.down),
      height: (pixelSize.height * ratio).rounded(.down))
    guard targetSize.width >= 1, targetSize.height >= 1 else { return nil }
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let upright = UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: targetSize))
    }
    return upright.cgImage
  }
}
