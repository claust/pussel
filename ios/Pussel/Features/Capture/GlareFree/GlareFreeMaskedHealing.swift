import Accelerate
import CoreGraphics
import CoreImage
import Foundation
import UIKit

/// The masked-healing recipe: photometric gain compensation, a
/// gain-corrected min-composite, and a feathered glare mask built from a
/// robust darkening estimate, blended over the reference. Ported from
/// `scripts/stitch_quality/stitch.py --mode masked`; see that file, its
/// directory's `README.md`, and `GlareFreeComposer.swift`'s own
/// doc comment for the full picture. Split into its own file (an extension
/// on `GlareFreeComposer`, not a body of that enum's own declaration) so
/// this substantial amount of plumbing doesn't push `GlareFreeComposer`
/// itself over SwiftLint's `type_body_length` ceiling; the masked-healing
/// tunable constants stay declared on `GlareFreeComposer` itself, next to
/// its other constants, as the single source of truth callers reference.
///
/// Every per-pixel step below runs through Accelerate (vDSP/vImage) rather
/// than a hand-written Swift loop. That is a pure performance change --
/// vDSP and vImage read and write the raw buffers handed to them with no
/// colorspace interpretation whatsoever, so the sRGB-encoded-bytes domain
/// this recipe depends on (see `healedComposite` and `GrayMap`) is
/// untouched, unlike Core Image's compositing filters, which would
/// linearize. The vector formulations are written to reproduce the scalar
/// arithmetic's operand order, so results are bit-identical apart from the
/// odd last-place difference where Accelerate contracts a multiply-add.
/// This matters more than it looks: the app's own test host and every
/// debug build run this code unoptimized, where a bounds-checked Swift
/// loop over three million pixels costs roughly ten times what the same
/// loop costs at `-O`, while an Accelerate call costs the same either way.
extension GlareFreeComposer {
  /// How many pixels each vectorized pass over a full-resolution frame
  /// handles at a time. The float scratch planes the RGBA8 passes
  /// de-interleave into are sized per span rather than per frame, which
  /// keeps them a few megabytes (and mostly cache-resident) instead of
  /// scaling with a 2048x1536 working buffer.
  static let vectorSpanPixels = 1 << 18

  /// One such span being worked on: a base pointer into a block of equally
  /// sized scratch float planes, plus how many pixels the span covers. The
  /// helpers below address the planes by index, which keeps their operand
  /// lists down to the buffers they actually read and write.
  struct VectorSpan {
    let work: UnsafeMutablePointer<Float>
    let count: Int
    var size: vDSP_Length { vDSP_Length(count) }
    func plane(_ index: Int) -> UnsafeMutablePointer<Float> { work + index * count }
  }

  /// Runs the masked-healing recipe over `verifiedFrames` and blends the
  /// result over `reference`. Returns nil only on an unexpected Core
  /// Image/Graphics failure (a bitmap render or allocation coming back
  /// empty) — never because the mask ends up empty everywhere, which is a
  /// normal, valid result: an unglared capture's healed output is
  /// legitimately just the reference.
  ///
  /// Ported from `scripts/stitch_quality/stitch.py --mode masked`
  /// (`register_and_composite_masked` + `blend_with_glare_mask`); see that
  /// file and its directory's `README.md` for the recipe's validation
  /// against real device captures.
  static func healedComposite(
    reference: CIImage, verifiedFrames: [CIImage], extent: CGRect
  ) -> UIImage? {
    let workingWidth = Int(extent.width.rounded())
    let workingHeight = Int(extent.height.rounded())
    guard workingWidth > 0, workingHeight > 0,
      let referenceRGBA = rgbaBuffer(of: reference, extent: extent)
    else { return nil }

    let maskSize = downscaledSize(of: extent, longSide: maskMaxDimension)
    let maskScale =
      Float(max(maskSize.width, maskSize.height)) / Float(max(workingWidth, workingHeight))
    guard let referenceMaskGray = grayMap(of: reference, extent: extent, downscaledTo: maskSize)
    else { return nil }

    var workingRGBA: [RGBABuffer] = []
    var warpedMaskGray: [GrayMap] = []
    for frame in verifiedFrames {
      guard let rgba = rgbaBuffer(of: frame, extent: extent),
        let gray = grayMap(of: frame, extent: extent, downscaledTo: maskSize)
      else { return nil }
      workingRGBA.append(rgba)
      warpedMaskGray.append(gray)
    }

    // Photometric gain compensation, one scalar per frame, estimated at
    // mask resolution (cheap) and applied both there (for the mask's own
    // darkening estimate) and at working resolution (where it actually
    // affects the visible min-composite).
    let gains = warpedMaskGray.map {
      gainFactor(reference: referenceMaskGray, warped: $0, range: gainGrayRange)
    }
    for index in workingRGBA.indices {
      workingRGBA[index].applyGain(gains[index])
    }
    let correctedMaskGray = zip(warpedMaskGray, gains).map { gray, gain in gray.scaled(by: gain) }

    // The mask's own candidacy signal: a robust (vote-of-covering-
    // frames) darkening estimate, blurred, thresholded, dilated, and
    // feathered into a smooth alpha — see `robustDarkening` and
    // `glareAlpha`'s docs for why a vote (not min) and why the brightness
    // floor. All at mask resolution; the spatial constants are tuned at
    // `workingMaxDimension` scale (matching `stitch.py`'s own working
    // size), so they're scaled down by `maskScale` here. (An earlier
    // version of this pipeline ran the mask math at full working
    // resolution instead, to rule out this extra downscale as a source of
    // the carpet leakage the benchmark README describes -- it made no
    // measurable difference to that leakage, so the coarser, ~5x cheaper
    // resolution stays; see git history / the PR description for the
    // investigation.)
    let darkeningRobust = robustDarkening(
      reference: referenceMaskGray, correctedFrames: correctedMaskGray)
    let tuning = MaskTuning(
      darkeningThreshold: maskDarkeningThreshold, brightnessFloor: maskBrightnessFloor,
      blurSigma: maskDarkeningBlurSigma * maskScale, dilateRadius: maskDilateRadius * maskScale,
      featherSigma: maskFeatherSigma * maskScale)
    let alphaAtMaskScale = glareAlpha(
      reference: referenceMaskGray, darkeningRobust: darkeningRobust, tuning: tuning)
    let alpha = upscale(
      alphaAtMaskScale, fromWidth: maskSize.width, fromHeight: maskSize.height,
      toWidth: workingWidth, toHeight: workingHeight)

    // Gain-corrected min-composite: the darkest gain-corrected value any
    // verified frame observed at each pixel, falling back to the
    // reference wherever no verified frame's footprint reaches at all.
    // Mirrors `stitch.py`'s `build_gain_corrected_min_composite`.
    let minComposite = buildMinStack(reference: referenceRGBA, frames: workingRGBA)

    // Final blend: pristine reference everywhere the mask is 0, the
    // gain-corrected min-composite (which actually heals glare) wherever
    // it's 1, feathered in between. Mirrors `stitch.py`'s
    // `blend_with_glare_mask`.
    let output = referenceRGBA.blended(with: minComposite, alpha: alpha)
    return output.makeImage().map { UIImage(cgImage: $0) }
  }

  /// A top-left-origin, row-major sRGB RGBA8 pixel buffer — the
  /// masked-healing math's representation of a full-resolution frame.
  /// Built once per frame via `rgbaBuffer(of:extent:)` and combined/blended
  /// with vDSP/vImage passes over the raw bytes, deliberately bypassing
  /// Core Image's own (linear) compositing filters — see `healedComposite`'s
  /// docs for why.
  struct RGBABuffer {
    /// The alpha at which a warped frame counts as covering a pixel, read
    /// from the frame's own alpha channel (`CIFilter.perspectiveTransform`'s
    /// output is transparent outside the mapped quad). A firm threshold
    /// rather than `> 0` sidesteps a thin ring of partial-alpha edge pixels
    /// the warp's resampling can leave right at the quad boundary.
    static let coveringAlpha: Float = 128

    let width: Int
    let height: Int
    /// `width * height * 4` bytes, RGBA, top-left-origin, row-major.
    var bytes: [UInt8]

    /// Multiplies RGB by `gain`, clamped to a valid byte range; leaves
    /// alpha (coverage) untouched. Mirrors `stitch.py`'s
    /// `compute_gain_factor` application: `np.clip(warped * gain, 0, 255)`.
    ///
    /// `gain` is one scalar for the whole frame, so every byte value's
    /// result is precomputed into a 256-entry table -- with exactly the
    /// scalar expression this used to evaluate per pixel, making the
    /// result bit-identical -- and applied in a single vImage lookup pass.
    /// Alpha goes through an identity table, i.e. is copied unchanged.
    mutating func applyGain(_ gain: Float) {
      guard gain != 1 else { return }
      var gained = [UInt8](repeating: 0, count: 256)
      var identity = [UInt8](repeating: 0, count: 256)
      for value in 0..<256 {
        gained[value] = UInt8(max(0, min(255, Float(value) * gain)).rounded())
        identity[value] = UInt8(value)
      }
      let (spanWidth, spanHeight) = (width, height)
      bytes.withUnsafeMutableBufferPointer { raw in
        var source = vImage_Buffer(
          data: raw.baseAddress, height: vImagePixelCount(spanHeight),
          width: vImagePixelCount(spanWidth), rowBytes: spanWidth * 4)
        var destination = source
        // The tables are applied in memory order, so for this buffer's
        // RGBA layout that is R, G, B and then (identity) A.
        _ = vImageTableLookUp_ARGB8888(
          &source, &destination, gained, gained, gained, identity,
          vImage_Flags(kvImageNoFlags))
      }
    }

    /// Per-channel linear interpolation toward `other`, weighted by
    /// `alpha` (one entry per pixel, in [0, 1]):
    /// `self * (1 - alpha) + other * alpha`. Mirrors `stitch.py`'s final
    /// `reference * (1 - alpha) + min_composite * alpha` blend. Alpha
    /// stays fully opaque in the result — the masked-healing composite is
    /// always meant to be viewed as a normal opaque photo.
    func blended(with other: RGBABuffer, alpha: [Float]) -> RGBABuffer {
      let count = width * height
      // Seeded opaque, so the alpha byte no channel pass writes is already
      // the 255 this blend promises.
      var output = [UInt8](repeating: 255, count: count * 4)
      let span = min(count, GlareFreeComposer.vectorSpanPixels)
      var work = [Float](repeating: 0, count: 4 * span)
      bytes.withUnsafeBufferPointer { mine in
        other.bytes.withUnsafeBufferPointer { theirs in
          alpha.withUnsafeBufferPointer { weight in
            output.withUnsafeMutableBufferPointer { destination in
              work.withUnsafeMutableBufferPointer { scratch in
                var first = 0
                while first < count {
                  let run = min(span, count - first)
                  Self.blendSpan(
                    mine: mine.baseAddress! + first * 4, theirs: theirs.baseAddress! + first * 4,
                    weight: weight.baseAddress! + first,
                    into: destination.baseAddress! + first * 4,
                    span: VectorSpan(work: scratch.baseAddress!, count: run))
                  first += run
                }
              }
            }
          }
        }
      }
      return RGBABuffer(width: width, height: height, bytes: output)
    }

    /// One `blended` span: each of the three color channels is
    /// de-interleaved into a float plane, lerped, and written back
    /// interleaved. Uses four of `span`'s scratch planes.
    private static func blendSpan(
      mine: UnsafePointer<UInt8>, theirs: UnsafePointer<UInt8>, weight: UnsafePointer<Float>,
      into output: UnsafeMutablePointer<UInt8>, span: VectorSpan
    ) {
      let size = span.size
      let (reference, healed) = (span.plane(0), span.plane(1))
      let (inverse, blend) = (span.plane(2), span.plane(3))
      var (low, high, half, one, negated) = (Float(0), Float(255), Float(0.5), Float(1), Float(-1))
      vDSP_vsmsa(weight, 1, &negated, &one, inverse, 1, size)
      for channel in 0..<3 {
        vDSP_vfltu8(mine + channel, 4, reference, 1, size)
        vDSP_vfltu8(theirs + channel, 4, healed, 1, size)
        vDSP_vmma(reference, 1, inverse, 1, healed, 1, weight, 1, blend, 1, size)
        vDSP_vclip(blend, 1, &low, &high, blend, 1, size)
        // Adding a half and truncating is exactly `.rounded()` (which
        // rounds halves away from zero) on the clamped, non-negative
        // value, so this reproduces the scalar `UInt8` conversion.
        vDSP_vsadd(blend, 1, &half, blend, 1, size)
        vDSP_vfixu8(blend, 1, output + channel, 4, size)
      }
    }

    /// Builds a `CGImage` directly from this buffer's raw bytes — row-
    /// major, top-left origin, exactly as stored: `rgbaBuffer(of:extent:)`
    /// already produced `bytes` in that order (see its docs), so no
    /// orientation correction is needed here either.
    func makeImage() -> CGImage? {
      var bytesCopy = bytes
      guard let srgb = CGColorSpace(name: CGColorSpace.sRGB),
        let provider = CGDataProvider(
          data: Data(bytes: &bytesCopy, count: bytesCopy.count) as CFData)
      else { return nil }
      return CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: srgb,
        // Explicit byte order, matching `rgbaBuffer(of:extent:)` and
        // `ImageUtilities.rgbaPixels(of:)` — an implicit order could
        // reinterpret the channel layout (swapping colors and, worse,
        // coverage alpha) on the way back into Core Graphics.
        bitmapInfo: CGBitmapInfo(
          rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Derives this buffer's gray+coverage representation directly from
    /// its own RGBA bytes, with no extra Core Image render pass. The three
    /// color channels are de-interleaved straight into the weighted sum,
    /// one strided byte-to-float conversion each, accumulated in the same
    /// left-to-right order the scalar expression used.
    func grayMap() -> GrayMap {
      let count = width * height
      var values = [Float](repeating: 0, count: count)
      var coverage = [Bool](repeating: false, count: count)
      var channel = [Float](repeating: 0, count: count)
      bytes.withUnsafeBufferPointer { source in
        let base = source.baseAddress!
        let size = vDSP_Length(count)
        values.withUnsafeMutableBufferPointer { gray in
          channel.withUnsafeMutableBufferPointer { plane in
            let (out, band) = (gray.baseAddress!, plane.baseAddress!)
            // `cv2.cvtColor(..., COLOR_BGR2GRAY)`'s weights -- matched
            // exactly so the brightness thresholds below mean what
            // `stitch.py` intends.
            var (red, green, blue) = (Float(0.299), Float(0.587), Float(0.114))
            vDSP_vfltu8(base, 4, out, 1, size)
            vDSP_vsmul(out, 1, &red, out, 1, size)
            vDSP_vfltu8(base + 1, 4, band, 1, size)
            vDSP_vsma(band, 1, &green, out, 1, out, 1, size)
            vDSP_vfltu8(base + 2, 4, band, 1, size)
            vDSP_vsma(band, 1, &blue, out, 1, out, 1, size)
          }
        }
        // Coverage stays a scalar pass: it is one byte compare per pixel
        // into a `[Bool]`, with nothing for vDSP to accumulate.
        let threshold = UInt8(RGBABuffer.coveringAlpha)
        coverage.withUnsafeMutableBufferPointer { covered in
          for pixel in 0..<count { covered[pixel] = base[pixel * 4 + 3] >= threshold }
        }
      }
      return GrayMap(width: width, height: height, values: values, coverage: coverage)
    }
  }

  /// The gain-corrected min-composite: the darkest RGB any covering frame
  /// observed at each pixel, falling back to `reference` wherever no
  /// frame covers at all. Mirrors `stitch.py`'s
  /// `build_gain_corrected_min_composite`, and now literally shares its
  /// white-seeded shape: every frame's non-covering pixels are lifted to
  /// white before the running minimum, which is what lets the whole
  /// per-pixel coverage conditional become a branch-free vDSP pass (a
  /// white sample can never win a minimum against a real one, and the
  /// pixels no frame covered at all are selected back to the reference at
  /// the end). Runs a span at a time; see `vectorSpanPixels`.
  static func buildMinStack(reference: RGBABuffer, frames: [RGBABuffer]) -> RGBABuffer {
    guard !frames.isEmpty else { return reference }
    let count = reference.width * reference.height
    // Seeded from the reference so the alpha channel (which no pass below
    // writes) carries the reference's own coverage, as it always has.
    var output = reference.bytes
    let span = min(count, vectorSpanPixels)
    var work = [Float](repeating: 0, count: 7 * span)
    output.withUnsafeMutableBufferPointer { destination in
      work.withUnsafeMutableBufferPointer { scratch in
        var first = 0
        while first < count {
          let run = min(span, count - first)
          minStackSpan(
            reference: reference, frames: frames, first: first,
            into: destination.baseAddress! + first * 4,
            span: VectorSpan(work: scratch.baseAddress!, count: run))
          first += run
        }
      }
    }
    return RGBABuffer(width: reference.width, height: reference.height, bytes: output)
  }

  /// One `buildMinStack` span, using seven of `span`'s scratch planes: the
  /// current frame's "missing" mask, the running "no frame has covered this
  /// yet" mask, two working planes, and the three running per-channel
  /// minima (planes 4, 5 and 6).
  private static func minStackSpan(
    reference: RGBABuffer, frames: [RGBABuffer], first: Int,
    into output: UnsafeMutablePointer<UInt8>, span: VectorSpan
  ) {
    let size = span.size
    let (missing, unseen) = (span.plane(0), span.plane(1))
    let (sample, blend) = (span.plane(2), span.plane(3))
    var (white, one, negated) = (Float(255), Float(1), Float(-1))
    vDSP_vfill(&white, unseen, 1, size)
    for plane in 0..<3 { vDSP_vfill(&white, span.plane(4 + plane), 1, size) }
    for frame in frames {
      frame.bytes.withUnsafeBufferPointer { source in
        let base = source.baseAddress! + first * 4
        // `missing` is 0 where this frame's own alpha says it covers and
        // 255 where it doesn't -- the coverage conditional, as a value.
        var (covering, magnitude, lift) = (RGBABuffer.coveringAlpha, Float(-127.5), Float(127.5))
        vDSP_vfltu8(base + 3, 4, missing, 1, size)
        vDSP_vlim(missing, 1, &covering, &magnitude, missing, 1, size)
        vDSP_vsadd(missing, 1, &lift, missing, 1, size)
        vDSP_vmin(unseen, 1, missing, 1, unseen, 1, size)
        for plane in 0..<3 {
          let darkest = span.plane(4 + plane)
          vDSP_vfltu8(base + plane, 4, sample, 1, size)
          vDSP_vmax(sample, 1, missing, 1, sample, 1, size)
          vDSP_vmin(darkest, 1, sample, 1, darkest, 1, size)
        }
      }
    }
    // `unseen` becomes an exact 0/1 selector (and `missing`, now free, its
    // complement), so the fallback is a plain two-term multiply-add.
    vDSP_vsdiv(unseen, 1, &white, unseen, 1, size)
    vDSP_vsmsa(unseen, 1, &negated, &one, missing, 1, size)
    reference.bytes.withUnsafeBufferPointer { source in
      let base = source.baseAddress! + first * 4
      for plane in 0..<3 {
        vDSP_vfltu8(base + plane, 4, sample, 1, size)
        vDSP_vmma(span.plane(4 + plane), 1, missing, 1, sample, 1, unseen, 1, blend, 1, size)
        vDSP_vfixu8(blend, 1, output + plane, 4, size)
      }
    }
  }

  /// Renders `image` (already sized to `extent`) to a top-left-origin,
  /// row-major sRGB RGBA8 `RGBABuffer` — the masked-healing math's working
  /// representation, read as plain encoded bytes rather than through any
  /// of Core Image's own (linear) compositing filters.
  ///
  /// Goes through a `CGImage` first (Core Image's own `createCGImage`,
  /// already used for this composer's final output, which correctly
  /// orients regardless of Core Image's internal coordinate convention)
  /// rather than `CIContext.render(_:toBitmap:...)` directly. Drawing a
  /// `CGImage` into a freshly created `CGContext` at `(0, 0, width,
  /// height)` with no transform applied already yields a top-left-origin,
  /// row-major buffer matching `CGImage`'s own byte layout -- verified
  /// directly (a known asymmetric fixture sampled back through both this
  /// function and `RGBABuffer.makeImage`) rather than assumed, since an
  /// extra flip here once silently produced a vertically mirrored
  /// composite that several coarse, high-tolerance tests still passed by
  /// coincidence.
  static func rgbaBuffer(of image: CIImage, extent: CGRect) -> RGBABuffer? {
    let width = Int(extent.width.rounded())
    let height = Int(extent.height.rounded())
    guard width > 0, height > 0,
      let cgImage = context.createCGImage(image, from: extent),
      let srgb = CGColorSpace(name: CGColorSpace.sRGB),
      let bitmapContext = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: srgb,
        // Explicit byte order (same convention as ImageUtilities
        // .rgbaPixels(of:)) so `bytes` is unambiguously R,G,B,A in memory.
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          | CGBitmapInfo.byteOrder32Big.rawValue)
    else { return nil }
    bitmapContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let data = bitmapContext.data else { return nil }
    let count = width * height * 4
    let bytes = [UInt8](
      UnsafeBufferPointer(start: data.bindMemory(to: UInt8.self, capacity: count), count: count))
    return RGBABuffer(width: width, height: height, bytes: bytes)
  }

  /// The pixel size `extent` downscales to so its long side is at most
  /// `longSide`, preserving aspect ratio — mirrors `registrationProxies`'s
  /// own downscale-size rule.
  static func downscaledSize(of extent: CGRect, longSide: CGFloat) -> (width: Int, height: Int) {
    let scale = min(1, longSide / max(extent.width, extent.height))
    let width = max(1, Int((extent.width * scale).rounded(.down)))
    let height = max(1, Int((extent.height * scale).rounded(.down)))
    return (width, height)
  }

  /// Renders `image` to a `GrayMap` at `size`, downscaling first (via the
  /// same `CIImage.transformed` idiom `registrationProxies` uses) when
  /// `size` is smaller than `extent` — the masked-healing mask math runs
  /// at a coarser resolution than compositing (see `maskMaxDimension`).
  static func grayMap(
    of image: CIImage, extent: CGRect, downscaledTo size: (width: Int, height: Int)
  ) -> GrayMap? {
    let scaled: CIImage
    let renderExtent: CGRect
    if size.width == Int(extent.width.rounded()), size.height == Int(extent.height.rounded()) {
      scaled = image
      renderExtent = extent
    } else {
      renderExtent = CGRect(x: 0, y: 0, width: size.width, height: size.height)
      scaled = image.transformed(
        by: CGAffineTransform(
          scaleX: CGFloat(size.width) / extent.width, y: CGFloat(size.height) / extent.height))
    }
    guard let buffer = rgbaBuffer(of: scaled, extent: renderExtent) else { return nil }
    return buffer.grayMap()
  }

  /// Bilinear-upscales a flattened `fromWidth`x`fromHeight` map to
  /// `toWidth`x`toHeight` — used to bring the mask-resolution glare alpha
  /// up to compositing resolution before the final blend.
  ///
  /// Separated into the same two stages the per-pixel version evaluated in
  /// (horizontal first, then vertical), but a whole row at a time. Every
  /// output column reads the same pair of source columns with the same
  /// pair of weights, so those are computed once and each source row is
  /// resampled horizontally with two `vDSP_vgathr` gathers and one
  /// multiply-add; the vertical stage is then a plain lerp between two
  /// already-horizontal rows.
  static func upscale(
    _ values: [Float], fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int
  ) -> [Float] {
    guard fromWidth != toWidth || fromHeight != toHeight else { return values }
    var left = [vDSP_Length](repeating: 0, count: toWidth)
    var right = [vDSP_Length](repeating: 0, count: toWidth)
    var fraction = [Float](repeating: 0, count: toWidth)
    var inverse = [Float](repeating: 0, count: toWidth)
    let scaleX = Float(fromWidth) / Float(toWidth)
    for x in 0..<toWidth {
      let srcX = (Float(x) + 0.5) * scaleX - 0.5
      let x0 = max(0, min(fromWidth - 1, Int(floor(srcX))))
      let x1 = min(fromWidth - 1, x0 + 1)
      let fracX = max(0, min(1, srcX - Float(x0)))
      // `vDSP_vgathr` indexes from 1, relative to the row it is handed.
      (left[x], right[x]) = (vDSP_Length(x0 + 1), vDSP_Length(x1 + 1))
      (fraction[x], inverse[x]) = (fracX, 1 - fracX)
    }
    var rows = [Float](repeating: 0, count: fromHeight * toWidth)
    var gathered = [Float](repeating: 0, count: 2 * toWidth)
    let width = vDSP_Length(toWidth)
    values.withUnsafeBufferPointer { source in
      rows.withUnsafeMutableBufferPointer { resampled in
        gathered.withUnsafeMutableBufferPointer { pair in
          let (nearer, further) = (pair.baseAddress!, pair.baseAddress! + toWidth)
          for y in 0..<fromHeight {
            let row = source.baseAddress! + y * fromWidth
            vDSP_vgathr(row, left, 1, nearer, 1, width)
            vDSP_vgathr(row, right, 1, further, 1, width)
            vDSP_vmma(
              nearer, 1, inverse, 1, further, 1, fraction, 1,
              resampled.baseAddress! + y * toWidth, 1, width)
          }
        }
      }
    }
    var output = [Float](repeating: 0, count: toWidth * toHeight)
    let scaleY = Float(fromHeight) / Float(toHeight)
    rows.withUnsafeBufferPointer { resampled in
      output.withUnsafeMutableBufferPointer { destination in
        for y in 0..<toHeight {
          let srcY = (Float(y) + 0.5) * scaleY - 0.5
          let y0 = max(0, min(fromHeight - 1, Int(floor(srcY))))
          let y1 = min(fromHeight - 1, y0 + 1)
          var fracY = max(0, min(1, srcY - Float(y0)))
          var inverseY = 1 - fracY
          let target = destination.baseAddress! + y * toWidth
          vDSP_vsmul(resampled.baseAddress! + y0 * toWidth, 1, &inverseY, target, 1, width)
          vDSP_vsma(resampled.baseAddress! + y1 * toWidth, 1, &fracY, target, 1, target, 1, width)
        }
      }
    }
    return output
  }

  /// A single-channel sRGB-encoded gray map (`cv2.cvtColor(...,
  /// COLOR_BGR2GRAY)`'s weights: 0.299R + 0.587G + 0.114B), row-major,
  /// top-left-origin, plus a per-pixel coverage flag. The masked-healing
  /// math below works entirely in this domain to match
  /// `scripts/stitch_quality/stitch.py`'s numpy/cv2 arithmetic on raw
  /// JPEG bytes — deliberately not Core Image's (linear) working color
  /// space, so `maskDarkeningThreshold`/`maskBrightnessFloor`/
  /// `gainGrayRange` mean the same thing here as they do in the Python
  /// reference. Not private, unlike most of this file's plumbing, so
  /// tests can build cases directly (mirrors `seedMatrix`'s reasoning).
  struct GrayMap {
    let width: Int
    let height: Int
    var values: [Float]
    var coverage: [Bool]

    /// A copy with every value multiplied by `gain` and clamped to the
    /// 0-255 byte range, mirroring `applyGain` (and `stitch.py`'s
    /// `np.clip(warped * gain, 0, 255)`) so the mask math sees the same
    /// clipped values as the composited color buffers. Gray is a linear
    /// combination of RGB and gain multiplies every channel identically,
    /// so below saturation this is exactly `gray(warped * gain)` without
    /// re-rendering the gain-corrected color image just to regray it;
    /// once a channel would saturate, clamping the gray instead of each
    /// channel differs only for near-white pixels at gain > 1, which the
    /// darkening signal (reference minus a BRIGHT robust gray) doesn't
    /// meaningfully read anyway.
    func scaled(by gain: Float) -> GrayMap {
      var result = self
      var (multiplier, low, high) = (gain, Float(0), Float(255))
      result.values.withUnsafeMutableBufferPointer { plane in
        let (base, size) = (plane.baseAddress!, vDSP_Length(plane.count))
        vDSP_vsmul(base, 1, &multiplier, base, 1, size)
        vDSP_vclip(base, 1, &low, &high, base, 1, size)
      }
      return result
    }
  }

  /// Bundles `glareAlpha`'s tunable thresholds so the function itself
  /// stays within a reasonable parameter count; see the matching
  /// `GlareFreeComposer.maskDarkeningThreshold` etc. constants for what
  /// each field means and how it's tuned. `blurSigma`/`dilateRadius`/
  /// `featherSigma` are all in pixels at `reference`'s own resolution.
  struct MaskTuning {
    let darkeningThreshold: Float
    let brightnessFloor: Float
    let blurSigma: Float
    let dilateRadius: Float
    let featherSigma: Float
  }

  /// `coverage`'s per-pixel flags as a 0/1 float plane, so the coverage
  /// conditionals below can be masked selects instead of branches. `Bool`
  /// occupies a single byte, and thresholding at a half after the integer
  /// conversion keeps that independent of which non-zero byte value the
  /// runtime happens to store for `true`.
  private static func coverageIndicator(
    _ coverage: [Bool], into output: UnsafeMutablePointer<Float>, count: Int
  ) {
    let size = vDSP_Length(count)
    var half = Float(0.5)
    coverage.withUnsafeBufferPointer { flags in
      flags.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: count) { raw in
        vDSP_vfltu8(raw, 1, output, 1, size)
      }
    }
    vDSP_vlim(output, 1, &half, &half, output, 1, size)
    vDSP_vsadd(output, 1, &half, output, 1, size)
  }

  /// Median reference/warped gray ratio over covered, mid-tone pixels --
  /// one frame's exposure gain. Corrects an exposure/white-balance
  /// mismatch between a corner shot and the reference before it enters
  /// the min-composite, so the darkest-pixel-wins selection can't be won
  /// by whichever frame happens to be globally darker rather than by
  /// which pixel actually has less glare. Mirrors `stitch.py`'s
  /// `compute_gain_factor`. Internal (not private) for direct unit
  /// testing.
  ///
  /// The qualifying-pixel scan stays scalar: it compacts a sparse,
  /// data-dependent subset into a dense array, which has no vector form,
  /// but it runs over raw pointers into a pre-sized buffer so neither
  /// bounds checks nor `append`'s uniqueness check is in the loop.
  static func gainFactor(reference: GrayMap, warped: GrayMap, range: ClosedRange<Float>) -> Float {
    var ratios = [Float](repeating: 0, count: warped.values.count)
    var kept = 0
    warped.values.withUnsafeBufferPointer { warpedValues in
      reference.values.withUnsafeBufferPointer { referenceValues in
        warped.coverage.withUnsafeBufferPointer { covered in
          ratios.withUnsafeMutableBufferPointer { output in
            for index in 0..<warpedValues.count {
              guard covered[index] else { continue }
              let warpedValue = warpedValues[index]
              let referenceValue = referenceValues[index]
              guard range.contains(warpedValue), range.contains(referenceValue) else { continue }
              output[kept] = referenceValue / max(warpedValue, 1)
              kept += 1
            }
          }
        }
      }
    }
    guard kept > 0 else { return 1.0 }
    return median(of: &ratios, count: kept)
  }

  /// Per-pixel darkening estimate for the glare MASK's own candidacy
  /// signal, robust to min-of-N order-statistic noise. Mirrors
  /// `stitch.py`'s `compute_darkening_robust` (see that function's
  /// docstring, and the benchmark README's "Mask signal tuning history",
  /// for the full rationale): a plain
  /// `max(0, reference - min_composite)` signal conflates real glare with
  /// order-statistics noise, since the min of several independent noisy
  /// samples of a high-variance texture (e.g. carpet) reads systematically
  /// lower than any one of them everywhere, with nothing to do with glare.
  ///
  /// An earlier iteration used the median across covering frames instead,
  /// but that still leaked on carpet: Vision's registration error is
  /// largest near the frame edges (the verification gate only checks the
  /// central 50%), so a border carpet pixel commonly had exactly 2 of 3
  /// (or 2 of 4) covering frames coincidentally agree on the same
  /// shifted-texture reading, and a plain median treats 2-of-N agreement
  /// as "most frames". The current rule is an explicit vote requiring
  /// broader agreement:
  ///
  /// - 2 or 3 covering frames: the MAX (brightest) covered gray -- an
  ///   ALL-of-N vote. Every covering frame must read darker than the
  ///   reference for the pixel to register as darkened; one bright frame
  ///   vetoes it. For 2 frames this matches the median-era rule (a
  ///   2-sample median is just their average, which would understate
  ///   darkening at a genuinely glared pixel where only one frame sees
  ///   through the glare, so max was already used there); for 3 frames
  ///   this TIGHTENS the median (effectively a 2-of-3 vote) into an
  ///   all-of-3 vote.
  /// - 4 covering frames: the SECOND-brightest covered gray -- a 3-of-4
  ///   vote. Four covering frames is common enough that requiring literal
  ///   unanimity would make the mask too eager to reject on one
  ///   coincidentally bright reading, so a single dissenting frame (e.g. a
  ///   corner shot that happens to share glare position with the
  ///   reference) is tolerated. A median-of-4 was, in effect, closer to
  ///   a 2-of-4 vote.
  ///
  /// Real glare survives this tightening because it moves between shots
  /// (each corner frame glares, if at all, at a different location), so a
  /// genuinely glared reference pixel typically has ALL (or all-but-one,
  /// at 4 frames) covering frames show the true darker surface. Fewer
  /// than 2 covering frames get 0 (no mask -- no meaningful vote at all).
  /// Internal (not private) for direct unit testing.
  ///
  /// The vote runs as whole-plane vDSP passes rather than a per-pixel
  /// branch, and reproduces the scalar version's semantics exactly: a
  /// non-covering frame contributes a sentinel far below any gray value
  /// (so it can never place in the top two), the running best/runner-up
  /// update becomes `runnerUp = max(runnerUp, min(best, sample))`, and
  /// the "exactly four covered" and "at least two covered" conditions
  /// become 0/1 selectors multiplied through at the end.
  static func robustDarkening(reference: GrayMap, correctedFrames: [GrayMap]) -> [Float] {
    let count = reference.values.count
    var darkening = [Float](repeating: 0, count: count)
    guard !correctedFrames.isEmpty else { return darkening }
    let size = vDSP_Length(count)
    var work = [Float](repeating: 0, count: 6 * count)
    work.withUnsafeMutableBufferPointer { scratch in
      let covered = scratch.baseAddress!
      let (masked, tally) = (covered + count, covered + 2 * count)
      let (best, runnerUp, spare) = (covered + 3 * count, covered + 4 * count, covered + 5 * count)
      // Finite, not -infinity: the count gate below multiplies the unused
      // `reference - sentinel` away, and `0 * infinity` would be a NaN.
      var absent = Float(-1e30)
      var rise = -absent
      vDSP_vfill(&absent, best, 1, size)
      vDSP_vfill(&absent, runnerUp, 1, size)
      for frame in correctedFrames {
        coverageIndicator(frame.coverage, into: covered, count: count)
        vDSP_vadd(covered, 1, tally, 1, tally, 1, size)
        frame.values.withUnsafeBufferPointer { values in
          vDSP_vmul(values.baseAddress!, 1, covered, 1, masked, 1, size)
        }
        vDSP_vsmsa(covered, 1, &rise, &absent, spare, 1, size)
        vDSP_vadd(masked, 1, spare, 1, masked, 1, size)
        vDSP_vmin(best, 1, masked, 1, spare, 1, size)
        vDSP_vmax(runnerUp, 1, spare, 1, runnerUp, 1, size)
        vDSP_vmax(best, 1, masked, 1, best, 1, size)
      }
      var (half, negativeHalf, one, negated) = (Float(0.5), Float(-0.5), Float(1), Float(-1))
      var (lessFour, two, zero) = (Float(-4), Float(2), Float(0))
      // `spare` becomes 1 exactly where four frames covered, `covered` its
      // complement, so the vote's two branches are one multiply-add.
      vDSP_vsadd(tally, 1, &lessFour, spare, 1, size)
      vDSP_vabs(spare, 1, spare, 1, size)
      vDSP_vlim(spare, 1, &half, &negativeHalf, spare, 1, size)
      vDSP_vsadd(spare, 1, &half, spare, 1, size)
      vDSP_vsmsa(spare, 1, &negated, &one, covered, 1, size)
      vDSP_vmma(runnerUp, 1, spare, 1, best, 1, covered, 1, masked, 1, size)
      reference.values.withUnsafeBufferPointer { values in
        vDSP_vsub(masked, 1, values.baseAddress!, 1, masked, 1, size)
      }
      vDSP_vthr(masked, 1, &zero, masked, 1, size)
      vDSP_vlim(tally, 1, &two, &half, spare, 1, size)
      vDSP_vsadd(spare, 1, &half, spare, 1, size)
      darkening.withUnsafeMutableBufferPointer { output in
        vDSP_vmul(masked, 1, spare, 1, output.baseAddress!, 1, size)
      }
    }
    return darkening
  }

  /// Feathered [0, 1] blend weight favoring the min-composite only in
  /// likely-glare regions. Mirrors `stitch.py`'s `compute_glare_alpha`:
  /// blur the darkening map, threshold it (AND a reference-brightness
  /// floor, to keep moving shadows out of the mask -- glare heals FROM a
  /// bright, washed-out reference pixel, a shadow doesn't), dilate to
  /// cover a glare patch's soft rim, then feather into a smooth alpha. See
  /// `MaskTuning`'s docs for the resolution `tuning`'s spatial fields are
  /// expected at. Internal (not private) for direct unit testing.
  ///
  /// Both thresholds are compared with `nextUp` limits: `vDSP_vlim`'s test
  /// is `>=`, and the smallest representable step above the limit turns
  /// that back into the strict `>` the scalar comparisons used, exactly.
  static func glareAlpha(reference: GrayMap, darkeningRobust: [Float], tuning: MaskTuning)
    -> [Float]
  {
    let (width, height) = (reference.width, reference.height)
    let count = width * height
    let size = vDSP_Length(count)
    var rawMask = gaussianBlur(
      darkeningRobust, width: width, height: height, sigma: tuning.blurSigma)
    var brightEnough = [Float](repeating: 0, count: count)
    var (darkeningLimit, brightnessLimit) = (
      tuning.darkeningThreshold.nextUp, tuning.brightnessFloor.nextUp
    )
    var (half, full, low, high) = (Float(0.5), Float(255), Float(0), Float(1))
    rawMask.withUnsafeMutableBufferPointer { mask in
      brightEnough.withUnsafeMutableBufferPointer { bright in
        let (candidate, floor) = (mask.baseAddress!, bright.baseAddress!)
        vDSP_vlim(candidate, 1, &darkeningLimit, &half, candidate, 1, size)
        vDSP_vsadd(candidate, 1, &half, candidate, 1, size)
        vDSP_vlim(reference.values, 1, &brightnessLimit, &half, floor, 1, size)
        vDSP_vsadd(floor, 1, &half, floor, 1, size)
        vDSP_vmul(candidate, 1, floor, 1, candidate, 1, size)
        vDSP_vsmul(candidate, 1, &full, candidate, 1, size)
      }
    }
    let dilated = dilateMax(rawMask, width: width, height: height, radius: tuning.dilateRadius)
    var feathered = gaussianBlur(
      dilated, width: width, height: height, sigma: tuning.featherSigma)
    feathered.withUnsafeMutableBufferPointer { alpha in
      let base = alpha.baseAddress!
      vDSP_vsdiv(base, 1, &full, base, 1, size)
      vDSP_vclip(base, 1, &low, &high, base, 1, size)
    }
    return feathered
  }

  /// The median of `values`' first `count` entries; numpy's convention for
  /// an even count (the average of the two middle elements) so this
  /// matches `np.median` exactly. Selected via quickselect rather than a
  /// full sort: the gain estimate feeds this a large fraction of the
  /// mask-resolution pixels per verified frame, where an O(n log n) sort
  /// (plus its full-size sorted copy) is measurable capture-time work for
  /// two order statistics.
  private static func median(of values: inout [Float], count: Int) -> Float {
    guard count > 0 else { return 0 }
    return values.withUnsafeMutableBufferPointer { buffer in
      let region = UnsafeMutableBufferPointer(rebasing: buffer[0..<count])
      let upper = quickselect(region, rank: count / 2)
      if count % 2 == 1 { return upper }
      // Quickselect leaves everything below index k partitioned <= the kth
      // order statistic, so the lower middle element is that prefix's max.
      var lower = upper
      vDSP_maxv(region.baseAddress!, 1, &lower, vDSP_Length(count / 2))
      return (lower + upper) / 2
    }
  }

  /// The `k`th-smallest element of `buffer` (0-based) via iterative
  /// Hoare-partition quickselect with a median-of-three pivot, expected
  /// O(n) with no allocations. Postcondition: `buffer[..<k]` holds only
  /// values `<=` the returned element.
  private static func quickselect(_ buffer: UnsafeMutableBufferPointer<Float>, rank: Int) -> Float {
    var low = 0
    var high = buffer.count - 1
    while low < high {
      // Median-of-three pivot, guarding against already-ordered input.
      let mid = low + (high - low) / 2
      if buffer[mid] < buffer[low] { buffer.swapAt(mid, low) }
      if buffer[high] < buffer[low] { buffer.swapAt(high, low) }
      if buffer[high] < buffer[mid] { buffer.swapAt(high, mid) }
      let pivot = buffer[mid]
      var left = low
      var right = high
      while left <= right {
        while buffer[left] < pivot { left += 1 }
        while buffer[right] > pivot { right -= 1 }
        if left <= right {
          buffer.swapAt(left, right)
          left += 1
          right -= 1
        }
      }
      if rank <= right {
        high = right
      } else if rank >= left {
        low = left
      } else {
        break
      }
    }
    return buffer[rank]
  }

  /// `values` widened by `radius` clamped samples on each side of every
  /// row (`vertical == false`) or by `radius` clamped rows above and below
  /// (`vertical == true`). With the boundary materialized this way, one
  /// separable pass is just a fixed shift of the whole padded buffer per
  /// kernel tap, with no per-sample index clamping left to do -- exactly
  /// the shape vDSP wants, and the reason the two filters below are a
  /// handful of whole-buffer calls instead of a triple loop.
  private static func padded(
    _ values: [Float], width: Int, height: Int, radius: Int, vertical: Bool
  ) -> [Float] {
    let stride = vertical ? width : width + 2 * radius
    let rows = vertical ? height + 2 * radius : height
    var output = [Float](repeating: 0, count: stride * rows)
    output.withUnsafeMutableBufferPointer { padding in
      values.withUnsafeBufferPointer { source in
        for row in 0..<rows {
          let line =
            source.baseAddress! + min(max(vertical ? row - radius : row, 0), height - 1) * width
          let target = padding.baseAddress! + row * stride
          (target + (vertical ? 0 : radius)).update(from: line, count: width)
          guard !vertical else { continue }
          for index in 0..<radius {
            target[index] = line[0]
            target[radius + width + index] = line[width - 1]
          }
        }
      }
    }
    return output
  }

  /// Separable Gaussian blur of a flattened `width`x`height` map,
  /// clamped-edge boundary (`stitch.py`'s `cv2.GaussianBlur` defaults to
  /// `BORDER_REFLECT_101`; clamped-edge is a minor, deliberate
  /// simplification here -- both only affect a thin border strip, and this
  /// pipeline's masks are never candidates near the frame edge anyway,
  /// since nothing but the reference itself covers a warped frame's own
  /// margin). A no-op for `sigma <= 0`.
  private static func gaussianBlur(_ values: [Float], width: Int, height: Int, sigma: Float)
    -> [Float]
  {
    guard sigma > 0 else { return values }
    let radius = max(1, Int((sigma * 3).rounded()))
    var kernel = [Float](repeating: 0, count: 2 * radius + 1)
    var sum: Float = 0
    for offset in -radius...radius {
      let weight = exp(-Float(offset * offset) / (2 * sigma * sigma))
      kernel[offset + radius] = weight
      sum += weight
    }
    for index in kernel.indices { kernel[index] /= sum }
    let horizontal = blurPass(values, width: width, height: height, kernel: kernel, vertical: false)
    return blurPass(horizontal, width: width, height: height, kernel: kernel, vertical: true)
  }

  /// One separable blur pass: each output row accumulated from the padded
  /// buffer once per kernel tap, ascending, so every output sample sums
  /// its taps in the same order the scalar inner loop did. A tap is a
  /// fixed shift of `step` elements — one sample along a padded row, one
  /// whole row down a padded column — which is what makes the whole pass
  /// a handful of `vDSP_vsma` calls with no index clamping left in it.
  private static func blurPass(
    _ values: [Float], width: Int, height: Int, kernel: [Float], vertical: Bool
  ) -> [Float] {
    let radius = (kernel.count - 1) / 2
    let source = padded(values, width: width, height: height, radius: radius, vertical: vertical)
    let stride = vertical ? width : width + 2 * radius
    let step = vertical ? width : 1
    var output = [Float](repeating: 0, count: width * height)
    source.withUnsafeBufferPointer { input in
      output.withUnsafeMutableBufferPointer { blurred in
        let size = vDSP_Length(width)
        for row in 0..<height {
          let line = input.baseAddress! + row * stride
          let target = blurred.baseAddress! + row * width
          for tap in 0...(2 * radius) {
            var weight = kernel[tap]
            vDSP_vsma(line + tap * step, 1, &weight, target, 1, target, 1, size)
          }
        }
      }
    }
    return output
  }

  /// Grayscale dilation (max filter) of a flattened `width`x`height` map
  /// over a `radius`-pixel square neighborhood, as two separable 1D max
  /// passes. `stitch.py`'s `compute_glare_alpha` dilates with a
  /// `cv2.MORPH_ELLIPSE` kernel; a square footprint is a deliberate
  /// simplification (larger corners, same reach along both axes) that
  /// turns an O(width * height * radius²) circular dilation into two O(width
  /// * height * radius) passes -- meaningful here, since this runs on
  /// every composite, including from a plain XCTest host without
  /// optimized-build guarantees.
  private static func dilateMax(_ values: [Float], width: Int, height: Int, radius: Float)
    -> [Float]
  {
    let pixelRadius = Int(radius.rounded())
    guard pixelRadius > 0 else { return values }
    let horizontal = maxPass(
      values, width: width, height: height, radius: pixelRadius, vertical: false)
    return maxPass(horizontal, width: width, height: height, radius: pixelRadius, vertical: true)
  }

  /// One separable max pass. Clamping the boundary (rather than clipping
  /// the kernel at it, as the scalar loop did) is exactly equivalent for a
  /// max filter -- a clamped sample only repeats a value the clipped
  /// window already contained -- so this is the same result from a running
  /// `vDSP_vmax` over the padded buffer, one tap at a time.
  private static func maxPass(
    _ values: [Float], width: Int, height: Int, radius: Int, vertical: Bool
  ) -> [Float] {
    let source = padded(values, width: width, height: height, radius: radius, vertical: vertical)
    let stride = vertical ? width : width + 2 * radius
    let step = vertical ? width : 1
    var output = [Float](repeating: 0, count: width * height)
    source.withUnsafeBufferPointer { input in
      output.withUnsafeMutableBufferPointer { dilated in
        let size = vDSP_Length(width)
        var zero = Float(0)
        for row in 0..<height {
          let line = input.baseAddress! + row * stride
          let target = dilated.baseAddress! + row * width
          target.update(from: line, count: width)
          for tap in 1...(2 * radius) {
            vDSP_vmax(target, 1, line + tap * step, 1, target, 1, size)
          }
          // The scalar filter seeded its running best at 0, so floor here.
          vDSP_vthr(target, 1, &zero, target, 1, size)
        }
      }
    }
    return output
  }
}
