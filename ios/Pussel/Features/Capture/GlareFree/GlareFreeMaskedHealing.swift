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
extension GlareFreeComposer {
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
  /// with plain array arithmetic, deliberately bypassing Core Image's own
  /// (linear) compositing filters — see `healedComposite`'s docs for why.
  struct RGBABuffer {
    let width: Int
    let height: Int
    /// `width * height * 4` bytes, RGBA, top-left-origin, row-major.
    var bytes: [UInt8]

    /// Whether pixel `index` (0..<width*height) is inside this frame's own
    /// warped footprint, read from its own alpha channel
    /// (`CIFilter.perspectiveTransform`'s output is transparent outside
    /// the mapped quad). A firm threshold rather than `> 0` sidesteps a
    /// thin ring of partial-alpha edge pixels the warp's resampling can
    /// leave right at the quad boundary.
    func covers(_ index: Int) -> Bool { bytes[index * 4 + 3] >= 128 }

    /// Multiplies RGB by `gain`, clamped to a valid byte range; leaves
    /// alpha (coverage) untouched. Mirrors `stitch.py`'s
    /// `compute_gain_factor` application: `np.clip(warped * gain, 0, 255)`.
    mutating func applyGain(_ gain: Float) {
      guard gain != 1 else { return }
      for pixel in 0..<(width * height) {
        let base = pixel * 4
        for channel in 0..<3 {
          let value = Float(bytes[base + channel]) * gain
          bytes[base + channel] = UInt8(max(0, min(255, value)).rounded())
        }
      }
    }

    /// Per-channel linear interpolation toward `other`, weighted by
    /// `alpha` (one entry per pixel, in [0, 1]):
    /// `self * (1 - alpha) + other * alpha`. Mirrors `stitch.py`'s final
    /// `reference * (1 - alpha) + min_composite * alpha` blend. Alpha
    /// stays fully opaque in the result — the masked-healing composite is
    /// always meant to be viewed as a normal opaque photo.
    func blended(with other: RGBABuffer, alpha: [Float]) -> RGBABuffer {
      var result = self
      for pixel in 0..<(width * height) {
        let base = pixel * 4
        let weight = alpha[pixel]
        for channel in 0..<3 {
          let mine = Float(bytes[base + channel])
          let theirs = Float(other.bytes[base + channel])
          result.bytes[base + channel] = UInt8(
            max(0, min(255, mine * (1 - weight) + theirs * weight)).rounded())
        }
        result.bytes[base + 3] = 255
      }
      return result
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
    /// its own RGBA bytes, with no extra Core Image render pass.
    func grayMap() -> GrayMap {
      var values = [Float](repeating: 0, count: width * height)
      var coverage = [Bool](repeating: false, count: width * height)
      for pixel in 0..<values.count {
        let base = pixel * 4
        let red = Float(bytes[base])
        let green = Float(bytes[base + 1])
        let blue = Float(bytes[base + 2])
        // `cv2.cvtColor(..., COLOR_BGR2GRAY)`'s weights -- matched exactly
        // so the brightness thresholds below mean what `stitch.py` intends.
        values[pixel] = 0.299 * red + 0.587 * green + 0.114 * blue
        coverage[pixel] = bytes[base + 3] >= 128
      }
      return GrayMap(width: width, height: height, values: values, coverage: coverage)
    }
  }

  /// The gain-corrected min-composite: the darkest RGB any covering frame
  /// observed at each pixel, falling back to `reference` wherever no
  /// frame covers at all. Mirrors `stitch.py`'s
  /// `build_gain_corrected_min_composite` (its white-seeded running
  /// minimum is mathematically the same thing: the first covering frame's
  /// own value can only ever be <= white, so seeding directly with it
  /// instead is equivalent and skips a redundant comparison).
  static func buildMinStack(reference: RGBABuffer, frames: [RGBABuffer]) -> RGBABuffer {
    var stack = reference
    var coveredAny = [Bool](repeating: false, count: reference.width * reference.height)
    for frame in frames {
      for pixel in 0..<(reference.width * reference.height) {
        guard frame.covers(pixel) else { continue }
        let base = pixel * 4
        if coveredAny[pixel] {
          for channel in 0..<3 {
            stack.bytes[base + channel] = min(
              stack.bytes[base + channel], frame.bytes[base + channel])
          }
        } else {
          for channel in 0..<3 {
            stack.bytes[base + channel] = frame.bytes[base + channel]
          }
          coveredAny[pixel] = true
        }
      }
    }
    return stack
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
  static func upscale(
    _ values: [Float], fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int
  ) -> [Float] {
    guard fromWidth != toWidth || fromHeight != toHeight else { return values }
    var output = [Float](repeating: 0, count: toWidth * toHeight)
    let scaleX = Float(fromWidth) / Float(toWidth)
    let scaleY = Float(fromHeight) / Float(toHeight)
    for y in 0..<toHeight {
      let srcY = (Float(y) + 0.5) * scaleY - 0.5
      let y0 = max(0, min(fromHeight - 1, Int(floor(srcY))))
      let y1 = min(fromHeight - 1, y0 + 1)
      let fracY = max(0, min(1, srcY - Float(y0)))
      for x in 0..<toWidth {
        let srcX = (Float(x) + 0.5) * scaleX - 0.5
        let x0 = max(0, min(fromWidth - 1, Int(floor(srcX))))
        let x1 = min(fromWidth - 1, x0 + 1)
        let fracX = max(0, min(1, srcX - Float(x0)))
        let topLeft = values[y0 * fromWidth + x0]
        let topRight = values[y0 * fromWidth + x1]
        let bottomLeft = values[y1 * fromWidth + x0]
        let bottomRight = values[y1 * fromWidth + x1]
        let top = topLeft * (1 - fracX) + topRight * fracX
        let bottom = bottomLeft * (1 - fracX) + bottomRight * fracX
        output[y * toWidth + x] = top * (1 - fracY) + bottom * fracY
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
      result.values = values.map { max(0, min(255, $0 * gain)) }
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

  /// Median reference/warped gray ratio over covered, mid-tone pixels --
  /// one frame's exposure gain. Corrects an exposure/white-balance
  /// mismatch between a corner shot and the reference before it enters
  /// the min-composite, so the darkest-pixel-wins selection can't be won
  /// by whichever frame happens to be globally darker rather than by
  /// which pixel actually has less glare. Mirrors `stitch.py`'s
  /// `compute_gain_factor`. Internal (not private) for direct unit
  /// testing.
  static func gainFactor(reference: GrayMap, warped: GrayMap, range: ClosedRange<Float>) -> Float {
    var ratios: [Float] = []
    ratios.reserveCapacity(warped.values.count)
    for index in 0..<warped.values.count {
      guard warped.coverage[index] else { continue }
      let warpedValue = warped.values[index]
      let referenceValue = reference.values[index]
      guard range.contains(warpedValue), range.contains(referenceValue) else { continue }
      ratios.append(referenceValue / max(warpedValue, 1))
    }
    guard !ratios.isEmpty else { return 1.0 }
    return median(of: ratios)
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
  static func robustDarkening(reference: GrayMap, correctedFrames: [GrayMap]) -> [Float] {
    var darkening = [Float](repeating: 0, count: reference.values.count)
    guard !correctedFrames.isEmpty else { return darkening }
    for index in 0..<reference.values.count {
      // Track the two brightest covering samples in a single pass — this
      // runs per pixel at mask resolution, where a per-pixel sort's
      // allocations would be a real CPU cost during capture.
      var count = 0
      var maxGray = -Float.infinity
      var secondMaxGray = -Float.infinity
      for frame in correctedFrames where frame.coverage[index] {
        let value = frame.values[index]
        count += 1
        if value > maxGray {
          secondMaxGray = maxGray
          maxGray = value
        } else if value > secondMaxGray {
          secondMaxGray = value
        }
      }
      guard count >= 2 else { continue }
      // 4 covering frames: second-brightest (3-of-4 vote). 2 or 3: the
      // brightest (all-of-N vote) -- see the docs above for why.
      let robustGray = count == 4 ? secondMaxGray : maxGray
      darkening[index] = max(0, reference.values[index] - robustGray)
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
  static func glareAlpha(reference: GrayMap, darkeningRobust: [Float], tuning: MaskTuning)
    -> [Float]
  {
    let width = reference.width
    let height = reference.height
    let blurredDarkening = gaussianBlur(
      darkeningRobust, width: width, height: height, sigma: tuning.blurSigma)
    var rawMask = [Float](repeating: 0, count: width * height)
    for index in 0..<rawMask.count {
      if blurredDarkening[index] > tuning.darkeningThreshold,
        reference.values[index] > tuning.brightnessFloor
      {
        rawMask[index] = 255
      }
    }
    let dilated = dilateMax(rawMask, width: width, height: height, radius: tuning.dilateRadius)
    let feathered = gaussianBlur(dilated, width: width, height: height, sigma: tuning.featherSigma)
    return feathered.map { min(1, max(0, $0 / 255)) }
  }

  /// The median of `values`; numpy's convention for an even count (the
  /// average of the two middle elements) so this matches `np.median`
  /// exactly. Selected via quickselect rather than a full sort: the gain
  /// estimate feeds this a large fraction of the mask-resolution pixels
  /// per verified frame, where an O(n log n) sort (plus its full-size
  /// sorted copy) is measurable capture-time work for two order
  /// statistics.
  private static func median(of values: [Float]) -> Float {
    let count = values.count
    guard count > 0 else { return 0 }
    var buffer = values
    let upper = quickselect(&buffer, rank: count / 2)
    if count % 2 == 1 { return upper }
    // Quickselect leaves everything below index k partitioned <= the kth
    // order statistic, so the lower middle element is that prefix's max.
    let lower = buffer[0..<(count / 2)].max() ?? upper
    return (lower + upper) / 2
  }

  /// The `k`th-smallest element of `buffer` (0-based) via iterative
  /// Hoare-partition quickselect with a median-of-three pivot, expected
  /// O(n) with no allocations. Postcondition: `buffer[..<k]` holds only
  /// values `<=` the returned element.
  private static func quickselect(_ buffer: inout [Float], rank: Int) -> Float {
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

    func clampIndex(_ value: Int, _ limit: Int) -> Int { min(max(value, 0), limit - 1) }

    var horizontal = [Float](repeating: 0, count: values.count)
    for y in 0..<height {
      let rowBase = y * width
      for x in 0..<width {
        var acc: Float = 0
        for offset in -radius...radius {
          acc += values[rowBase + clampIndex(x + offset, width)] * kernel[offset + radius]
        }
        horizontal[rowBase + x] = acc
      }
    }
    var output = [Float](repeating: 0, count: values.count)
    for x in 0..<width {
      for y in 0..<height {
        var acc: Float = 0
        for offset in -radius...radius {
          acc += horizontal[clampIndex(y + offset, height) * width + x] * kernel[offset + radius]
        }
        output[y * width + x] = acc
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

    var horizontal = [Float](repeating: 0, count: values.count)
    for y in 0..<height {
      let rowBase = y * width
      for x in 0..<width {
        var best: Float = 0
        for neighborX in max(0, x - pixelRadius)...min(width - 1, x + pixelRadius) {
          best = max(best, values[rowBase + neighborX])
        }
        horizontal[rowBase + x] = best
      }
    }
    var output = [Float](repeating: 0, count: values.count)
    for x in 0..<width {
      for y in 0..<height {
        var best: Float = 0
        for neighborY in max(0, y - pixelRadius)...min(height - 1, y + pixelRadius) {
          best = max(best, horizontal[neighborY * width + x])
        }
        output[y * width + x] = best
      }
    }
    return output
  }
}
