import CoreGraphics
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import UIKit
import Vision
import os
import simd

/// One measurement of where the reference shot's content currently sits in
/// the live camera frame.
struct GlareGuideUpdate: Equatable {
  /// Content displacement in unit coordinates (top-left origin): a point at
  /// unit position `u` in the reference frame currently appears at
  /// `u + offset` in the live frame. Nil when the frame could not be
  /// registered against the reference — tracking lost.
  let offset: CGSize?
  /// Width / height of the analyzed live frame, so the view can map unit
  /// positions through the preview's aspect-fill.
  let frameAspect: CGFloat
}

/// Live tracker behind the glare-free screen's puzzle-anchored guide dot:
/// `BoxCameraSession` pushes downscaled preview frames in from its video
/// queue, each is registered against the center reference shot with
/// Vision's translational aligner, and the measured content offset is
/// delivered to `onUpdate` on the main actor. The controller turns that
/// into a dot pinned to a fixed spot on the puzzle — and into the
/// auto-shutter once the dot reaches the screen center.
///
/// Translation-only on purpose: the guide needs ~10 Hz robustness, not
/// sub-pixel warps. The composer redoes full homographic registration on
/// the captured photos; a few percent of guide error just moves the dot a
/// few points on screen. Registration runs on small highlight-capped,
/// blurred proxies (see `GlareFreeComposer` for why capping matters), and
/// each measurement must pass a direct image-difference check before it is
/// trusted — a failed check reports "tracking lost" rather than a dot
/// jumping to wherever a specular blob dragged the correlation.
///
/// Threading matches `BarcodeScanStreamer`: `shouldAcceptFrame`/`submit`
/// are lock-guarded and safe from any thread, `setReference`/`onUpdate`
/// are main-actor-confined.
final class GlareGuideTracker: LiveFrameConsumer, @unchecked Sendable {
  /// Long-side cap for the tracking proxies. A quarter of the composer's
  /// registration resolution: the dot only needs ~1% accuracy, and small
  /// proxies keep each measurement a few milliseconds.
  static let proxyMaxDimension: CGFloat = 384
  /// Blur radius as a fraction of the proxy long side — the composer's
  /// coarse-stage blur, chosen there for a wide, glare-resistant basin.
  private static let blurFraction: CGFloat = 1.0 / 64.0
  /// Acceptance cap for the mean absolute difference (linear RGB, central
  /// half) between the reference proxy and the translated frame proxy.
  /// Looser than the composer's 0.04: these proxies are far blurrier, but
  /// a hand-held frame also differs from the reference by the perspective
  /// a translation cannot model.
  private static let maxAlignmentError: CGFloat = 0.06
  /// Offsets beyond this are geometrically implausible for a guided shot
  /// (the puzzle would be mostly out of frame) — treated as tracking lost.
  private static let maxPlausibleOffset: CGFloat = 0.6

  private static let context = CIContext()
  private static let log = Logger(subsystem: "dk.delectosoft.pussel", category: "glare-guide")

  private let lock = NSLock()
  private var throttle = PiecePreviewThrottle()
  /// Lock-guarded; whether the previous measurement had a fix, so only
  /// acquired/lost *transitions* are logged — not every 10 Hz frame.
  private var wasTracking = false
  /// Lock-guarded; nil until the center reference shot is taken.
  private var referenceProxy: CGImage?
  /// Lock-guarded: the last verified proxy-space translation, reused as a
  /// registration prior for the next frame. A hand moves continuously, so
  /// the previous fix is almost always within a few pixels of the truth.
  private var lastTranslation: CGVector?
  /// Lock-guarded: where the current step expects the content to end up
  /// (unit offset, top-left origin) — the step's anchor steered to the
  /// screen center. Used as a fallback prior, which matters because
  /// Vision's translational correlation can misfire on large raw
  /// displacements; registering the small residual after pre-shifting by
  /// the expectation is far better conditioned.
  private var expectedOffset: CGSize?

  /// Called on the main actor with every measurement. Set once by the
  /// owning view before streaming starts.
  @MainActor var onUpdate: ((GlareGuideUpdate) -> Void)?

  /// Installs the center shot as the registration reference. Synchronous
  /// but cheap (one ~384 px Core Image render); call it when the first
  /// step's photo lands.
  @MainActor func setReference(_ image: UIImage) {
    let proxy = Self.referenceProxy(of: image)
    lock.withLock {
      referenceProxy = proxy
      lastTranslation = nil
      wasTracking = false
      throttle = PiecePreviewThrottle()
    }
  }

  /// The reference tracking proxy of a captured photo, with its EXIF
  /// orientation baked into the pixels first. A device photo's `cgImage`
  /// is the landscape sensor bitmap plus an orientation tag, while the
  /// live frames this reference is registered against arrive physically
  /// rotated upright — using the raw bitmap would compare a 90°-rotated
  /// pair, and tracking would never lock. Internal so the regression test
  /// can exercise exactly the path `setReference` uses.
  static func referenceProxy(of image: UIImage) -> CGImage? {
    let pixelSize = CGSize(
      width: image.size.width * image.scale, height: image.size.height * image.scale)
    guard pixelSize.width > 0, pixelSize.height > 0 else { return nil }
    let scale = min(1, proxyMaxDimension / max(pixelSize.width, pixelSize.height))
    let target = CGSize(
      width: (pixelSize.width * scale).rounded(.down),
      height: (pixelSize.height * scale).rounded(.down))
    guard target.width >= 1, target.height >= 1 else { return nil }
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let upright = UIGraphicsImageRenderer(size: target, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: target))
    }
    return upright.cgImage.flatMap { trackingProxy(of: $0) }
  }

  /// Stops tracking (frames are ignored until a new reference is set) —
  /// called when the flow restarts.
  @MainActor func clearReference() {
    lock.withLock {
      referenceProxy = nil
      lastTranslation = nil
      expectedOffset = nil
      wasTracking = false
      throttle = PiecePreviewThrottle()
    }
  }

  /// Tells the tracker where the current step is steering the content —
  /// the controller's expected shift for the active anchor. Nil when
  /// there is no expectation (the reference step).
  @MainActor func setExpectedOffset(_ offset: CGSize?) {
    lock.withLock { expectedOffset = offset }
  }

  // MARK: - LiveFrameConsumer

  func shouldAcceptFrame(now: Date) -> Bool {
    lock.withLock { referenceProxy != nil && throttle.shouldSend(now: now) }
  }

  func submit(cgImage: CGImage, now: Date) {
    let state = lock.withLock { () -> (CGImage, [CGVector])? in
      guard let referenceProxy, throttle.shouldSend(now: now) else { return nil }
      throttle.markSent(at: now)
      var priors: [CGVector] = []
      if let lastTranslation {
        priors.append(lastTranslation)
      }
      if let expectedOffset {
        priors.append(
          Self.translationVector(
            ofUnitOffset: expectedOffset,
            proxySize: CGSize(width: referenceProxy.width, height: referenceProxy.height)))
      }
      priors.append(.zero)
      return (referenceProxy, priors)
    }
    guard let (reference, priors) = state else { return }
    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self else { return }
      defer { self.lock.withLock { self.throttle.markCompleted() } }
      let (update, translation) = Self.measure(
        frame: cgImage, referenceProxy: reference, priors: priors)
      let isTracking = update.offset != nil
      let transitioned = self.lock.withLock { () -> Bool in
        if let translation {
          self.lastTranslation = translation
        }
        let changed = self.wasTracking != isTracking
        self.wasTracking = isTracking
        return changed
      }
      if transitioned {
        if let offset = update.offset {
          let x = offset.width
          let y = offset.height
          Self.log.info(
            "tracking acquired, offset (\(x, format: .fixed(precision: 3)), \(y, format: .fixed(precision: 3)))"
          )
        } else {
          Self.log.info("tracking lost")
        }
      }
      await MainActor.run { self.onUpdate?(update) }
    }
  }

  // MARK: - Measurement

  /// Registers `frame` against the reference proxy and converts the found
  /// translation into a unit content offset. Returns the update plus the
  /// verified proxy-space translation (nil when lost) so the caller can
  /// carry it forward as the next frame's prior. Static so it is directly
  /// testable without a camera or throttle.
  ///
  /// Each prior is tried in turn, yielding up to two candidate fixes: the
  /// prior refined by Vision registering the residual of the pre-shifted
  /// proxy, and the raw prior itself. A `.zero` prior is plain direct
  /// registration. Every candidate must pass the image-difference
  /// verification, so an accepted raw prior is not a guess — it is
  /// template matching at a hypothesized displacement. The shape exists
  /// because Vision's correlation demonstrably misfires: steering-sized
  /// shifts came back with a wildly wrong x on the Simulator, and there
  /// even the residual of a nearly aligned pair can be garbage — the
  /// verification gate keeps whichever candidate actually matches the
  /// pixels.
  static func measure(
    frame: CGImage, referenceProxy: CGImage, priors: [CGVector] = [.zero]
  ) -> (GlareGuideUpdate, CGVector?) {
    let aspect = CGFloat(frame.width) / CGFloat(frame.height)
    let lost = (GlareGuideUpdate(offset: nil, frameAspect: aspect), CGVector?.none)
    let proxySize = CGSize(width: referenceProxy.width, height: referenceProxy.height)
    let extent = CGRect(origin: .zero, size: proxySize)
    // Render the frame proxy at exactly the reference proxy's pixel size so
    // the registration translation lives in one well-defined space. Live
    // frames and the reference photo share the camera's aspect, so this is
    // a pure downscale in practice, not a distorting stretch.
    guard let frameProxy = trackingProxy(of: frame, size: proxySize) else { return lost }
    for prior in priors {
      var candidates: [CGVector] = []
      if prior == .zero {
        if let direct = translation(mapping: frameProxy, ontoReference: referenceProxy) {
          candidates.append(direct)
        }
      } else {
        if let preShifted = rendered(frameProxy, shiftedBy: prior, extent: extent),
          let residual = translation(mapping: preShifted, ontoReference: referenceProxy)
        {
          candidates.append(CGVector(dx: prior.dx + residual.dx, dy: prior.dy + residual.dy))
        }
        candidates.append(prior)
      }
      for total in candidates {
        guard
          alignmentError(of: total, floating: frameProxy, reference: referenceProxy) ?? 1
            < maxAlignmentError
        else { continue }
        // `total` maps live-frame pixels into reference pixels (lower-left
        // origin): p_ref = p_live + t. A reference point therefore shows
        // up in the live frame displaced by −t — flipped to top-left units
        // for the UI.
        let offset = CGSize(
          width: -total.dx / proxySize.width,
          height: total.dy / proxySize.height)
        guard abs(offset.width) < maxPlausibleOffset, abs(offset.height) < maxPlausibleOffset
        else { continue }
        return (GlareGuideUpdate(offset: offset, frameAspect: aspect), total)
      }
    }
    return lost
  }

  /// The proxy-space translation at which a unit content offset (top-left
  /// origin) would place the live frame — the inverse of the conversion in
  /// `measure`, used to turn the expected offset into a prior.
  static func translationVector(ofUnitOffset offset: CGSize, proxySize: CGSize) -> CGVector {
    CGVector(dx: -offset.width * proxySize.width, dy: offset.height * proxySize.height)
  }

  /// Renders the proxy shifted by `translation` over neutral gray — the
  /// pre-shifted input whose residual registration a prior hypothesis
  /// measures. Gray, not black: the fill is registered against, and a hard
  /// dark border would hand the correlation false structure.
  private static func rendered(
    _ image: CGImage, shiftedBy translation: CGVector, extent: CGRect
  ) -> CGImage? {
    let gray = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: extent)
    let shifted = CIImage(cgImage: image)
      .transformed(by: CGAffineTransform(translationX: translation.dx, y: translation.dy))
      .composited(over: gray)
      .cropped(to: extent)
    return context.createCGImage(shifted, from: extent)
  }

  private static func translation(
    mapping floating: CGImage, ontoReference reference: CGImage
  ) -> CGVector? {
    let request = VNTranslationalImageRegistrationRequest(targetedCGImage: floating, options: [:])
    let handler = VNImageRequestHandler(cgImage: reference, options: [:])
    guard (try? handler.perform([request])) != nil,
      let transform = request.results?.first?.alignmentTransform
    else { return nil }
    return CGVector(dx: transform.tx, dy: transform.ty)
  }

  /// The tracking proxy of `image`: downscaled (to `size`, or to
  /// `proxyMaxDimension` on the long side), brightness capped at flat gray
  /// so glare carries no gradients, then blurred — the same conditioning
  /// the composer applies before its own registration. Internal so tests
  /// can build a reference proxy without the main-actor plumbing.
  static func trackingProxy(of image: CGImage, size: CGSize? = nil) -> CGImage? {
    let width = CGFloat(image.width)
    let height = CGFloat(image.height)
    guard width > 0, height > 0 else { return nil }
    let target: CGSize
    if let size {
      target = size
    } else {
      let scale = min(1, proxyMaxDimension / max(width, height))
      target = CGSize(
        width: (width * scale).rounded(.down), height: (height * scale).rounded(.down))
    }
    guard target.width >= 1, target.height >= 1 else { return nil }
    let extent = CGRect(origin: .zero, size: target)
    let downscaled = CIImage(cgImage: image)
      .transformed(by: CGAffineTransform(scaleX: target.width / width, y: target.height / height))
    let cap = CIImage(color: CIColor(red: 0.55, green: 0.55, blue: 0.55)).cropped(to: extent)
    let minimum = CIFilter.minimumCompositing()
    minimum.inputImage = downscaled.cropped(to: extent)
    minimum.backgroundImage = cap
    guard let capped = minimum.outputImage else { return nil }
    let blur = CIFilter.gaussianBlur()
    blur.inputImage = capped.clampedToExtent()
    blur.radius = Float(max(target.width, target.height) * blurFraction)
    guard let blurred = blur.outputImage else { return nil }
    return context.createCGImage(blurred.cropped(to: extent), from: extent)
  }

  /// Mean absolute difference (linear RGB) between the reference proxy and
  /// the frame proxy shifted by `translation` — the same verification idea
  /// as `GlareFreeComposer.alignmentError`, including the linear-space
  /// readout (an sRGB re-encode would inflate small averages several-fold).
  ///
  /// Averaged over the central half *clipped to where the shifted frame
  /// actually has pixels*: at steering-sized offsets the shifted frame no
  /// longer covers the whole window, and letting the filler band into the
  /// average would report "lost" precisely when the user reaches the
  /// target.
  private static func alignmentError(
    of translation: CGVector, floating: CGImage, reference: CGImage
  ) -> CGFloat? {
    let extent = CGRect(x: 0, y: 0, width: reference.width, height: reference.height)
    let gray = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: extent)
    let shifted = CIImage(cgImage: floating)
      .transformed(by: CGAffineTransform(translationX: translation.dx, y: translation.dy))
      .composited(over: gray)
      .cropped(to: extent)
    let window =
      extent
      .insetBy(dx: extent.width * 0.25, dy: extent.height * 0.25)
      .intersection(extent.offsetBy(dx: translation.dx, dy: translation.dy))
    guard !window.isEmpty else { return nil }
    let difference = CIFilter.differenceBlendMode()
    difference.inputImage = shifted
    difference.backgroundImage = CIImage(cgImage: reference)
    guard let diffImage = difference.outputImage else { return nil }
    let average = CIFilter.areaAverage()
    average.inputImage = diffImage
    average.extent = window
    guard let output = average.outputImage,
      let linear = CGColorSpace(name: CGColorSpace.linearSRGB)
    else { return nil }
    var pixel = [UInt8](repeating: 0, count: 4)
    context.render(
      output, toBitmap: &pixel, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
      format: .RGBA8, colorSpace: linear)
    return (CGFloat(pixel[0]) + CGFloat(pixel[1]) + CGFloat(pixel[2])) / 3 / 255
  }
}
