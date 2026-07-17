import CoreGraphics
import CoreImage
import Foundation
import Vision
import simd

/// A piece candidate detected on-device in a live camera frame. The on-device
/// analogue of the backend's `PiecePreviewResponse` (see `piece_detector.py`):
/// same normalized coordinate contract (top-left origin, [0,1] of the frame),
/// same area/aspect confidence semantics.
struct PieceLiveDetection: Equatable {
  /// Dense outline of the detected region, normalized to the analyzed frame.
  let polygon: [NormalizedPoint]
  /// Axis-aligned bounding box of `polygon`, normalized to the frame.
  let bbox: CGRect
  /// Product of the area and aspect band scores, in (0, 1].
  let confidence: Double
  /// On-device stand-in for the backend's quality gate: a single clean
  /// component, well inside the frame, in the full-confidence area/aspect
  /// bands. The final scan capture is still verified server-side, so this
  /// only gates when the auto-capture is allowed to fire.
  let lockable: Bool
}

/// Vision availability failure — subject lifting needs OS support that the
/// Simulator (and very old hardware) may not provide. Thrown only for error
/// codes that mean the request can never run here (see
/// `PieceLiveDetector.isUnavailable`); distinct from "no piece in this
/// frame" (a `nil` detection), so the caller can latch its fallback to the
/// server-side preview endpoint immediately instead of retrying per-frame.
struct PieceLiveDetectorUnavailable: Error {}

/// A Vision failure that doesn't prove the platform can't run subject
/// lifting — the frame is lost, but the next one may well succeed. The
/// caller should treat one of these like a dropped frame and only demote to
/// the server path if they keep happening.
struct PieceLiveDetectorTransientFailure: Error {}

/// On-device replacement for the `/api/v1/piece/preview` round-trip: Apple's
/// subject-lift segmentation (`VNGenerateForegroundInstanceMaskRequest`, the
/// on-device counterpart of the backend's rembg) followed by contour
/// extraction on the resulting mask (`VNDetectContoursRequest`) and the same
/// area/aspect plausibility gates as `piece_detector.detect_region`.
///
/// Unlike the server path (320px working image, polygon simplified and capped
/// at 60 points) this analyzes the frame at streaming resolution and returns a
/// smoothed, arc-length-resampled 120-point outline, so the overlay follows
/// tab curvature instead of cutting corners.
///
/// Stateless and thread-safe: each `detect` call builds its own Vision
/// requests. Called from `PiecePreviewStreamer`'s background task, never the
/// main thread.
final class PieceLiveDetector: Sendable {
  // MARK: - Gates ported from backend/app/services/piece_detector.py

  /// The detected region must cover at least this fraction of the frame.
  static let minPieceAreaRatio = 0.005
  /// ... and at most this fraction (a face/torso covers far more).
  static let maxPieceAreaRatio = 0.35
  /// Area band (fraction of frame) that gets full confidence.
  static let fullConfidenceAreaRange = 0.01...0.15
  /// Bounding-box aspect ratio (long/short side) limits.
  static let maxAspectRatio = 3.5
  static let fullConfidenceMaxAspect = 2.0

  // MARK: - Outline post-processing

  /// Points in the returned outline. The backend preview capped at 60 by
  /// dropping vertices; here the contour is arc-length-resampled, so every
  /// point is equally spaced along the outline and 120 comfortably follows
  /// tab curvature at overlay scale.
  static let polygonPointCount = 120
  /// Circular moving-average window (in contour samples) applied before
  /// resampling, to soften mask-pixel staircase without eroding tabs.
  static let smoothingWindow = 5
  /// A contour point within this normalized distance of the frame edge marks
  /// the detection as border-touching (segmentation likely clipped the
  /// piece) — mirrors the backend's 2px margin on its ≤320px working image.
  static let borderTouchMargin = 0.008
  /// A secondary contour with at least this fraction of the largest
  /// contour's area counts as a distinct component; more than one such
  /// component means an unclean segmentation (mirrors
  /// `piece_geometry.contour.LARGE_COMPONENT_AREA_FRAC`).
  static let largeComponentAreaFrac = 0.02

  /// Detects the most salient piece-like region in an upright frame.
  ///
  /// - Parameter cgImage: The downscaled, upright camera frame.
  /// - Returns: The detection, or nil when nothing piece-like is in frame.
  /// - Throws: `PieceLiveDetectorUnavailable` when Vision's subject lifting
  ///   cannot run at all on this platform (the caller should fall back to
  ///   the server preview endpoint), or `PieceLiveDetectorTransientFailure`
  ///   for any other Vision error (the caller should treat the frame as
  ///   dropped).
  func detect(cgImage: CGImage) throws -> PieceLiveDetection? {
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    let maskRequest = VNGenerateForegroundInstanceMaskRequest()
    do {
      try handler.perform([maskRequest])
    } catch {
      throw
        Self.isUnavailable(error)
        ? PieceLiveDetectorUnavailable() : PieceLiveDetectorTransientFailure()
    }
    guard let observation = maskRequest.results?.first, !observation.allInstances.isEmpty else {
      return nil
    }
    let maskBuffer: CVPixelBuffer
    do {
      maskBuffer = try observation.generateScaledMaskForImage(
        forInstances: observation.allInstances, from: handler)
    } catch {
      // Classified like a `perform` failure: a genuine Vision error must not
      // masquerade as "empty frame" (which would clear the overlay and dodge
      // the caller's transient-failure demotion).
      throw
        Self.isUnavailable(error)
        ? PieceLiveDetectorUnavailable() : PieceLiveDetectorTransientFailure()
    }

    guard let contours = Self.topLevelContours(ofMask: maskBuffer) else { return nil }
    let areas = contours.map { abs(Self.signedArea(of: $0.normalizedPoints)) }
    guard let largestIndex = areas.indices.max(by: { areas[$0] < areas[$1] }),
      areas[largestIndex] > 0
    else { return nil }

    // Flip Vision's lower-left-origin normalized points to the top-left
    // origin the overlay (and the backend contract) use.
    let contour = contours[largestIndex].normalizedPoints.map {
      SIMD2<Double>(Double($0.x), 1.0 - Double($0.y))
    }
    guard contour.count >= 3 else { return nil }

    // Area/aspect plausibility gates, identical bands to the backend.
    let areaRatio = areas[largestIndex]
    let bbox = Self.boundingBox(of: contour)
    let pixelWidth = bbox.width * Double(cgImage.width)
    let pixelHeight = bbox.height * Double(cgImage.height)
    let aspect = max(pixelWidth, pixelHeight) / max(1, min(pixelWidth, pixelHeight))
    let areaScore = Self.bandScore(
      areaRatio, hardLow: Self.minPieceAreaRatio,
      fullLow: Self.fullConfidenceAreaRange.lowerBound,
      fullHigh: Self.fullConfidenceAreaRange.upperBound, hardHigh: Self.maxPieceAreaRatio)
    let aspectScore = Self.bandScore(
      aspect, hardLow: 0.0, fullLow: 1.0,
      fullHigh: Self.fullConfidenceMaxAspect, hardHigh: Self.maxAspectRatio)
    guard areaScore > 0, aspectScore > 0 else { return nil }

    let outline = Self.resampled(
      Self.smoothed(contour, window: Self.smoothingWindow), count: Self.polygonPointCount)

    // Border touching is judged on the RAW contour: smoothing pulls points
    // inward, so a clipped piece could otherwise sneak past the margin and
    // read as lockable (mirrors the backend, which checks the unsmoothed
    // contour too).
    let borderTouching = contour.contains { point in
      point.x <= Self.borderTouchMargin || point.y <= Self.borderTouchMargin
        || point.x >= 1.0 - Self.borderTouchMargin || point.y >= 1.0 - Self.borderTouchMargin
    }
    let largeComponents = areas.filter { $0 >= Self.largeComponentAreaFrac * areaRatio }.count
    let lockable =
      !borderTouching && largeComponents == 1 && areaScore >= 1.0 && aspectScore >= 1.0

    return PieceLiveDetection(
      polygon: outline.map { NormalizedPoint(x: $0.x, y: $0.y) },
      bbox: CGRect(x: bbox.minX, y: bbox.minY, width: bbox.width, height: bbox.height),
      confidence: max(areaScore * aspectScore, 0.001),
      lockable: lockable
    )
  }

  /// Whether a `perform` error proves the request can never run on this
  /// platform (unsupported request/revision), as opposed to a one-off
  /// failure worth retrying on the next frame.
  static func isUnavailable(_ error: Error) -> Bool {
    let nsError = error as NSError
    guard nsError.domain == VNErrorDomain,
      let code = VNErrorCode(rawValue: nsError.code)
    else { return false }
    switch code {
    case .unsupportedRequest, .unsupportedRevision, .notImplemented:
      return true
    default:
      return false
    }
  }

  // MARK: - Contour extraction

  /// Runs `VNDetectContoursRequest` over a subject-lift mask and returns the
  /// top-level contours. The mask is a float pixel buffer (background 0,
  /// subject 1) at the analyzed frame's resolution, so the contour tracer is
  /// configured for a bright subject on a dark background with no contrast
  /// boost.
  private static func topLevelContours(ofMask maskBuffer: CVPixelBuffer) -> [VNContour]? {
    let request = VNDetectContoursRequest()
    request.contrastAdjustment = 1.0
    request.detectsDarkOnLight = false
    request.maximumImageDimension = 1024
    let handler = VNImageRequestHandler(
      ciImage: CIImage(cvPixelBuffer: maskBuffer), options: [:])
    guard (try? handler.perform([request])) != nil,
      let observation = request.results?.first
    else { return nil }
    let contours = observation.topLevelContours
    return contours.isEmpty ? nil : contours
  }

  // MARK: - Geometry helpers

  /// Shoelace signed area of a closed polygon in normalized coordinates —
  /// its magnitude is the covered fraction of the frame.
  private static func signedArea(of points: [simd_float2]) -> Double {
    guard points.count >= 3 else { return 0 }
    var sum = 0.0
    for index in points.indices {
      let current = points[index]
      let next = points[(index + 1) % points.count]
      sum += Double(current.x) * Double(next.y) - Double(next.x) * Double(current.y)
    }
    return sum / 2
  }

  private static func boundingBox(of points: [SIMD2<Double>]) -> CGRect {
    var minX = points[0].x
    var maxX = points[0].x
    var minY = points[0].y
    var maxY = points[0].y
    for point in points.dropFirst() {
      minX = Swift.min(minX, point.x)
      maxX = Swift.max(maxX, point.x)
      minY = Swift.min(minY, point.y)
      maxY = Swift.max(maxY, point.y)
    }
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }

  /// Circular moving average over a closed contour — the cheap counterpart
  /// of the backend's Gaussian contour smoothing, enough to soften the
  /// mask-pixel staircase at overlay scale.
  static func smoothed(_ points: [SIMD2<Double>], window: Int) -> [SIMD2<Double>] {
    let count = points.count
    guard count > window, window > 1 else { return points }
    let half = window / 2
    var result = [SIMD2<Double>]()
    result.reserveCapacity(count)
    for index in 0..<count {
      var sum = SIMD2<Double>(0, 0)
      for offset in -half...half {
        sum += points[((index + offset) % count + count) % count]
      }
      result.append(sum / Double(2 * half + 1))
    }
    return result
  }

  /// Arc-length-equidistant resampling of a closed contour to `count`
  /// points — the Swift port of `piece_geometry.contour.resample_contour`,
  /// so every outline the overlay draws has evenly spaced vertices
  /// regardless of how densely Vision traced each stretch.
  static func resampled(_ points: [SIMD2<Double>], count: Int) -> [SIMD2<Double>] {
    guard points.count >= 2, count >= 3 else { return points }
    var cumulative = [0.0]
    cumulative.reserveCapacity(points.count + 1)
    for index in points.indices {
      let current = points[index]
      let next = points[(index + 1) % points.count]
      cumulative.append(cumulative[index] + simd_distance(current, next))
    }
    let total = cumulative[points.count]
    guard total > 0 else { return points }

    var result = [SIMD2<Double>]()
    result.reserveCapacity(count)
    var segment = 0
    for sample in 0..<count {
      let target = total * Double(sample) / Double(count)
      while segment < points.count - 1, cumulative[segment + 1] < target {
        segment += 1
      }
      let segmentLength = cumulative[segment + 1] - cumulative[segment]
      let fraction = segmentLength > 0 ? (target - cumulative[segment]) / segmentLength : 0
      let start = points[segment]
      let end = points[(segment + 1) % points.count]
      result.append(start + (end - start) * fraction)
    }
    return result
  }

  /// Confidence band scoring, ported verbatim from the backend's
  /// `_band_score`: 1.0 inside [fullLow, fullHigh], tapering linearly to 0.0
  /// at the hard limits.
  static func bandScore(
    _ value: Double, hardLow: Double, fullLow: Double, fullHigh: Double, hardHigh: Double
  ) -> Double {
    if value < fullLow {
      if value <= hardLow { return 0.0 }
      return (value - hardLow) / (fullLow - hardLow)
    }
    if value > fullHigh {
      if value >= hardHigh { return 0.0 }
      return (hardHigh - value) / (hardHigh - fullHigh)
    }
    return 1.0
  }
}
