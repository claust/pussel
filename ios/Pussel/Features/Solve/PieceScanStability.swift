import CoreGraphics
import Foundation

/// Pure decision logic for when a live piece detection has been stable-and-
/// lockable long enough to trigger an auto-capture. No I/O, no timers, no
/// actor isolation — a plain value type, directly unit-testable without
/// mocking the network or a clock.
///
/// Stability is measured as axis-aligned bounding-box IoU between consecutive
/// lockable detections. Full polygon IoU would be more precise but requires a
/// clipping algorithm (O(n²)) and is fragile on non-convex shapes; bbox IoU
/// is O(1), unit-testable with pencil arithmetic, and sufficient here because
/// the piece camera is held close — when the piece isn't moving, its bbox
/// barely drifts, and the minIoU threshold already ignores tiny jitter.
///
/// `PieceCaptureView` / its controller is responsible for calling `reset()`
/// after acting on a fire so the tracker is ready for the next piece.
struct PieceScanStabilityTracker {
  // MARK: - Configuration

  /// Minimum wall-clock duration a consecutive lockable streak must span
  /// before the tracker fires. Default 1.0 s matches M10's "hold for ~1 s"
  /// brief.
  let minDuration: TimeInterval
  /// Minimum number of lockable samples within the streak. Guards against a
  /// long but very sparsely sampled streak (e.g. one frame per 0.9 s).
  let minSamples: Int
  /// Minimum IoU between consecutive lockable bbox pairs to extend a streak;
  /// below this the piece has drifted and the streak resets.
  let minIoU: Double

  // MARK: - Private streak state

  private var streakStart: Date?
  private var streakSampleCount: Int = 0
  private var lastPolygon: [NormalizedPoint]?
  /// Once the tracker fires it latches here and ignores further lockable
  /// detections until `reset()` is called. This prevents the controller
  /// from seeing multiple fires for the same held pose.
  private var hasFired: Bool = false

  // MARK: - Init

  init(minDuration: TimeInterval = 1.0, minSamples: Int = 3, minIoU: Double = 0.8) {
    self.minDuration = minDuration
    self.minSamples = minSamples
    self.minIoU = minIoU
  }

  // MARK: - Public API

  /// Feed every preview update. Returns `true` exactly once per stable
  /// streak — on the ingest where the stability criterion first fires. The
  /// tracker latches after firing; call `reset()` to arm it for the next
  /// piece.
  ///
  /// Only `.lockable` states extend a streak; `.detected` and `.none` reset
  /// it. A detected-but-not-lockable frame means the backend's quality gate
  /// didn't pass (unclean contour, corner disagreement), so counting it would
  /// give a false sense of stability.
  @discardableResult
  mutating func ingest(_ state: PiecePreviewState, at date: Date) -> Bool {
    guard !hasFired else { return false }

    guard case .lockable(let polygon, _) = state else {
      // Any non-lockable state breaks the streak.
      resetStreak()
      return false
    }

    let bbox = boundingBox(of: polygon)
    guard polygon.count >= 3, isValidBbox(bbox) else {
      // Degenerate polygon (< 3 points, or a line/point whose bbox has zero
      // area) — treat as noise. The real pipeline already guards count >= 3
      // in PiecePreviewStreamer.apply, but this value type documents the same
      // contract, so it enforces it independently rather than trusting the
      // caller (a 2-point diagonal has a positive-area bbox and would slip
      // past the area check alone).
      resetStreak()
      return false
    }

    if let last = lastPolygon {
      let lastBbox = boundingBox(of: last)
      guard isValidBbox(lastBbox), iou(bbox, lastBbox) >= minIoU else {
        // The piece jumped or rotated — restart counting from this frame.
        resetStreak(firstPolygon: polygon, at: date)
        return false
      }
    } else {
      // First lockable frame in this streak.
      streakStart = date
      streakSampleCount = 0
    }

    lastPolygon = polygon
    streakSampleCount += 1

    guard let start = streakStart else { return false }
    let elapsed = date.timeIntervalSince(start)
    guard elapsed >= minDuration, streakSampleCount >= minSamples else { return false }

    hasFired = true
    return true
  }

  /// Resets all streak state so the tracker is ready for a new piece.
  /// Call this after acting on a fire (e.g. after the verdict UX completes).
  mutating func reset() {
    resetStreak()
    hasFired = false
  }

  // MARK: - Private helpers

  private mutating func resetStreak(firstPolygon: [NormalizedPoint]? = nil, at date: Date? = nil) {
    if let firstPolygon, let date {
      // Seed the new streak with the frame that caused the reset so the
      // detection that broke the old streak isn't lost — it's the first
      // sample of the next one.
      lastPolygon = firstPolygon
      streakStart = date
      streakSampleCount = 1
    } else {
      lastPolygon = nil
      streakStart = nil
      streakSampleCount = 0
    }
  }

  /// Axis-aligned bounding box of a polygon, in [0,1] normalized coordinates.
  private func boundingBox(of polygon: [NormalizedPoint]) -> CGRect {
    guard !polygon.isEmpty else { return .zero }
    var minX = polygon[0].x
    var maxX = polygon[0].x
    var minY = polygon[0].y
    var maxY = polygon[0].y
    for pt in polygon.dropFirst() {
      minX = min(minX, pt.x)
      maxX = max(maxX, pt.x)
      minY = min(minY, pt.y)
      maxY = max(maxY, pt.y)
    }
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }

  /// A bbox is non-degenerate when it has positive area. The polygon
  /// point-count check (≥ 3) lives in `ingest`, where the polygon is still in
  /// hand; together they reject lines and points, which would produce an IoU
  /// of 0 against everything.
  private func isValidBbox(_ rect: CGRect) -> Bool {
    rect.width > 0 && rect.height > 0
  }

  /// Intersection-over-union of two axis-aligned rectangles. Returns 0 when
  /// the rectangles don't overlap.
  private func iou(_ rectA: CGRect, _ rectB: CGRect) -> Double {
    let intersection = rectA.intersection(rectB)
    guard !intersection.isNull else { return 0 }
    let intersectionArea = Double(intersection.width * intersection.height)
    let unionArea =
      Double(rectA.width * rectA.height) + Double(rectB.width * rectB.height) - intersectionArea
    guard unionArea > 0 else { return 0 }
    return intersectionArea / unionArea
  }
}
