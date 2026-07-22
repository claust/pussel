import Foundation

/// Pure debounce logic for live barcode reads. No I/O, no timers, no actor
/// isolation — a plain value type, directly unit-testable, mirroring
/// `PieceScanStabilityTracker`'s role for the piece scanner.
///
/// A single-frame read isn't trusted: Vision can misread a digit on one
/// frame, so a lookup only fires after the same payload arrives
/// `requiredConsecutiveHits` frames in a row. The tracker latches after
/// firing so a barcode held in front of the camera fires exactly once —
/// a different payload (or a frame with no barcode) re-arms it.
struct BarcodeStabilityTracker: Equatable {
  let requiredConsecutiveHits: Int

  private var lastPayload: String?
  private var streak = 0
  private var hasFired = false

  init(requiredConsecutiveHits: Int = 3) {
    self.requiredConsecutiveHits = requiredConsecutiveHits
  }

  /// Feed every frame's detection result (nil when the frame had no valid
  /// barcode). Returns the payload exactly once — on the ingest where its
  /// streak first reaches `requiredConsecutiveHits` — and nil otherwise.
  @discardableResult
  mutating func ingest(_ payload: String?) -> String? {
    guard let payload else {
      // No barcode in frame: break the streak and re-arm, so the next
      // stable read (same code or another) can fire fresh.
      lastPayload = nil
      streak = 0
      hasFired = false
      return nil
    }
    if payload == lastPayload {
      streak += 1
    } else {
      lastPayload = payload
      streak = 1
      hasFired = false
    }
    guard !hasFired, streak >= requiredConsecutiveHits else { return nil }
    hasFired = true
    return payload
  }

  /// Resets all state so the next read starts a fresh streak.
  mutating func reset() {
    lastPayload = nil
    streak = 0
    hasFired = false
  }
}
