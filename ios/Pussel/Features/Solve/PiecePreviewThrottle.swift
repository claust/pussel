import Foundation

/// Pure decision logic for whether a new live-preview frame should be
/// analyzed, given the current throttle state. No I/O, no timers, no
/// actor isolation — a plain value type, so it's directly unit-testable
/// without mocking the detector or a clock.
///
/// `PiecePreviewStreamer` is the sole owner of a live instance and is
/// responsible for calling `markSent`/`markCompleted` around each analysis;
/// this type only computes the yes/no answer and the state transitions.
struct PiecePreviewThrottle: Equatable {
  /// Minimum time between analyses. With on-device detection the real pacer
  /// is the in-flight gate (one Vision inference at a time, typically
  /// 50–150 ms); this floor just keeps the overlay's worst case at ~10Hz so
  /// a very fast device doesn't spend the whole CPU on segmentation. The
  /// server fallback path is additionally paced by its own round-trip,
  /// which the in-flight gate serializes exactly as before.
  static let minInterval: TimeInterval = 0.1

  private(set) var lastSend: Date?
  private(set) var isInFlight = false

  init() {}

  /// Whether a frame arriving at `now` should be sent: never while a
  /// request is already in flight, and never sooner than `minInterval`
  /// after the last send.
  func shouldSend(now: Date) -> Bool {
    guard !isInFlight else { return false }
    guard let lastSend else { return true }
    return now.timeIntervalSince(lastSend) >= Self.minInterval
  }

  /// Marks a send as started; `shouldSend` returns false until a matching
  /// `markCompleted()`.
  mutating func markSent(at now: Date) {
    lastSend = now
    isInFlight = true
  }

  /// Marks the in-flight request as finished, successfully or not, so the
  /// next frame past `minInterval` can send again.
  mutating func markCompleted() {
    isInFlight = false
  }
}
