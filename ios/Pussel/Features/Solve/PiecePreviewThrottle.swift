import Foundation

/// Pure decision logic for whether a new live-preview frame should be sent
/// to the backend, given the current throttle state. No I/O, no timers, no
/// actor isolation — a plain value type, so it's directly unit-testable
/// without mocking the network or a clock.
///
/// `PiecePreviewStreamer` is the sole owner of a live instance and is
/// responsible for calling `markSent`/`markCompleted` around each request;
/// this type only computes the yes/no answer and the state transitions.
struct PiecePreviewThrottle: Equatable {
  /// Minimum time between requests — the ~4Hz cadence M9 targets for
  /// tracking responsiveness without hammering the backend.
  static let minInterval: TimeInterval = 0.25

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
