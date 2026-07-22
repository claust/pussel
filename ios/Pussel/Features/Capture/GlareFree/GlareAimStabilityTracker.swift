import Foundation

/// Pure decision logic for the glare-free auto-shutter: fires exactly once
/// when the guide dot has stayed on target long enough, then latches until
/// `reset()` (called when the flow moves to the next step). A single
/// on-target frame is not a steady hand — requiring a short dwell keeps
/// the shutter from firing while the dot merely sweeps through the ring.
///
/// The `PieceScanStabilityTracker` idea reduced to one boolean signal: no
/// I/O, no timers, directly unit-testable with injected dates.
struct GlareAimStabilityTracker: Equatable {
  /// How long the dot must stay continuously on target.
  static let minDuration: TimeInterval = 0.4
  /// Minimum on-target measurements in the streak, so a stalled frame
  /// stream (two updates 0.4 s apart) cannot fake a steady aim.
  static let minSamples = 2

  private var streakStart: Date?
  private var samples = 0
  private var hasFired = false

  init() {}

  /// Feeds one measurement; returns true exactly once, when the on-target
  /// streak first satisfies both thresholds.
  mutating func ingest(onTarget: Bool, at date: Date) -> Bool {
    guard !hasFired else { return false }
    guard onTarget else {
      streakStart = nil
      samples = 0
      return false
    }
    let start = streakStart ?? date
    streakStart = start
    samples += 1
    guard samples >= Self.minSamples, date.timeIntervalSince(start) >= Self.minDuration else {
      return false
    }
    hasFired = true
    return true
  }

  /// Re-arms the tracker for the next step's aim.
  mutating func reset() {
    self = GlareAimStabilityTracker()
  }
}
