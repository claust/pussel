import CoreGraphics
import Foundation

/// Forwards one camera session's frame stream to several consumers.
///
/// A session holds a single `LiveFrameConsumer`, which was enough while each
/// capture screen wanted one thing from its frames. The glare-free screen
/// wants two: the guide tracker that steers the burst, and the barcode
/// streamer that watches for a Ravensburger code while the reference shot is
/// still being aimed. Downscaling is done once by the session and the same
/// `CGImage` handed to every consumer — the expensive half of the work is
/// shared, only the analysis is duplicated.
///
/// Each consumer keeps its own throttle: `shouldAcceptFrame` is true when
/// *any* of them would take the frame, and `submit` then asks each one
/// again, so a busy consumer simply sits a frame out instead of stalling
/// the others.
final class FanOutFrameConsumer: LiveFrameConsumer, @unchecked Sendable {
  private let consumers: [any LiveFrameConsumer]

  /// - Parameter consumers: The consumers to forward to, retained for this
  ///   object's lifetime (camera sessions hold their consumer weakly, so
  ///   something must own them — the presenting view owns this).
  init(_ consumers: [any LiveFrameConsumer]) {
    self.consumers = consumers
  }

  func shouldAcceptFrame(now: Date) -> Bool {
    consumers.contains { $0.shouldAcceptFrame(now: now) }
  }

  func submit(cgImage: CGImage, now: Date) {
    for consumer in consumers where consumer.shouldAcceptFrame(now: now) {
      consumer.submit(cgImage: cgImage, now: now)
    }
  }
}
