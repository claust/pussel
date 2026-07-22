import CoreGraphics
import Foundation

/// A consumer of a camera session's live low-res frame stream — the barcode
/// streamer on the box screen, the glare-guide tracker on the glare-free
/// screen. Both methods must be safe to call from the session's video queue.
///
/// The two-call shape exists so the session can check `shouldAcceptFrame`
/// *before* paying to downscale and detach a frame that would only be
/// dropped by the consumer's throttle.
protocol LiveFrameConsumer: AnyObject {
  /// Whether a frame arriving at `now` is worth preparing and submitting.
  func shouldAcceptFrame(now: Date) -> Bool
  /// Analyzes an already-downscaled upright frame. Must return quickly
  /// (dispatch real work to a background task).
  func submit(cgImage: CGImage, now: Date)
}

/// `BarcodeScanStreamer` already speaks this shape — the protocol was
/// extracted from it when the glare-free screen became the second consumer.
extension BarcodeScanStreamer: LiveFrameConsumer {}
