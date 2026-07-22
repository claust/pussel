import CoreGraphics
import Foundation

/// Owns the throttle/in-flight state machine for the live barcode pipeline:
/// `BoxCameraSession` pushes downscaled frames in from its video-capture
/// queue, `BarcodeDetector` reads them on a background task, and each
/// result — including "no barcode in this frame" — is delivered to
/// `onDetection` on the main actor.
///
/// The `PiecePreviewStreamer` counterpart for the box camera, minus its
/// server-fallback branch: barcode decoding is classic CV and runs wherever
/// Vision does, so a failed frame is simply dropped. Split isolation matches
/// too: `shouldAcceptFrame`/`submit` are lock-guarded and safe from any
/// thread; `onDetection` is main-actor-confined.
final class BarcodeScanStreamer: @unchecked Sendable {
  private let detector = BarcodeDetector()
  private let lock = NSLock()
  private var throttle = PiecePreviewThrottle()

  /// Called on the main actor with every analyzed frame's result (nil when
  /// the frame had no valid barcode). Set once by the owning view before
  /// streaming starts.
  @MainActor var onDetection: ((EAN13Detection?) -> Void)?

  /// Whether a frame arriving now is worth downscaling and analyzing. Call
  /// this before doing that work — safe from any thread.
  func shouldAcceptFrame(now: Date = Date()) -> Bool {
    lock.withLock { throttle.shouldSend(now: now) }
  }

  /// Analyzes an already-downscaled upright frame and delivers the result.
  /// Safe to call from any thread (typically the video-capture queue); the
  /// detection runs on a detached background task and the delivery hops to
  /// the main actor. A no-op if the throttle would reject the frame.
  func submit(cgImage: CGImage, now: Date = Date()) {
    let accepted = lock.withLock { () -> Bool in
      guard throttle.shouldSend(now: now) else { return false }
      throttle.markSent(at: now)
      return true
    }
    guard accepted else { return }
    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self else { return }
      defer { self.lock.withLock { self.throttle.markCompleted() } }
      // A throwing Vision request is a dropped frame — the next one retries.
      let detection = (try? self.detector.detect(cgImage: cgImage)) ?? nil
      await MainActor.run { self.onDetection?(detection) }
    }
  }

  /// Clears the throttle — called when streaming pauses (photo capture,
  /// view disappearing) so streaming resumes cleanly afterward.
  func reset() {
    lock.withLock { throttle = PiecePreviewThrottle() }
  }
}
