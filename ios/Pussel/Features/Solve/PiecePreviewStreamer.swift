import CoreGraphics
import Foundation
import Observation
import UIKit

/// The latest live-preview detection, published by `PiecePreviewStreamer`.
/// Mirrors the backend's quality flags: a plain `detected` region (yellow
/// outline) versus one the backend judged `lockable` (green outline, per
/// `include_quality=true`'s `lockable` flag).
enum PiecePreviewState: Equatable {
  case none
  case detected(polygon: [NormalizedPoint], confidence: Double)
  case lockable(polygon: [NormalizedPoint], confidence: Double)

  var polygon: [NormalizedPoint]? {
    switch self {
    case .none: return nil
    case .detected(let polygon, _), .lockable(let polygon, _): return polygon
    }
  }

  var isLockable: Bool {
    if case .lockable = self { return true }
    return false
  }
}

/// Owns the throttle/in-flight state machine for the live piece-outline
/// pipeline, and publishes the latest detection so `PieceCaptureView` can
/// draw it. One instance per piece capture screen, created alongside its
/// `PieceCameraSession`.
///
/// Frames are analyzed **on-device** by `PieceLiveDetector` (Vision
/// subject-lift + contours) — no network round-trip, so the outline tracks
/// the camera at inference speed rather than upload speed. When Vision's
/// subject lifting isn't available at all (e.g. the Simulator), the streamer
/// falls back — permanently, per instance — to the legacy
/// `POST /api/v1/piece/preview` server path with a JPEG-encoded frame.
///
/// Split isolation by design: `shouldAcceptFrame`/`submit` are safe to call
/// from any thread (lock-guarded, matching `StubURLProtocol`'s precedent
/// for cross-thread state in this codebase — see PusselTests/APIClientTests.swift)
/// — `PieceCameraSession` calls them from its dedicated video-capture
/// queue, before doing any downscale work, so a throttled/in-flight frame
/// is dropped cheaply. The UI-facing `state`/`frameSize`/`updatedAt` are
/// main-actor-isolated, since SwiftUI only ever reads them from
/// `PieceCaptureView`'s body.
@Observable
final class PiecePreviewStreamer: @unchecked Sendable {
  private let api: APIClient
  private let detector = PieceLiveDetector()
  private let lock = NSLock()
  private var throttle = PiecePreviewThrottle()
  /// Latched (lock-guarded) after the first `PieceLiveDetectorUnavailable`,
  /// so every later frame goes straight to the server path instead of paying
  /// a doomed Vision attempt per frame.
  private var visionUnavailable = false
  /// JPEG quality for frames sent down the server fallback path (matches the
  /// pre-on-device streaming pipeline).
  private static let fallbackJPEGQuality: CGFloat = 0.6

  @MainActor private(set) var state: PiecePreviewState = .none
  /// Pixel size of the frame `state`'s polygon was measured against — the
  /// upright-portrait downscaled JPEG `PieceCameraSession` sent. Its aspect
  /// ratio drives the overlay's aspect-fill mapping
  /// (`PiecePreviewGeometry.viewPolygon`).
  @MainActor private(set) var frameSize: CGSize = .zero
  /// When `state` was last updated. `PieceCaptureView` hides the overlay
  /// once this gets too old, so a stalled request stream (e.g. persistent
  /// network errors) doesn't leave a stale outline glued to the screen.
  @MainActor private(set) var updatedAt: Date = .distantPast

  init(api: APIClient) {
    self.api = api
  }

  /// Whether a frame arriving now is worth downscaling and sending. Call
  /// this before doing that work — safe from any thread.
  func shouldAcceptFrame(now: Date = Date()) -> Bool {
    lock.withLock { throttle.shouldSend(now: now) }
  }

  /// Analyzes an already-downscaled upright frame and publishes the result.
  /// Safe to call from any thread (typically the video-capture queue); the
  /// detection runs on a detached background task and the state publication
  /// hops to the main actor. A no-op if the throttle would reject the frame
  /// (e.g. it raced another accepted frame between `shouldAcceptFrame` and
  /// here).
  func submit(cgImage: CGImage, frameSize: CGSize, now: Date = Date()) {
    let accepted = lock.withLock { () -> Bool in
      guard throttle.shouldSend(now: now) else { return false }
      throttle.markSent(at: now)
      return true
    }
    guard accepted else { return }
    let useServerPath = lock.withLock { visionUnavailable }
    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self else { return }
      defer { self.lock.withLock { self.throttle.markCompleted() } }
      var fallBack = useServerPath
      if !fallBack {
        do {
          let detection = try self.detector.detect(cgImage: cgImage)
          await MainActor.run { self.apply(detection, frameSize: frameSize, now: Date()) }
        } catch {
          // Vision subject lifting can't run here at all (e.g. Simulator) —
          // latch so this and every later frame use the server path.
          self.lock.withLock { self.visionUnavailable = true }
          fallBack = true
        }
      }
      if fallBack {
        await self.submitToServer(cgImage: cgImage, frameSize: frameSize)
      }
    }
  }

  /// Legacy server-side preview: JPEG-encode the frame and POST it to
  /// `/api/v1/piece/preview`. Only reached when on-device Vision is
  /// unavailable on this platform.
  private func submitToServer(cgImage: CGImage, frameSize: CGSize) async {
    let jpegData = UIImage(cgImage: cgImage).jpegData(
      compressionQuality: Self.fallbackJPEGQuality)
    guard let jpegData else { return }
    do {
      let response = try await api.previewPiece(imageData: jpegData)
      await MainActor.run { self.apply(response, frameSize: frameSize, now: Date()) }
    } catch {
      // Leave the last-known state on a transient error — a single
      // dropped frame shouldn't blank the outline mid-track. The
      // staleness check in PieceCaptureView hides it if errors persist.
    }
  }

  /// Resets to no detection and clears the throttle — called when
  /// streaming pauses (photo capture, view disappearing) so a stale
  /// outline doesn't linger, and so streaming resumes cleanly afterward.
  @MainActor
  func reset() {
    lock.withLock { throttle = PiecePreviewThrottle() }
    state = .none
    frameSize = .zero
    updatedAt = .distantPast
  }

  @MainActor
  private func apply(_ detection: PieceLiveDetection?, frameSize: CGSize, now: Date) {
    self.frameSize = frameSize
    updatedAt = now
    guard let detection, detection.polygon.count >= 3 else {
      state = .none
      return
    }
    state =
      detection.lockable
      ? .lockable(polygon: detection.polygon, confidence: detection.confidence)
      : .detected(polygon: detection.polygon, confidence: detection.confidence)
  }

  @MainActor
  private func apply(_ response: PiecePreviewResponse, frameSize: CGSize, now: Date) {
    self.frameSize = frameSize
    updatedAt = now
    guard response.found, response.polygon.count >= 3 else {
      state = .none
      return
    }
    state =
      response.lockable == true
      ? .lockable(polygon: response.polygon, confidence: response.confidence)
      : .detected(polygon: response.polygon, confidence: response.confidence)
  }
}
