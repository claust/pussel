import AVFoundation
import UIKit

/// Persistent camera session for capturing pieces one after another with a
/// manual shutter. Device-only — on the Simulator `isCameraAvailable` is false
/// and the UI falls back to PhotosPicker.
final class PieceCameraSession: NSObject, AVCapturePhotoCaptureDelegate {
  static var isCameraAvailable: Bool {
    AVCaptureDevice.default(for: .video) != nil
  }

  let session = AVCaptureSession()
  /// Serializes startRunning/stopRunning so they cannot interleave.
  private let sessionQueue = DispatchQueue(label: "dk.delectosoft.pussel.piece-camera")
  private let photoOutput = AVCapturePhotoOutput()
  private var isConfigured = false
  private var captureContinuation: CheckedContinuation<UIImage?, Never>?
  /// Whether the view still wants the camera. `start()` awaits the permission
  /// prompt, which can outlive the screen that asked, so the last caller's
  /// intent decides — not how far along the awaits happen to be.
  private var wantsRunning = false

  /// Returns false when the camera cannot run — access denied, or no usable
  /// device — so the caller can report it rather than show a dead preview.
  func start() async -> Bool {
    wantsRunning = true
    guard await AVCaptureDevice.requestAccess(for: .video) else { return false }
    configureIfNeeded()
    guard isConfigured else { return false }
    // The screen was dismissed while the permission prompt was up: stop() has
    // already run, and starting now would strand the camera with no one left
    // to stop it. Not a failure — nothing to report, nothing to run.
    guard wantsRunning else { return true }
    let session = self.session
    // startRunning/stopRunning both block, so they run off the main thread —
    // through one serial queue so a stop queued behind a start runs after it.
    sessionQueue.async {
      guard !session.isRunning else { return }
      session.startRunning()
    }
    return true
  }

  func stop() {
    wantsRunning = false
    let session = self.session
    sessionQueue.async {
      guard session.isRunning else { return }
      session.stopRunning()
    }
  }

  func capturePhoto() async -> UIImage? {
    // Ignore shutter taps while a capture is in flight — overwriting the
    // stored continuation would leak it and hang the first caller.
    guard isConfigured, captureContinuation == nil else { return nil }
    return await withCheckedContinuation { continuation in
      captureContinuation = continuation
      photoOutput.capturePhoto(with: AVCapturePhotoSettings(), delegate: self)
    }
  }

  func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    let image = photo.fileDataRepresentation().flatMap(UIImage.init(data:))
    captureContinuation?.resume(returning: image)
    captureContinuation = nil
  }

  private func configureIfNeeded() {
    guard !isConfigured,
      let device = AVCaptureDevice.default(for: .video),
      let input = try? AVCaptureDeviceInput(device: device),
      session.canAddInput(input)
    else { return }
    session.beginConfiguration()
    session.sessionPreset = .photo
    session.addInput(input)
    guard session.canAddOutput(photoOutput) else {
      session.commitConfiguration()
      return
    }
    session.addOutput(photoOutput)
    session.commitConfiguration()
    isConfigured = true
  }
}
