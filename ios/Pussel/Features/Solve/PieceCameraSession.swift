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
    private let photoOutput = AVCapturePhotoOutput()
    private var isConfigured = false
    private var captureContinuation: CheckedContinuation<UIImage?, Never>?

    func start() async {
        guard await AVCaptureDevice.requestAccess(for: .video) else { return }
        configureIfNeeded()
        guard isConfigured, !session.isRunning else { return }
        let session = self.session
        Task.detached {
            // startRunning blocks, so keep it off the main thread.
            session.startRunning()
        }
    }

    func stop() {
        guard session.isRunning else { return }
        let session = self.session
        Task.detached {
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

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let image = photo.fileDataRepresentation().flatMap(UIImage.init(data:))
        captureContinuation?.resume(returning: image)
        captureContinuation = nil
    }

    private func configureIfNeeded() {
        guard !isConfigured,
              let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }
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
