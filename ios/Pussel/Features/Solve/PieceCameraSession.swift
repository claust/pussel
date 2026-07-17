import AVFoundation
import CoreImage
import UIKit

/// Persistent camera session for capturing pieces one after another with a
/// manual shutter, plus a live low-res frame stream that feeds
/// `PiecePreviewStreamer` for the M9 outline overlay. Device-only for the
/// photo/streaming path — on the Simulator `isCameraAvailable` is false and
/// the UI falls back to PhotosPicker (the DEBUG preview loop below is the
/// Simulator's stand-in for live frames; see `startDebugPreviewLoop`).
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

  // MARK: - Live preview streaming (M9)

  private let videoOutput = AVCaptureVideoDataOutput()
  /// Dedicated queue for both receiving frames and mutating the streaming
  /// state below, so the capture delegate never races the main-actor calls
  /// that enable/disable it — matches `sessionQueue`'s role for
  /// start/stop.
  private let videoQueue = DispatchQueue(label: "dk.delectosoft.pussel.piece-camera.video")
  /// videoQueue-confined.
  private var isStreamingPreview = false
  /// videoQueue-confined. `PiecePreviewStreamer` itself is thread-safe for
  /// the calls made here (see its doc comment).
  private weak var previewStreamer: PiecePreviewStreamer?
  private static let ciContext = CIContext()

  /// Long side, in pixels, live-preview frames are downscaled to before
  /// JPEG-encoding and sending — small enough to keep the ~4Hz stream
  /// cheap, large enough for the backend's region detector to work with.
  private static let previewFrameMaxLongSide: CGFloat = 480
  private static let previewFrameJPEGQuality: CGFloat = 0.6

  /// Wires the streamer this session forwards frames to. Call once after
  /// creating the session (before `start()`), from the main actor.
  func attachPreviewStreamer(_ streamer: PiecePreviewStreamer) {
    videoQueue.async { [weak self] in
      self?.previewStreamer = streamer
    }
  }

  /// Starts/stops forwarding camera frames to the preview streamer. The
  /// capture view pauses this during photo capture (see `capturePhoto()`)
  /// and while it isn't visible.
  func setPreviewStreamingEnabled(_ enabled: Bool) {
    videoQueue.async { [weak self] in
      self?.isStreamingPreview = enabled
    }
  }

  // MARK: - Lifecycle

  /// Returns false when the camera cannot run — access denied, or no usable
  /// device — so the caller can report it rather than show a dead preview.
  func start() async -> Bool {
    wantsRunning = true
    #if DEBUG
      // Registered even when the real camera can't configure (e.g. the
      // Simulator has no device below) so `pusseldebug://previewloop` can
      // still reach this session and demo the overlay over a black preview.
      Self.debugActive = self
    #endif
    #if targetEnvironment(simulator)
      // The Simulator has no camera. Touching AVCaptureDevice's authorization
      // (requestAccess) pops an iOS 26 permission dialog that can't be
      // dismissed in a headless CI/agent session, and there's no device to
      // configure anyway — so skip the real capture path entirely. The view
      // presents its normal chrome over a black preview, and the DEBUG
      // `previewloop` command feeds fake frames straight into the streamer
      // (see feedDebugFrame). Real-device builds are unaffected: the #else
      // branch below is byte-identical to the original.
      return true
    #else
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
    #endif
  }

  func stop() {
    wantsRunning = false
    #if DEBUG
      stopDebugPreviewLoop()
      if Self.debugActive === self { Self.debugActive = nil }
    #endif
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
    // Pause the preview stream so it doesn't race the shutter for the
    // photo output's bandwidth; resumes once the shot is developed.
    setPreviewStreamingEnabled(false)
    defer { setPreviewStreamingEnabled(true) }
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
    if session.canAddOutput(videoOutput) {
      videoOutput.alwaysDiscardsLateVideoFrames = true
      videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
      session.addOutput(videoOutput)
      // Rotate the delivered buffers to portrait at the connection level, so
      // the downscaled JPEG sent to the backend is visually upright —
      // exactly what the user sees in the portrait preview layer. 90° is
      // the portrait rotation for a back camera (its sensor is mounted
      // landscape; AVCaptureDevice.RotationCoordinator reports 90 for
      // portrait on the back camera, and 90 is the angle equivalent of the
      // legacy `.portrait` videoOrientation). With the frame and the
      // preview showing the same upright image, the overlay needs no
      // orientation guess — just the shared aspect-fill bounds mapping
      // (PiecePreviewGeometry.viewPolygon).
      if let connection = videoOutput.connection(with: .video),
        connection.isVideoRotationAngleSupported(90)
      {
        connection.videoRotationAngle = 90
      }
    }
    session.commitConfiguration()
    isConfigured = true
  }

  /// Downscales a captured frame to ~`previewFrameMaxLongSide` on its long
  /// side and JPEG-encodes it for the preview endpoint. Pure/CPU-only, so
  /// it's shared by both the real camera path (below) and the DEBUG
  /// preview loop.
  private static func downscaledJPEG(ciImage: CIImage) -> (data: Data, size: CGSize)? {
    let extent = ciImage.extent
    guard extent.width > 0, extent.height > 0 else { return nil }
    let scale = min(1, previewFrameMaxLongSide / max(extent.width, extent.height))
    let scaled =
      scale < 1 ? ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) : ciImage
    guard let cgImage = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
    let size = CGSize(width: cgImage.width, height: cgImage.height)
    guard let data = UIImage(cgImage: cgImage).jpegData(compressionQuality: previewFrameJPEGQuality)
    else { return nil }
    return (data, size)
  }

  #if DEBUG
    /// The currently active session, so `pusseldebug://previewloop` can
    /// reach the piece-capture screen's camera without `AppModel` knowing
    /// about view-local camera state (mirrors how the debug command file
    /// reaches `AppModel` itself; see `DebugDriver`). Registered by
    /// `start()`, cleared by `stop()`.
    static weak var debugActive: PieceCameraSession?

    private var debugPreviewTimer: Timer?

    /// Feeds `image` through the same downscale → stream → overlay path as
    /// a real camera frame, on a ~3Hz repeating timer — the Simulator has
    /// no camera, so this is how `pusseldebug://previewloop?path=<host
    /// image path>` demos M9 there. `pusseldebug://previewloop?stop=1`
    /// stops it (`stopDebugPreviewLoop`).
    func startDebugPreviewLoop(image: UIImage) {
      stopDebugPreviewLoop()
      // Bake any EXIF orientation into the pixels first: `cgImage` alone
      // drops UIImage's orientation metadata, which would feed a rotated
      // frame for a phone-shot fixture. Real camera frames don't need this
      // — their buffers arrive already upright (videoRotationAngle = 90).
      let upright: UIImage
      if image.imageOrientation == .up, image.cgImage != nil {
        upright = image
      } else {
        upright = UIGraphicsImageRenderer(size: image.size).image { _ in
          image.draw(in: CGRect(origin: .zero, size: image.size))
        }
      }
      guard let cgImage = upright.cgImage else { return }
      let ciImage = CIImage(cgImage: cgImage)
      feedDebugFrame(ciImage)
      let interval: TimeInterval = 1.0 / 3.0
      let timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
        self?.feedDebugFrame(ciImage)
      }
      RunLoop.main.add(timer, forMode: .common)
      debugPreviewTimer = timer
    }

    func stopDebugPreviewLoop() {
      debugPreviewTimer?.invalidate()
      debugPreviewTimer = nil
    }

    private func feedDebugFrame(_ ciImage: CIImage) {
      videoQueue.async { [weak self] in
        guard let self, self.isStreamingPreview, let previewStreamer = self.previewStreamer else {
          return
        }
        let now = Date()
        guard previewStreamer.shouldAcceptFrame(now: now) else { return }
        guard let frame = Self.downscaledJPEG(ciImage: ciImage) else { return }
        previewStreamer.submit(jpegData: frame.data, frameSize: frame.size, now: now)
      }
    }
  #endif
}

extension PieceCameraSession: AVCaptureVideoDataOutputSampleBufferDelegate {
  /// Runs on `videoQueue`. Checks the throttle *before* downscaling so a
  /// frame that would just be dropped never pays for the JPEG encode.
  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard isStreamingPreview, let previewStreamer,
      let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
    else { return }
    let now = Date()
    guard previewStreamer.shouldAcceptFrame(now: now) else { return }
    guard let frame = Self.downscaledJPEG(ciImage: CIImage(cvPixelBuffer: pixelBuffer)) else {
      return
    }
    previewStreamer.submit(jpegData: frame.data, frameSize: frame.size, now: now)
  }
}
