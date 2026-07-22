import AVFoundation
import CoreImage
import UIKit

/// Persistent camera session for the live box-capture screen: a manual
/// shutter for photographing the puzzle box, plus a live low-res frame
/// stream that feeds the attached `LiveFrameConsumer` — the barcode
/// streamer on the box screen, the glare-guide tracker on the glare-free
/// screen. Device-only for the photo/streaming
/// path — on the Simulator `isCameraAvailable` is false and the UI falls
/// back to PhotosPicker (the DEBUG preview loop below is the Simulator's
/// stand-in for live frames; see `startDebugPreviewLoop`).
///
/// Deliberately a sibling of `PieceCameraSession` rather than a shared base:
/// the piece session's streaming half is hard-typed to
/// `PiecePreviewStreamer` and load-bearing for the piece-scan flows, so the
/// ~150-line session skeleton is duplicated here instead of refactored under
/// a working feature. Candidate for extraction if a third live-camera screen
/// ever appears.
final class BoxCameraSession: NSObject, AVCapturePhotoCaptureDelegate {
  static var isCameraAvailable: Bool {
    AVCaptureDevice.default(for: .video) != nil
  }

  let session = AVCaptureSession()
  /// Serializes startRunning/stopRunning so they cannot interleave.
  private let sessionQueue = DispatchQueue(label: "dk.delectosoft.pussel.box-camera")
  private let photoOutput = AVCapturePhotoOutput()
  private var isConfigured = false
  private var captureContinuation: CheckedContinuation<UIImage?, Never>?
  /// Whether the view still wants the camera. `start()` awaits the permission
  /// prompt, which can outlive the screen that asked, so the last caller's
  /// intent decides — not how far along the awaits happen to be.
  private var wantsRunning = false

  // MARK: - Live barcode streaming

  private let videoOutput = AVCaptureVideoDataOutput()
  /// Dedicated queue for both receiving frames and mutating the streaming
  /// state below, so the capture delegate never races the main-actor calls
  /// that enable/disable it — matches `sessionQueue`'s role for start/stop.
  private let videoQueue = DispatchQueue(label: "dk.delectosoft.pussel.box-camera.video")
  /// videoQueue-confined.
  private var isStreaming = false
  /// videoQueue-confined. Consumers themselves are thread-safe for the
  /// calls made here (see `LiveFrameConsumer`).
  private weak var frameConsumer: (any LiveFrameConsumer)?
  private static let ciContext = CIContext()

  /// Long side, in pixels, live frames are downscaled to before barcode
  /// analysis. Larger than the piece camera's 720: EAN-13's bar modules
  /// need more resolved width than a piece silhouette, and the extra pixels
  /// go straight into decode reliability at arm's length.
  private static let frameMaxLongSide: CGFloat = 1080

  /// Wires the consumer this session forwards frames to — the barcode
  /// streamer or the glare-guide tracker, depending on the presenting
  /// screen. Call once after creating the session (before `start()`), from
  /// the main actor.
  func attachFrameConsumer(_ consumer: any LiveFrameConsumer) {
    videoQueue.async { [weak self] in
      self?.frameConsumer = consumer
    }
  }

  /// Starts/stops forwarding camera frames to the frame consumer. The
  /// capture view pauses this during photo capture (see `capturePhoto()`)
  /// and while it isn't visible.
  func setStreamingEnabled(_ enabled: Bool) {
    videoQueue.async { [weak self] in
      self?.isStreaming = enabled
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
      // still reach this session and drive the barcode flow end-to-end.
      Self.debugActive = self
    #endif
    #if targetEnvironment(simulator)
      // The Simulator has no camera; see PieceCameraSession.start() for why
      // the real capture path (including the permission prompt) is skipped
      // entirely there. The DEBUG `previewloop` command feeds fake frames
      // straight into the streamer instead (see feedDebugFrame).
      return true
    #else
      guard await AVCaptureDevice.requestAccess(for: .video) else { return false }
      configureIfNeeded()
      guard isConfigured else { return false }
      // The screen was dismissed while the permission prompt was up: stop()
      // has already run, and starting now would strand the camera with no
      // one left to stop it. Not a failure — nothing to report.
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
    #if targetEnvironment(simulator)
      // No real camera on the Simulator — return the debug loop's image (if
      // one was fed) so the capture flow can complete end-to-end there.
      if let debugCaptureImage {
        return debugCaptureImage
      }
    #endif
    // Ignore shutter taps while a capture is in flight — overwriting the
    // stored continuation would leak it and hang the first caller.
    guard isConfigured, captureContinuation == nil else { return nil }
    // Pause the frame stream so it doesn't race the shutter for the photo
    // output's bandwidth; resumes once the shot is developed.
    setStreamingEnabled(false)
    defer { setStreamingEnabled(true) }
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
      // Rotate the delivered buffers to portrait at the connection level so
      // analyzed frames match the upright preview — same reasoning as
      // PieceCameraSession.configureIfNeeded.
      if let connection = videoOutput.connection(with: .video),
        connection.isVideoRotationAngleSupported(90)
      {
        connection.videoRotationAngle = 90
      }
    }
    session.commitConfiguration()
    isConfigured = true
  }

  /// Downscales a captured frame to ~`frameMaxLongSide` on its long side and
  /// renders it to a CGImage for barcode analysis (the render also detaches
  /// the frame from AVFoundation's recycled pixel-buffer pool, so
  /// `BarcodeDetector` can work on it asynchronously). Pure/CPU-only, so
  /// it's shared by the real camera path and the DEBUG preview loop.
  private static func downscaledFrame(ciImage: CIImage) -> CGImage? {
    let extent = ciImage.extent
    guard extent.width > 0, extent.height > 0 else { return nil }
    let scale = min(1, frameMaxLongSide / max(extent.width, extent.height))
    let scaled =
      scale < 1 ? ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) : ciImage
    return ciContext.createCGImage(scaled, from: scaled.extent)
  }

  #if DEBUG
    /// The currently active session, so `pusseldebug://previewloop` can
    /// reach the box-capture screen's camera on the Simulator — mirrors
    /// `PieceCameraSession.debugActive`. Registered by `start()`, cleared by
    /// `stop()`.
    static weak var debugActive: BoxCameraSession?

    private var debugPreviewTimer: Timer?

    /// The upright host image last fed by `startDebugPreviewLoop`. Stored so
    /// `capturePhoto()` can return it on the Simulator, giving the manual
    /// shutter path a real image without a physical camera.
    private(set) var debugCaptureImage: UIImage?

    /// Feeds `image` through the same downscale → detect path as a real
    /// camera frame, on a ~3Hz repeating timer — the Simulator has no
    /// camera, so this is how `pusseldebug://previewloop?path=<host image
    /// path>` drives the barcode flow there. `?stop=1` stops it.
    func startDebugPreviewLoop(image: UIImage) {
      stopDebugPreviewLoop()
      // Bake any EXIF orientation into the pixels first — see
      // PieceCameraSession.startDebugPreviewLoop.
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
      debugCaptureImage = upright
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
      debugCaptureImage = nil
    }

    private func feedDebugFrame(_ ciImage: CIImage) {
      videoQueue.async { [weak self] in
        guard let self, self.isStreaming, let frameConsumer = self.frameConsumer else {
          return
        }
        let now = Date()
        guard frameConsumer.shouldAcceptFrame(now: now) else { return }
        guard let frame = Self.downscaledFrame(ciImage: ciImage) else { return }
        frameConsumer.submit(cgImage: frame, now: now)
      }
    }
  #endif
}

extension BoxCameraSession: AVCaptureVideoDataOutputSampleBufferDelegate {
  /// Runs on `videoQueue`. Checks the throttle *before* downscaling so a
  /// frame that would just be dropped never pays for the render.
  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard isStreaming, let frameConsumer,
      let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
    else { return }
    let now = Date()
    guard frameConsumer.shouldAcceptFrame(now: now) else { return }
    guard let frame = Self.downscaledFrame(ciImage: CIImage(cvPixelBuffer: pixelBuffer)) else {
      return
    }
    frameConsumer.submit(cgImage: frame, now: now)
  }
}
