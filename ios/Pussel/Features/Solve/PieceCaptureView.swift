import AVFoundation
import PhotosUI
import QuartzCore
import SwiftUI
import UIKit

/// Full-screen piece capture, presented when the "+" tile in the piece list is
/// tapped. The camera only runs while this is on screen. Device-only — on the
/// Simulator `PieceCameraSession.isCameraAvailable` is false and the "+" tile
/// opens a PhotosPicker instead of this view.
struct PieceCaptureView: View {
  @Environment(AppModel.self) private var model
  @Environment(\.dismiss) private var dismiss
  @State private var camera = PieceCameraSession()
  @State private var previewStreamer: PiecePreviewStreamer?
  @State private var photoItem: PhotosPickerItem?
  @State private var isCapturing = false

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()
      CameraPreview(
        session: camera.session,
        previewState: previewStreamer?.state ?? .none,
        updatedAt: previewStreamer?.updatedAt ?? .distantPast,
        frameSize: previewStreamer?.frameSize ?? .zero
      )
      .ignoresSafeArea()
    }
    .overlay(alignment: .top) {
      Button("Cancel") { dismiss() }
        .padding()
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    // Two overlays rather than one HStack: the shutter centers on the screen
    // itself, so it stays put no matter how wide the library button is.
    .overlay(alignment: .bottom) {
      shutterButton.padding(.bottom, 24)
    }
    .overlay(alignment: .bottomLeading) {
      libraryButton
        .padding(.leading, 32)
        .padding(.bottom, 40)
    }
    .task {
      let streamer = previewStreamer ?? PiecePreviewStreamer(api: model.api)
      previewStreamer = streamer
      camera.attachPreviewStreamer(streamer)
      let started = await camera.start()
      // Dismissed while the permission prompt was up — the user left on
      // purpose, so there is nothing to complain about.
      guard !Task.isCancelled else { return }
      guard started else {
        #if DEBUG
          // The Simulator has no camera; stay on screen over the black
          // background so `pusseldebug://previewloop` can still demo the
          // overlay instead of bouncing straight back out. A real device
          // failure (permission denied, etc.) still reports and dismisses.
          if !PieceCameraSession.isCameraAvailable {
            camera.setPreviewStreamingEnabled(true)
            return
          }
        #endif
        model.reportPieceError("Pussel cannot use the camera. Check camera access in Settings.")
        dismiss()
        return
      }
      camera.setPreviewStreamingEnabled(true)
    }
    .onDisappear {
      camera.setPreviewStreamingEnabled(false)
      camera.stop()
      previewStreamer?.reset()
    }
    .keepsScreenAwake()
    .onChange(of: photoItem) { _, item in
      guard let item else { return }
      Task {
        await model.addPiece(from: item)
        photoItem = nil
        // Dismiss either way: a failure sets an error on the solve screen,
        // which stays hidden behind this cover.
        dismiss()
      }
    }
  }

  private var libraryButton: some View {
    PhotosPicker(selection: $photoItem, matching: .images) {
      Image(systemName: "photo.on.rectangle")
        .font(.title2)
        .foregroundStyle(.white)
    }
    .accessibilityLabel("Add piece from library")
  }

  private var shutterButton: some View {
    Button {
      // Ignore taps while a shot is developing, so a nil result below means
      // the capture itself failed rather than a double tap.
      guard !isCapturing else { return }
      isCapturing = true
      Task {
        let image = await camera.capturePhoto()
        isCapturing = false
        guard let image else {
          model.reportPieceError("Could not take that photo. Try again.")
          dismiss()
          return
        }
        model.addPiece(image: image)
        dismiss()
      }
    } label: {
      ZStack {
        Circle().fill(.white).frame(width: 58, height: 58)
        Circle().strokeBorder(.white, lineWidth: 3).frame(width: 70, height: 70)
      }
    }
    .accessibilityLabel("Capture piece")
  }
}

/// Camera preview with the M9 live piece-outline overlay: a `CAShapeLayer`
/// drawn on top of the `AVCaptureVideoPreviewLayer`, mapped from
/// `PiecePreviewStreamer`'s normalized polygon via
/// `PiecePreviewGeometry.viewPolygon` (aspect-fill onto the overlay bounds,
/// mirroring the preview layer's `.resizeAspectFill` of the same upright
/// portrait feed).
struct CameraPreview: UIViewRepresentable {
  let session: AVCaptureSession
  let previewState: PiecePreviewState
  let updatedAt: Date
  /// Pixel size of the frame the current polygon was measured against —
  /// the upright-portrait downscaled JPEG that was sent to the backend. Its
  /// aspect ratio drives the aspect-fill mapping onto the overlay bounds.
  let frameSize: CGSize

  /// The overlay hides once `updatedAt` is older than this — a stalled
  /// request stream (e.g. persistent network errors) shouldn't leave a
  /// stale outline glued to the screen.
  static let staleInterval: TimeInterval = 1.5
  /// Implicit-transaction duration for path/color changes, so the outline
  /// glides between detections rather than snapping.
  private static let pathAnimationDuration: TimeInterval = 0.12

  final class PreviewView: UIView {
    // Overrides UIView's class property, so `static` isn't an option (it can't
    // be combined with `override`); silence static_over_final_class.
    // swiftlint:disable:next static_over_final_class
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    // Safe by construction: layerClass above guarantees `layer` is an
    // AVCaptureVideoPreviewLayer, so this cast can never fail.
    // swiftlint:disable:next force_cast
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    let outlineLayer = CAShapeLayer()

    override init(frame: CGRect) {
      super.init(frame: frame)
      outlineLayer.fillColor = UIColor.clear.cgColor
      outlineLayer.lineWidth = 3
      outlineLayer.shadowColor = UIColor.black.cgColor
      outlineLayer.shadowOpacity = 0.6
      outlineLayer.shadowRadius = 3
      outlineLayer.shadowOffset = .zero
      outlineLayer.isHidden = true
      layer.addSublayer(outlineLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
      fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
      super.layoutSubviews()
      outlineLayer.frame = bounds
    }
  }

  func makeUIView(context: Context) -> PreviewView {
    let view = PreviewView()
    view.previewLayer.session = session
    view.previewLayer.videoGravity = .resizeAspectFill
    return view
  }

  func updateUIView(_ uiView: PreviewView, context: Context) {
    guard let polygon = previewState.polygon, polygon.count >= 3,
      Date().timeIntervalSince(updatedAt) < Self.staleInterval
    else {
      uiView.outlineLayer.isHidden = true
      return
    }

    // One mapping path for device and Simulator: the streamed frame is
    // rotated to upright portrait at the connection level (see
    // PieceCameraSession.configureIfNeeded), so it shows exactly what the
    // `.resizeAspectFill` preview layer shows — same image, same aspect.
    // Aspect-filling the frame-normalized polygon onto the overlay's bounds
    // therefore reproduces the preview layer's geometry with no
    // orientation guesswork (and on the Simulator there is no live preview
    // layer to convert through anyway; the backdrop is just black).
    let viewPoints = PiecePreviewGeometry.viewPolygon(
      fromFramePolygon: polygon, frameSize: frameSize, viewBounds: uiView.bounds.size)
    let path = UIBezierPath()
    path.move(to: viewPoints[0])
    for point in viewPoints.dropFirst() {
      path.addLine(to: point)
    }
    path.close()

    CATransaction.begin()
    CATransaction.setAnimationDuration(Self.pathAnimationDuration)
    uiView.outlineLayer.path = path.cgPath
    uiView.outlineLayer.strokeColor =
      (previewState.isLockable ? UIColor.systemGreen : UIColor.systemYellow).cgColor
    uiView.outlineLayer.isHidden = false
    CATransaction.commit()
  }
}
