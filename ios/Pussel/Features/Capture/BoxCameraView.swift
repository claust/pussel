import AVFoundation
import SwiftUI
import UIKit

/// Full-screen live box capture, presented from CapturePuzzleView's "Take
/// Puzzle Photo" button. Dual-mode with no mode switch: the frame stream is
/// continuously scanned for an EAN-13 barcode — a stable read fires the
/// Ravensburger lookup automatically and, on a hit, hands the box image to
/// `onBarcodeJPEG` — while the shutter photographs the box as before via
/// `onImage`. A lookup miss is silent: the camera just stays in photo mode.
struct BoxCameraView: View {
  /// Manual shutter result — routed to the existing detect-frame path.
  let onImage: (UIImage) -> Void
  /// Resolved barcode-lookup box JPEG — routed straight to confirm-trim.
  let onBarcodeJPEG: (Data) -> Void

  @Environment(AppModel.self) private var model
  @Environment(\.dismiss) private var dismiss
  @State private var camera = BoxCameraSession()
  @State private var streamer = BarcodeScanStreamer()
  @State private var controller: BarcodeCaptureController?
  @State private var isCapturing = false

  private var isLookingUp: Bool {
    controller?.phase == .lookingUp
  }

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()
      BoxCameraPreview(session: camera.session)
        .ignoresSafeArea()
    }
    .overlay(alignment: .top) {
      Button("Cancel") { dismiss() }
        .padding()
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .overlay(alignment: .bottom) {
      VStack(spacing: 16) {
        if isLookingUp {
          lookupBanner
        }
        shutterButton
      }
      .padding(.bottom, 24)
    }
    .task {
      let controller = self.controller ?? BarcodeCaptureController(client: model.api)
      self.controller = controller
      controller.onFound = { jpeg in
        dismiss()
        onBarcodeJPEG(jpeg)
      }
      streamer.onDetection = { [weak controller] detection in
        controller?.ingest(detection)
      }
      camera.attachFrameConsumer(streamer)
      let started = await camera.start()
      // Dismissed while the permission prompt was up — the user left on
      // purpose, so there is nothing to complain about.
      guard !Task.isCancelled else { return }
      guard started else {
        #if DEBUG
          // The Simulator has no camera; stay on screen over the black
          // background so `pusseldebug://previewloop` can still drive the
          // barcode flow. A real device failure still reports and dismisses.
          if !BoxCameraSession.isCameraAvailable {
            camera.setStreamingEnabled(true)
            return
          }
        #endif
        model.flow.errorMessage =
          "Pussel cannot use the camera. Check camera access in Settings."
        dismiss()
        return
      }
      camera.setStreamingEnabled(true)
    }
    .onDisappear {
      camera.setStreamingEnabled(false)
      camera.stop()
      streamer.reset()
    }
    .keepsScreenAwake()
  }

  /// Passive progress banner over the live preview while a stable barcode
  /// read is being resolved — the automatic flow's only UI. Styled after
  /// PieceScanView's verdict capsule.
  private var lookupBanner: some View {
    HStack(spacing: 10) {
      ProgressView()
        .tint(.white)
      Text("Looking up puzzle…")
        .font(.callout.weight(.semibold))
        .foregroundStyle(.white)
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 10)
    .background(Capsule().fill(.ultraThinMaterial.opacity(0.9)))
    .shadow(radius: 4)
    .transition(.scale(scale: 0.9).combined(with: .opacity))
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
          model.flow.errorMessage = "Could not take that photo. Try again."
          dismiss()
          return
        }
        dismiss()
        onImage(image)
      }
    } label: {
      ZStack {
        Circle().fill(.white).frame(width: 58, height: 58)
        Circle().strokeBorder(.white, lineWidth: 3).frame(width: 70, height: 70)
      }
    }
    .opacity(isLookingUp ? 0.4 : 1)
    .disabled(isLookingUp)
    .accessibilityLabel("Photograph the puzzle box")
  }
}

/// Plain live camera preview for the box capture screen — just an
/// `AVCaptureVideoPreviewLayer`, no overlay (the barcode flow's only UI is
/// the lookup banner above).
struct BoxCameraPreview: UIViewRepresentable {
  let session: AVCaptureSession

  final class PreviewView: UIView {
    // Overrides UIView's class property, so `static` isn't an option (it
    // can't be combined with `override`); silence static_over_final_class.
    // swiftlint:disable:next static_over_final_class
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    // Safe by construction: layerClass above guarantees `layer` is an
    // AVCaptureVideoPreviewLayer, so this cast can never fail.
    // swiftlint:disable:next force_cast
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
  }

  func makeUIView(context: Context) -> PreviewView {
    let view = PreviewView()
    view.previewLayer.session = session
    view.previewLayer.videoGravity = .resizeAspectFill
    return view
  }

  func updateUIView(_ uiView: PreviewView, context: Context) {}
}
