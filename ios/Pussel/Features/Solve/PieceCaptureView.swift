import AVFoundation
import PhotosUI
import SwiftUI

/// Full-screen piece capture, presented when the "+" tile in the piece list is
/// tapped. The camera only runs while this is on screen. Device-only — on the
/// Simulator `PieceCameraSession.isCameraAvailable` is false and the "+" tile
/// opens a PhotosPicker instead of this view.
struct PieceCaptureView: View {
  @Environment(AppModel.self) private var model
  @Environment(\.dismiss) private var dismiss
  @State private var camera = PieceCameraSession()
  @State private var photoItem: PhotosPickerItem?

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()
      CameraPreview(session: camera.session)
        .ignoresSafeArea()
    }
    .overlay(alignment: .top) {
      Button("Cancel") { dismiss() }
        .padding()
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .overlay(alignment: .bottom) {
      HStack {
        libraryButton.frame(maxWidth: .infinity, alignment: .leading)
        shutterButton
        // Balances the library button so the shutter stays centered.
        Color.clear.frame(maxWidth: .infinity)
      }
      .padding(.horizontal, 32)
      .padding(.bottom, 24)
    }
    .task {
      await camera.start()
    }
    .onDisappear {
      camera.stop()
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
      Task {
        if let image = await camera.capturePhoto() {
          model.addPiece(image: image)
          dismiss()
        }
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

struct CameraPreview: UIViewRepresentable {
  let session: AVCaptureSession

  final class PreviewView: UIView {
    // Overrides UIView's class property, so `static` isn't an option (it can't
    // be combined with `override`); silence static_over_final_class.
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
