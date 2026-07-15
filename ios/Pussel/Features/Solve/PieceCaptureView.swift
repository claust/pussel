import AVFoundation
import PhotosUI
import SwiftUI

/// Manual piece capture: a persistent live camera view with a shutter button
/// on device, and a PhotosPicker path that also serves as the only option on
/// the Simulator.
struct PieceCaptureView: View {
  @Environment(AppModel.self) private var model
  @State private var camera = PieceCameraSession()
  @State private var photoItem: PhotosPickerItem?

  var body: some View {
    VStack(spacing: 12) {
      if PieceCameraSession.isCameraAvailable {
        CameraPreview(session: camera.session)
          .frame(height: 280)
          .clipShape(RoundedRectangle(cornerRadius: 12))
          .overlay(alignment: .bottom) {
            shutterButton.padding(.bottom, 14)
          }
          .task {
            await camera.start()
          }
          .onDisappear {
            camera.stop()
          }
      }
      PhotosPicker(selection: $photoItem, matching: .images) {
        Label(
          PieceCameraSession.isCameraAvailable ? "Add Piece from Library" : "Pick Piece Photo",
          systemImage: "photo.on.rectangle"
        )
        .frame(maxWidth: .infinity)
      }
      .buttonStyle(.bordered)
    }
    .onChange(of: photoItem) { _, item in
      guard let item else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self),
          let image = UIImage(data: data)
        {
          model.addPiece(image: image)
        }
        photoItem = nil
      }
    }
  }

  private var shutterButton: some View {
    Button {
      Task {
        if let image = await camera.capturePhoto() {
          model.addPiece(image: image)
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
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
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
