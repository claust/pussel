import AVFoundation
import SwiftUI

/// Plain live camera preview for a `BoxCameraSession` — just an
/// `AVCaptureVideoPreviewLayer`, no overlay. Used by the capture screen's
/// non-AR path (the Simulator/E2E fallback, where ARKit isn't available and
/// the guidance comes from image registration instead); the AR path renders
/// its own `ARSCNView`.
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
