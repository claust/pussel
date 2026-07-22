import CoreGraphics
import Foundation
import Vision

/// A checksum-valid EAN-13 read from a live camera frame.
struct EAN13Detection: Equatable {
  /// The 13-digit barcode payload.
  let payload: String
  /// Vision's normalized bounding box of the barcode within the frame
  /// (bottom-left origin, [0,1]); only its long side is used, as a proxy
  /// for "close enough to trust the read".
  let boundingBox: CGRect
}

/// On-device EAN-13 reader for the live box-capture preview. A thin wrapper
/// around `VNDetectBarcodesRequest` restricted to `.ean13`, pinned to
/// request revision 1: later revisions decode via an ML model whose
/// inference context the Simulator cannot create ("Error code: 9" per
/// frame, observed live), while revision 1 is classic CV and runs
/// everywhere — and is entirely adequate for a close-range 1D EAN-13 on a
/// 1080px frame. Unlike `PieceLiveDetector`'s subject lifting, no
/// availability/fallback split is needed.
///
/// Stateless and thread-safe: each `detect` call builds its own request.
/// Called from `BarcodeScanStreamer`'s background task, never the main
/// thread.
final class BarcodeDetector: Sendable {
  /// Detections whose bounding box's long side spans less than this
  /// fraction of the frame are ignored: EAN-13's fine bar modules need
  /// several pixels each to resolve, and far-away reads are where misreads
  /// live. The long side is the barcode's bar-to-bar axis whichever way the
  /// code is rotated — for a vertical barcode the box is narrow and tall,
  /// so gating on width alone would reject every rotated read.
  static let minBoundingBoxLongSide: CGFloat = 0.15

  /// Reads an EAN-13 from an upright frame, or nil when none is legible.
  ///
  /// Only payloads that pass the local checksum (`EAN13.isValidChecksum`)
  /// and the `minBoundingBoxLongSide` size gate are returned; a failed
  /// Vision request throws and the caller drops the frame.
  func detect(cgImage: CGImage) throws -> EAN13Detection? {
    let request = VNDetectBarcodesRequest()
    request.revision = VNDetectBarcodesRequestRevision1
    request.symbologies = [.ean13]
    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
    try handler.perform([request])
    let observations = request.results ?? []
    for observation in observations {
      guard let payload = observation.payloadStringValue,
        EAN13.isValidChecksum(payload),
        max(observation.boundingBox.width, observation.boundingBox.height)
          >= Self.minBoundingBoxLongSide
      else { continue }
      return EAN13Detection(payload: payload, boundingBox: observation.boundingBox)
    }
    return nil
  }
}
