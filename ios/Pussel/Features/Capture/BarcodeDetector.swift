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
/// request revision 1: every later revision decodes via an ML model whose
/// inference context the Simulator cannot create ("Error code: 9" per
/// frame, observed live, and re-confirmed on the iOS 26 Simulator for
/// revisions 2, 3 and 4 as well as the modern `DetectBarcodesRequest`,
/// whose only revision is 4). Revision 1 is classic CV, runs everywhere,
/// and is entirely adequate for a close-range 1D EAN-13 on a 1080px frame:
/// across 240 synthetic renders (four payloads × six rotations × five
/// sizes × with/without speckle) decoded on macOS, where every revision
/// runs, revisions 1 and 3 returned the same payload in every case that
/// either read one — revision 3 never disagreed, it only missed three
/// small tilted codes revision 1 read. So the pin costs no accuracy and
/// the checksum filter in `detect` sees the same payloads either way.
/// Unlike `PieceLiveDetector`'s subject lifting, no availability/fallback
/// split is needed.
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

  /// `VNDetectBarcodesRequestRevision1`, written as the literal value the
  /// SDK header gives it: the constant carries a blanket "deprecated in iOS
  /// 17.0: renamed to `VNDetectBarcodesRequestRevision3`" annotation, so
  /// naming it warns on every build even though the rename does not hold
  /// here — see the type's doc comment for why revision 1 stays.
  private static let classicRevision = 1

  /// Reads an EAN-13 from an upright frame, or nil when none is legible.
  ///
  /// Only payloads that pass the local checksum (`EAN13.isValidChecksum`)
  /// and the `minBoundingBoxLongSide` size gate are returned; a failed
  /// Vision request throws and the caller drops the frame.
  func detect(cgImage: CGImage) throws -> EAN13Detection? {
    let request = VNDetectBarcodesRequest()
    request.revision = Self.classicRevision
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
