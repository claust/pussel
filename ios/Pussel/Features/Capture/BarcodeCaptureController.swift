import Foundation
import Observation

// MARK: - Lookup protocol

/// The one API call the barcode capture flow makes, expressed as a protocol
/// so tests can script responses without a network — mirrors
/// `PieceGeometryScanning` for the piece scanner.
@MainActor
protocol BarcodeLookupClient {
  func lookupBarcode(ean: String) async throws -> BarcodeLookupResponse
}

extension APIClient: BarcodeLookupClient {}

// MARK: - Controller

/// Orchestrates the automatic barcode → box-image flow on the live box
/// camera: feeds every frame's detection into `BarcodeStabilityTracker`,
/// fires the backend lookup exactly once per stable read, and hands the
/// decoded box JPEG to `onFound` — the owning view dismisses and enters the
/// confirm-trim step from there. Fully automatic by design: no confirmation
/// chip, just the `.lookingUp` phase for a passive progress banner.
///
/// Misses (`found == false`) blacklist the EAN for this controller's
/// lifetime, so a non-Ravensburger box held in view doesn't re-fire a doomed
/// lookup every few frames; the camera silently stays in photo mode.
/// Network/transport errors do *not* blacklist — the next stable read
/// retries, since the code itself may be perfectly resolvable.
@MainActor
@Observable
final class BarcodeCaptureController {
  enum Phase: Equatable {
    case scanning
    case lookingUp
  }

  private(set) var phase: Phase = .scanning
  /// Called with the decoded box JPEG and the backend's OCR'd piece-count
  /// estimate (nil when the box couldn't be read) when a lookup succeeds.
  /// Set once by the owning view.
  var onFound: ((Data, Int?) -> Void)?

  private let client: any BarcodeLookupClient
  private var tracker: BarcodeStabilityTracker
  /// EANs the backend answered `found == false` for; per-instance, so
  /// reopening the capture screen retries once.
  private var missedEANs: Set<String> = []

  init(
    client: any BarcodeLookupClient,
    tracker: BarcodeStabilityTracker = BarcodeStabilityTracker()
  ) {
    self.client = client
    self.tracker = tracker
  }

  /// Called on the main actor for every analyzed frame (nil when the frame
  /// had no valid barcode). Guards on `.scanning` so mid-lookup frames are
  /// silently dropped.
  func ingest(_ detection: EAN13Detection?) {
    guard phase == .scanning else { return }
    guard let payload = tracker.ingest(detection?.payload) else { return }
    guard !missedEANs.contains(payload) else { return }
    phase = .lookingUp
    Task { await lookup(ean: payload) }
  }

  private func lookup(ean: String) async {
    // Deliberately no tracker.reset() here: the tracker stays latched on
    // this payload, so a barcode that never leaves the frame cannot re-fire
    // — resetting would restart a 3-frame streak and, after a transport
    // error, hot-loop retries at ~1/s against a dead network. The latch
    // clears itself when the barcode leaves the frame or the payload
    // changes (see BarcodeStabilityTracker.ingest), which is exactly the
    // "retry on the next stable read" semantics documented above.
    defer { phase = .scanning }
    do {
      let response = try await client.lookupBarcode(ean: ean)
      guard response.found,
        let dataURL = response.boxImage,
        let jpeg = ImageUtilities.decodeDataURL(dataURL)
      else {
        // A genuine miss — and a `found` response whose image doesn't
        // decode, which would loop forever if retried.
        missedEANs.insert(ean)
        return
      }
      onFound?(jpeg, response.pieceCountEstimate)
    } catch {
      // Transport/server error: leave the EAN retryable and stay silently
      // in photo mode, per the no-error-UI design for this flow.
    }
  }
}
