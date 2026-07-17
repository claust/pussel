import Foundation
import UIKit

// MARK: - Geometry scanning protocol

/// The subset of `APIClient`'s geometry API surface needed by
/// `PieceScanController`. Defined here (not in `APIClient.swift`) so the
/// controller file is self-contained, and so fakes in tests can conform to it
/// without touching the real network client.
///
/// The `enrollUncertain` parameter on `uploadPieceGeometry` has a default on
/// `APIClient` itself (= false); the protocol declares it without a default so
/// the concrete call site always makes the intent explicit and tests can
/// intercept both paths without default-argument ambiguity.
@MainActor
protocol PieceGeometryScanning {
  func uploadPieceGeometry(
    puzzleId: String, jpegData: Data, enrollUncertain: Bool
  ) async throws -> PieceGeometryUploadResponse
  func listPieceGeometry(puzzleId: String) async throws -> PieceGeometryListResponse
}

// Conform `APIClient` to the protocol. The existing method already carries the
// same signature with a default argument, so the protocol witness resolves to
// it directly — the extension only names the conformance, nothing else.
extension APIClient: PieceGeometryScanning {}

// MARK: - Haptic

/// Distinct scan outcomes mapped to distinct feedback patterns. Having an enum
/// rather than a raw `UINotificationFeedbackGenerator.FeedbackType` lets the
/// scanner's protocol express intent — not just "what kind of buzz" but "which
/// scan event" — and lets the test fake record the semantic outcome without
/// importing UIKit.
enum ScanHaptic {
  case success
  case warning
  case failure
}

// MARK: - State types

/// The scan-and-lock state machine has four coarse phases matching the main
/// actor's responsibility lifecycle: watching → capturing → showing the result
/// → back to watching.
enum ScanPhase: Equatable {
  case scanning
  case capturing
  case verdict(ScanVerdict)
}

/// Terminal outcome of one capture–upload cycle. Each case is shown to the
/// user with a distinct colour / message, and the re-arm duration is keyed to
/// it.
enum ScanVerdict: Equatable {
  /// First time this piece was seen: enrolled and added to the gallery.
  case locked(pieceId: String, edgeTypes: [GeometryEdgeType])
  /// This piece was already in the store — show which one it matched.
  case alreadyScanned(pieceId: String)
  /// The upload succeeded but the store isn't sure whether this is a new or
  /// duplicate piece. The user can confirm ("Add as new") via
  /// `confirmUncertainAsNew()`.
  case uncertain
  /// The photo itself couldn't be measured (contour quality gate failed —
  /// e.g. clutter merged into the silhouette), so there is no fingerprint to
  /// compare or enroll. Distinct from `.uncertain`: offering "add as new"
  /// here would be a dead end, because `on_uncertain=enroll` has nothing to
  /// enroll — the only way forward is a better capture.
  case unreadable
  /// Network/server error or a nil capture.
  case failure(String)
}

// MARK: - Gallery item

/// One enrolled piece in the local gallery, kept between scan-and-lock cycles
/// so the bottom strip is always up to date.
struct ScannedPiece: Identifiable, Equatable {
  let pieceId: String
  let edgeTypes: [GeometryEdgeType]
  /// The full-res JPEG captured at lock time. `nil` for pieces loaded from the
  /// server list (`loadEnrolled`), where no local capture happened.
  let thumbnailJPEG: Data?

  /// `Identifiable` conformance — the backend's stable piece id.
  var id: String { pieceId }
}

// MARK: - Controller

/// Scan-and-lock state machine. Drives the camera's stability tracker and
/// maps the geometry API's responses to the verdict UI.
///
/// Dependencies are injected at init so unit tests can run deterministically
/// without a real camera, network, or clock:
/// - `geometryClient` — real `APIClient` in production, a recording fake in tests.
/// - `capture` — `async` closure that returns a full-res JPEG; tests can return
///   fixture bytes without touching any camera.
/// - `haptic` — test recorder instead of UIKit's feedback generator.
/// - `sleep` — `Task.sleep` by default; tests inject an instant no-op so the
///   re-arm delay doesn't pause the test suite.
/// - `tracker` — the stability detector; default-init is fine for production,
///   and tests can call `ingest` directly rather than faking it.
@MainActor
@Observable
final class PieceScanController {

  // MARK: - Re-arm delays

  /// How long the verdict banner stays visible before the scanner re-arms.
  /// `internal` so tests can reference them without magic numbers.
  static let rearmDelayLocked: TimeInterval = 1.5
  static let rearmDelayAlreadyScanned: TimeInterval = 2.0
  static let rearmDelayFailure: TimeInterval = 2.0
  /// Uncertain returns to scanning almost immediately — the "Add as new" chip
  /// below provides the user's second opinion independently of the re-arm.
  static let rearmDelayUncertain: TimeInterval = 0.6
  /// Long enough to read "couldn't read the piece" and reposition, short
  /// enough that scanning feels continuous.
  static let rearmDelayUnreadable: TimeInterval = 1.2

  // MARK: - Dependencies

  /// Read fresh for every request rather than captured once: the backend's
  /// puzzle store is in-memory, so a backend restart invalidates the id
  /// mid-session and `recoverPuzzle` (re-upload) mints a new one — the
  /// retry must see it.
  private let puzzleId: () -> String
  private let geometryClient: any PieceGeometryScanning
  /// Returns the full-res JPEG for the current frame, already downscaled and
  /// encoded. Nil means the capture failed (e.g. no camera in the Simulator
  /// when the DEBUG capture-image isn't set).
  private let capture: () async -> Data?
  private let haptic: (ScanHaptic) -> Void
  /// Called when a geometry POST 404s (the backend forgot this puzzle — its
  /// store is in-memory and a restart wipes it mid-session). Should obtain a
  /// fresh puzzle id (`SolveSession.reupload`) and return whether it
  /// succeeded; the POST is then retried once against `puzzleId()`. Without
  /// this, every auto-lock on a stale session dead-ends in a red
  /// "Puzzle not found" banner — found by the user on the first device run.
  private let recoverPuzzle: (() async -> Bool)?
  /// Called once per enrolled piece (a `new` verdict, auto or via the
  /// uncertain-confirm) with the geometry piece id and the captured JPEG.
  /// The view wires this to `SolveSession.enqueueScanned`, which is what
  /// puts scanned pieces into the puzzle page's piece list and persists
  /// their photos — the scan gallery restores its thumbnails from those
  /// entries via `thumbnailForPiece`.
  private let onEnrolled: ((String, Data) -> Void)?
  /// Locally stored photo for a geometry piece id, for gallery items whose
  /// piece was scanned in an earlier scanner visit (the server list carries
  /// no images). Nil (absent or no match) falls back to the placeholder tile.
  private let thumbnailForPiece: ((String) -> Data?)?
  /// Injected sleep so tests can override with a no-op instant function and
  /// avoid any real `Task.sleep` pauses.
  private let sleep: (TimeInterval) async -> Void
  private var tracker: PieceScanStabilityTracker

  // MARK: - Observable state

  private(set) var phase: ScanPhase = .scanning
  private(set) var gallery: [ScannedPiece] = []
  /// Non-nil while the user can still confirm an uncertain verdict as a new
  /// piece. Cleared either by `confirmUncertainAsNew()` or
  /// `dismissUncertainChip()`.
  private(set) var pendingUncertainJPEG: Data?
  /// Set to the matching piece's id on an `.alreadyScanned` verdict so the
  /// gallery strip can briefly highlight it. Cleared when the scanner re-arms.
  private(set) var lastMatchedPieceId: String?

  // MARK: - Init

  init(
    puzzleId: @escaping () -> String,
    geometryClient: any PieceGeometryScanning,
    capture: @escaping () async -> Data?,
    haptic: @escaping (ScanHaptic) -> Void = pieceScanDefaultHaptic,
    recoverPuzzle: (() async -> Bool)? = nil,
    onEnrolled: ((String, Data) -> Void)? = nil,
    thumbnailForPiece: ((String) -> Data?)? = nil,
    sleep: @escaping (TimeInterval) async -> Void = pieceScanDefaultSleep,
    tracker: PieceScanStabilityTracker = PieceScanStabilityTracker()
  ) {
    self.puzzleId = puzzleId
    self.geometryClient = geometryClient
    self.capture = capture
    self.haptic = haptic
    self.recoverPuzzle = recoverPuzzle
    self.onEnrolled = onEnrolled
    self.thumbnailForPiece = thumbnailForPiece
    self.sleep = sleep
    self.tracker = tracker
  }

  // MARK: - Streaming feed

  /// Called for every preview update. Guards on `.scanning` so mid-capture or
  /// mid-verdict calls are silently dropped — the tracker's own latch (once
  /// fired, ignores further lockable detections until `reset()`) is a second
  /// line of defence, but the phase guard is cheaper.
  func ingest(_ state: PiecePreviewState, at date: Date) {
    guard phase == .scanning else { return }
    if tracker.ingest(state, at: date) {
      Task { await lockAndPost() }
    }
  }

  // MARK: - Core capture + upload flow

  private func lockAndPost() async {
    phase = .capturing
    guard let jpeg = await capture() else {
      setVerdict(.failure("Could not capture the piece."))
      haptic(.failure)
      scheduleRearm(delay: Self.rearmDelayFailure)
      return
    }
    await post(jpeg: jpeg, enrollUncertain: false)
  }

  /// Shared POST path used by both the auto-capture and the uncertain-confirm
  /// flows. Caller is responsible for clearing `pendingUncertainJPEG` before
  /// entering here (to close the race window for a double-tap confirm).
  ///
  /// A 404 means the backend forgot this puzzle (in-memory store, restart) —
  /// recover a fresh puzzle id and retry once, so a stale saved session heals
  /// itself on the first lock instead of red-bannering every scan.
  private func post(jpeg: Data, enrollUncertain: Bool, isRetry: Bool = false) async {
    do {
      let response = try await geometryClient.uploadPieceGeometry(
        puzzleId: puzzleId(), jpegData: jpeg, enrollUncertain: enrollUncertain)
      applyResponse(response, jpeg: jpeg)
    } catch let error as APIError where error.status == 404 && !isRetry {
      if let recoverPuzzle, await recoverPuzzle() {
        await post(jpeg: jpeg, enrollUncertain: enrollUncertain, isRetry: true)
        return
      }
      setVerdict(.failure("Puzzle session expired. Close the scanner and re-add the puzzle."))
      haptic(.failure)
      scheduleRearm(delay: Self.rearmDelayFailure)
    } catch {
      setVerdict(.failure(error.localizedDescription))
      haptic(.failure)
      scheduleRearm(delay: Self.rearmDelayFailure)
    }
  }

  private func applyResponse(_ response: PieceGeometryUploadResponse, jpeg: Data) {
    switch response.status {
    case .new:
      let pieceId = response.pieceId ?? UUID().uuidString
      let edgeTypes = response.edgeTypes
      gallery.append(ScannedPiece(pieceId: pieceId, edgeTypes: edgeTypes, thumbnailJPEG: jpeg))
      pendingUncertainJPEG = nil
      onEnrolled?(pieceId, jpeg)
      setVerdict(.locked(pieceId: pieceId, edgeTypes: edgeTypes))
      haptic(.success)
      scheduleRearm(delay: Self.rearmDelayLocked)

    case .matched:
      let matchId = response.matchPieceId ?? response.pieceId ?? "?"
      pendingUncertainJPEG = nil
      lastMatchedPieceId = matchId
      setVerdict(.alreadyScanned(pieceId: matchId))
      haptic(.warning)
      scheduleRearm(delay: Self.rearmDelayAlreadyScanned)

    case .uncertain:
      // Two flavors share the wire status. A genuine gray-zone verdict
      // always carries the closest match's z-score (a comparison was made);
      // a nil z-score means the pipeline failed on this photo (no clean
      // contour → no fingerprint), and re-posting the same bytes with
      // on_uncertain=enroll would loop forever — found the hard way in the
      // M10 E2E, where a cluttered photo kept the confirm chip alive with
      // nothing behind it.
      guard response.zScore != nil else {
        setVerdict(.unreadable)
        scheduleRearm(delay: Self.rearmDelayUnreadable)
        return
      }
      // Keep the JPEG so the user can confirm it as a new piece.
      // No haptic — an alarming buzz would be counter-productive here.
      pendingUncertainJPEG = jpeg
      setVerdict(.uncertain)
      scheduleRearm(delay: Self.rearmDelayUncertain)
    }
  }

  // MARK: - Uncertain confirm

  /// Enrolls the pending uncertain piece as a new piece. A no-op if
  /// `pendingUncertainJPEG` is already nil (double-tap protection: the first
  /// call clears it before awaiting the network, so a second call finds nil
  /// and returns immediately).
  func confirmUncertainAsNew() async {
    guard let jpeg = pendingUncertainJPEG else { return }
    // Clear immediately — before the await — so a second tap while the
    // request is in flight is a clean no-op.
    pendingUncertainJPEG = nil
    phase = .capturing
    await post(jpeg: jpeg, enrollUncertain: true)
  }

  /// Removes the uncertain chip without enrolling the piece.
  func dismissUncertainChip() {
    pendingUncertainJPEG = nil
  }

  #if DEBUG
    /// The currently presented controller, so `pusseldebug://scanconfirm` can
    /// drive the uncertain-confirm chip without a synthetic tap (Simulator
    /// windows are often untappable in headless sessions — see
    /// `PieceCameraSession.debugActive` for the precedent). Registered by
    /// `PieceScanView.task`, cleared on disappear.
    static weak var debugActive: PieceScanController?
  #endif

  // MARK: - Gallery pre-fill

  /// Fetches the already-enrolled pieces from the server and merges them into
  /// the gallery. Pieces already present (matched by `pieceId`) keep their
  /// local thumbnails; pieces known only to the server are appended with
  /// `thumbnailJPEG: nil`. Errors are silently swallowed — an empty gallery
  /// strip on open is cosmetic, and the next POST will surface any real backend
  /// problem.
  func loadEnrolled() async {
    guard let response = try? await geometryClient.listPieceGeometry(puzzleId: puzzleId())
    else { return }
    let knownIds = Set(gallery.map(\.pieceId))
    for piece in response.pieces where !knownIds.contains(piece.pieceId) {
      gallery.append(
        ScannedPiece(
          pieceId: piece.pieceId,
          edgeTypes: piece.edgeTypes,
          // Pieces enrolled in an earlier scanner visit were enqueued into
          // the session's piece list at lock time — recover their photos
          // from there rather than showing a placeholder.
          thumbnailJPEG: thumbnailForPiece?(piece.pieceId)))
    }
  }

  // MARK: - Private helpers

  private func setVerdict(_ verdict: ScanVerdict) {
    phase = .verdict(verdict)
  }

  /// Schedules re-arm after the verdict display duration. Uses the injected
  /// `sleep` closure so tests can override with an instant function and avoid
  /// real pauses. Guards that `phase` is still in the verdict that scheduled
  /// this rearm before resetting — a `confirmUncertainAsNew()` call that
  /// arrives during the uncertain window will already have moved the phase to
  /// `.capturing`, so we must not then stomp it back to `.scanning`.
  private func scheduleRearm(delay: TimeInterval) {
    let verdictAtSchedule = phase
    Task {
      await self.sleep(delay)
      // Only reset if the phase hasn't been advanced by another action (e.g.
      // confirmUncertainAsNew changed .uncertain → .capturing before we woke).
      guard self.phase == verdictAtSchedule else { return }
      self.lastMatchedPieceId = nil
      self.tracker.reset()
      self.phase = .scanning
    }
  }

}

// MARK: - Default injectable dependencies

/// Default haptic handler used by `PieceScanController`. Defined at module
/// scope rather than as a static method so it can be passed as a default
/// argument without the "covariant Self in default expression" restriction.
private func pieceScanDefaultHaptic(_ kind: ScanHaptic) {
  let type: UINotificationFeedbackGenerator.FeedbackType
  switch kind {
  case .success: type = .success
  case .warning: type = .warning
  case .failure: type = .error
  }
  UINotificationFeedbackGenerator().notificationOccurred(type)
}

/// Default sleep closure: wraps `Task.sleep` in a non-throwing shell so it
/// can serve as a plain `(TimeInterval) async -> Void` default argument.
private func pieceScanDefaultSleep(_ seconds: TimeInterval) async {
  try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
}
