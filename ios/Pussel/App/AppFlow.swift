import Foundation
import Observation

/// The linear real-mode wizard, mirroring the web app's RealModePhase
/// (frontend/src/app/real/page.tsx) minus the corner-adjust step.
enum AppPhase {
  case capturePuzzle
  case confirmTrim(TrimCandidate)
  case solving(SolveSession)
}

/// How the puzzle overview photo was supplied, so "Retake" can reopen the
/// same picker instead of dropping back to the chooser screen.
enum CaptureSource {
  case camera
  case library
}

/// A detect-frame result awaiting user confirmation.
struct TrimCandidate {
  /// The photo as uploaded for detection — capped at the upload size, so the
  /// detected `corners` are normalized to this image's frame.
  let rawJPEG: Data
  /// The same photo kept at zoom quality, to re-warp into a sharp display
  /// copy once the trim is accepted (see `AppModel.acceptTrim`). Nil when the
  /// higher-resolution encode failed; the trim then falls back to the
  /// server's upload-resolution crop.
  ///
  /// This is the *source photo*, not the crop taken from it — hence the
  /// `zoomSource` naming it shares with `ImageUtilities.zoomSourceMaxDimension`,
  /// which caps it. `SolveSession.zoomJPEG` is the other end of the pipeline:
  /// the straightened crop that actually gets drawn.
  let zoomSourceJPEG: Data?
  let detection: DetectFrameResponse
  let source: CaptureSource

  var trimmedJPEG: Data? {
    ImageUtilities.decodeDataURL(detection.trimmedImage)
  }
}

@Observable
@MainActor
final class AppFlowStore {
  var phase: AppPhase = .capturePuzzle
  var isBusy = false
  var errorMessage: String?

  /// Set when the user taps "Retake" so CapturePuzzleView reopens the same
  /// picker on appear; cleared once consumed.
  var pendingRetake: CaptureSource?

  func reset() {
    phase = .capturePuzzle
    isBusy = false
    errorMessage = nil
    pendingRetake = nil
  }
}

/// State for one solving session. The trimmed image is kept so the puzzle can
/// be re-uploaded for a fresh puzzle_id when the backend restarts (its puzzle
/// store is in-memory, so a stored id can 404 mid-session). Every meaningful
/// change is mirrored to disk through `store` so nothing is lost when the app
/// is closed (see PuzzleStore).
@Observable
@MainActor
final class SolveSession {
  /// Stable local identity used as the on-disk folder name.
  let id: UUID
  /// Human-friendly label shown on the home screen (defaults to a date).
  var name: String
  let createdAt: Date
  var puzzleId: String
  let trimmedJPEG: Data
  /// A higher-resolution copy of `trimmedJPEG` for the zoom viewer, or nil
  /// for puzzles captured before it was kept (and when the local re-warp
  /// failed). Never uploaded — `trimmedJPEG` remains the image the backend
  /// predicted against. See `zoomJPEG`.
  let displayJPEG: Data?
  /// Total piece count entered by the user when the puzzle was added —
  /// the puzzle's target size, not the number of captured pieces (which
  /// `PuzzleSummary.pieceCount` reports).
  let targetPieceCount: Int
  /// Grid estimated by `GridEstimator` from `targetPieceCount` and `trimmedJPEG`'s
  /// pixel dimensions; drives the overlay's marker sizing.
  let rows: Int
  let cols: Int
  var entries: [CaptureEntry] = []
  var isProcessing = false
  var puzzleExpired = false
  var errorMessage: String?

  @ObservationIgnored private let store: PuzzleStore?

  init(
    id: UUID = UUID(),
    name: String,
    puzzleId: String,
    trimmedJPEG: Data,
    displayJPEG: Data? = nil,
    targetPieceCount: Int,
    rows: Int,
    cols: Int,
    createdAt: Date = Date(),
    store: PuzzleStore? = nil
  ) {
    self.id = id
    self.name = name
    self.puzzleId = puzzleId
    self.trimmedJPEG = trimmedJPEG
    self.displayJPEG = displayJPEG
    self.targetPieceCount = targetPieceCount
    self.rows = rows
    self.cols = cols
    self.createdAt = createdAt
    self.store = store
  }

  /// Writes the current state to disk. Cheap and idempotent.
  func persist() {
    store?.save(self)
  }

  var placedEntries: [CaptureEntry] {
    entries.filter { $0.result != nil }
  }

  /// Bytes the zoom viewer draws: the sharp copy when there is one, otherwise
  /// the upload-resolution crop. The inline overlay deliberately stays on
  /// `trimmedJPEG` — it is drawn a few hundred points wide, so decoding the
  /// zoom copy for it would cost memory it can't show.
  var zoomJPEG: Data { displayJPEG ?? trimmedJPEG }

  func enqueue(jpeg: Data, api: APIClient) {
    entries.append(CaptureEntry(jpeg: jpeg))
    persist()
    processNext(api: api)
  }

  /// Resumes any pieces left `.queued` from a reloaded session.
  func resume(api: APIClient) {
    processNext(api: api)
  }

  func retry(id: UUID, api: APIClient) {
    guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
    entries[index].status = .queued
    processNext(api: api)
  }

  func remove(id: UUID) {
    entries.removeAll { $0.id == id }
    persist()
  }

  /// Serially drains the queue — one in-flight prediction at a time, like
  /// the web's usePredictionWorker (rembg segmentation is the slow step).
  func processNext(api: APIClient) {
    guard !isProcessing, let index = entries.firstIndex(where: { $0.status == .queued }) else {
      return
    }
    isProcessing = true
    let entry = entries[index]
    entries[index].status = .predicting
    Task {
      await self.process(entry: entry, api: api)
      self.isProcessing = false
      self.processNext(api: api)
    }
  }

  private func process(entry: CaptureEntry, api: APIClient) async {
    do {
      let piece = try await api.processPiece(puzzleId: puzzleId, jpegData: entry.uploadJPEG)
      update(id: entry.id) { current in
        current.status = .done
        current.result = piece
        if let cleaned = piece.cleanedImage, let data = ImageUtilities.decodeDataURL(cleaned) {
          current.displayImage = data
        }
      }
      persist()
    } catch let error as APIError where error.status == 404 {
      puzzleExpired = true
      update(id: entry.id) { $0.status = .expired }
    } catch {
      update(id: entry.id) { $0.status = .error(error.localizedDescription) }
    }
  }

  /// Gets a fresh puzzle_id for the kept trimmed image after a backend
  /// restart, then re-queues entries that failed on the dead id.
  func reupload(api: APIClient) async {
    errorMessage = nil
    do {
      let response = try await api.uploadPuzzle(jpegData: trimmedJPEG, pieceCount: targetPieceCount)
      puzzleId = response.puzzleId
      puzzleExpired = false
      for index in entries.indices where entries[index].status == .expired {
        entries[index].status = .queued
      }
      persist()
      processNext(api: api)
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  private func update(id: UUID, _ mutate: (inout CaptureEntry) -> Void) {
    guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
    mutate(&entries[index])
  }
}
