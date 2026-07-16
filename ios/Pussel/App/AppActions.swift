import PhotosUI
import SwiftUI
import UIKit

/// Wizard actions shared by the UI and the debug deep links.
extension AppModel {
  /// Default label for a new puzzle — the capture date/time.
  private static let puzzleNameFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.dateStyle = .medium
    formatter.timeStyle = .short
    return formatter
  }()

  /// Detects the puzzle frame and moves to the confirm-trim step.
  func startTrim(image: UIImage, source: CaptureSource) async {
    guard let jpeg = ImageUtilities.normalizedJPEG(from: image) else {
      flow.errorMessage = "Could not process the photo."
      return
    }
    flow.errorMessage = nil
    flow.isBusy = true
    defer { flow.isBusy = false }
    // Kept alongside the upload-size copy so the accepted trim can be re-warped
    // sharply for the zoom viewer. Encoded off the main actor and in parallel
    // with the detection round-trip, so a big photo neither freezes the screen
    // nor delays the trim. Best-effort: a nil here costs zoom detail, not the
    // capture.
    async let zoomSourceJPEG = Task.detached(priority: .userInitiated) {
      ImageUtilities.normalizedJPEG(
        from: image, maxDimension: ImageUtilities.zoomSourceMaxDimension, quality: 0.85)
    }.value
    do {
      let detection = try await api.detectFrame(jpegData: jpeg)
      flow.phase = .confirmTrim(
        TrimCandidate(
          rawJPEG: jpeg, zoomSourceJPEG: await zoomSourceJPEG, detection: detection, source: source)
      )
    } catch {
      flow.errorMessage = error.localizedDescription
    }
  }

  /// Uploads the accepted trimmed image and starts a solve session. Any
  /// user-applied rotation (`quarterTurns` × 90° clockwise) is baked into the
  /// uploaded and stored image so the puzzle appears upright everywhere.
  /// `pieceCount` is the user-entered total piece count; the grid (rows,
  /// cols) is estimated from it and the final rotated image's pixel
  /// dimensions, then persisted on the session for overlay marker sizing.
  func acceptTrim(_ candidate: TrimCandidate, quarterTurns: Int = 0, pieceCount: Int) async {
    // Ignore repeat taps (e.g. a double-tap on "Use This") while an upload
    // is already in flight so we don't start concurrent uploads/sessions.
    guard !flow.isBusy else { return }
    guard let decoded = candidate.trimmedJPEG else {
      flow.errorMessage = "Could not decode the trimmed image."
      return
    }
    guard let trimmed = ImageUtilities.rotatedJPEG(from: decoded, quarterTurns: quarterTurns) else {
      flow.errorMessage = "Could not rotate the trimmed image."
      return
    }
    guard let trimmedSize = UIImage(data: trimmed)?.size else {
      flow.errorMessage = "Could not measure the trimmed image."
      return
    }
    let grid = GridEstimator.estimate(
      pieceCount: pieceCount, imageWidth: trimmedSize.width, imageHeight: trimmedSize.height)
    flow.errorMessage = nil
    flow.isBusy = true
    defer { flow.isBusy = false }
    // Started alongside the upload rather than before it: the warp is pure
    // local CPU and the upload is pure waiting, so overlapping them hides the
    // zoom copy's cost behind the network entirely. Nothing downstream needs
    // it until the session is built.
    async let displayJPEG = zoomCopy(of: candidate, quarterTurns: quarterTurns)
    do {
      let response = try await api.uploadPuzzle(jpegData: trimmed, pieceCount: pieceCount)
      let session = SolveSession(
        name: Self.puzzleNameFormatter.string(from: Date()),
        puzzleId: response.puzzleId,
        trimmedJPEG: trimmed,
        displayJPEG: await displayJPEG,
        targetPieceCount: pieceCount,
        rows: grid.rows,
        cols: grid.cols,
        store: store
      )
      // Persist immediately so the puzzle survives even before any pieces
      // are added and shows up on the home screen right away.
      session.persist()
      flow.phase = .solving(session)
    } catch {
      flow.errorMessage = error.localizedDescription
    }
  }

  /// Re-warps the kept zoom-quality photo to the accepted trim, so the solve
  /// screen can zoom into real detail instead of magnifying the
  /// upload-resolution crop the backend returned.
  ///
  /// Uses the corners from the same detect-frame response the accepted
  /// preview came from, so both copies frame the same region; `quarterTurns`
  /// is the user's rotation, baked in here exactly as it is for the uploaded
  /// image so the two stay in the same orientation.
  ///
  /// Best-effort throughout: every failure returns nil, leaving the session to
  /// fall back to `trimmedJPEG` rather than blocking a working capture over a
  /// display nicety.
  ///
  /// Runs off the main actor: decoding a multi-megapixel photo, warping it and
  /// encoding the result takes long enough that doing it inline would freeze
  /// the screen — including the very spinner meant to cover it.
  private func zoomCopy(of candidate: TrimCandidate, quarterTurns: Int) async -> Data? {
    guard let source = candidate.zoomSourceJPEG else { return nil }
    let corners = candidate.detection.corners
    return await Task.detached(priority: .userInitiated) {
      guard let image = UIImage(data: source),
        let corrected = ImageUtilities.perspectiveCorrected(from: image, corners: corners)
      else {
        return nil
      }
      // Rotate before encoding, so the copy is compressed once rather than
      // decoded and re-encoded to turn it.
      return ImageUtilities.normalizedJPEG(
        from: ImageUtilities.rotated(corrected, quarterTurns: quarterTurns),
        maxDimension: ImageUtilities.zoomMaxDimension, quality: 0.85)
    }.value
  }

  /// Queues a captured piece photo for prediction in the current session.
  func addPiece(image: UIImage) {
    guard case .solving(let session) = flow.phase else { return }
    guard let jpeg = ImageUtilities.normalizedJPEG(from: image, maxDimension: 1600, quality: 0.9)
    else {
      session.errorMessage = "Could not process the photo."
      return
    }
    session.errorMessage = nil
    session.enqueue(jpeg: jpeg, api: api)
  }

  /// Surfaces a piece-capture failure on the solve screen. The capture UI
  /// sits in a full-screen cover, so callers dismiss after reporting.
  func reportPieceError(_ message: String) {
    guard case .solving(let session) = flow.phase else { return }
    session.errorMessage = message
  }

  /// Adds a piece picked from the photo library. Both piece pickers route
  /// here so a failed load reports itself on the solve screen instead of
  /// looking like a dropped tap.
  func addPiece(from item: PhotosPickerItem) async {
    guard case .solving(let session) = flow.phase else { return }
    guard let data = try? await item.loadTransferable(type: Data.self),
      let image = UIImage(data: data)
    else {
      session.errorMessage = "Could not load the selected photo."
      return
    }
    addPiece(image: image)
  }

  /// Reopens a stored puzzle from disk and resumes solving it. Pieces that
  /// were never predicted are re-queued; if the server forgot this puzzle_id
  /// (in-memory store restarted) the solve view's expired banner recovers it.
  func openPuzzle(_ id: UUID) {
    guard let session = store.loadSession(id: id) else {
      flow.errorMessage = "Could not open that puzzle."
      return
    }
    flow.errorMessage = nil
    flow.phase = .solving(session)
    session.resume(api: api)
  }

  /// Deletes a stored puzzle, leaving a brief window to undo it. If it is the
  /// one being solved, returns to the capture screen.
  func deletePuzzle(_ id: UUID) {
    if case .solving(let session) = flow.phase, session.id == id {
      flow.reset()
    }
    store.deleteWithUndo(id)
  }

  /// Restores the most recently deleted puzzle, if its undo window is open.
  func undoDelete() {
    store.undoDelete()
  }
}
