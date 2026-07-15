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
        do {
            let detection = try await api.detectFrame(jpegData: jpeg)
            flow.phase = .confirmTrim(TrimCandidate(rawJPEG: jpeg, detection: detection, source: source))
        } catch {
            flow.errorMessage = error.localizedDescription
        }
    }

    /// Uploads the accepted trimmed image and starts a solve session. Any
    /// user-applied rotation (`quarterTurns` × 90° clockwise) is baked into the
    /// uploaded and stored image so the puzzle appears upright everywhere.
    func acceptTrim(_ candidate: TrimCandidate, quarterTurns: Int = 0) async {
        guard let decoded = candidate.trimmedJPEG,
              let trimmed = ImageUtilities.rotatedJPEG(from: decoded, quarterTurns: quarterTurns) else {
            flow.errorMessage = "Could not decode the trimmed image."
            return
        }
        flow.errorMessage = nil
        flow.isBusy = true
        defer { flow.isBusy = false }
        do {
            let response = try await api.uploadPuzzle(jpegData: trimmed)
            let session = SolveSession(
                name: Self.puzzleNameFormatter.string(from: Date()),
                puzzleId: response.puzzleId,
                trimmedJPEG: trimmed,
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

    /// Queues a captured piece photo for prediction in the current session.
    func addPiece(image: UIImage) {
        guard case .solving(let session) = flow.phase,
              let jpeg = ImageUtilities.normalizedJPEG(from: image, maxDimension: 1600, quality: 0.9) else { return }
        session.enqueue(jpeg: jpeg, api: api)
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

    /// Permanently deletes a stored puzzle. If it is the one being solved,
    /// returns to the capture screen.
    func deletePuzzle(_ id: UUID) {
        if case .solving(let session) = flow.phase, session.id == id {
            flow.reset()
        }
        store.delete(id)
    }
}
