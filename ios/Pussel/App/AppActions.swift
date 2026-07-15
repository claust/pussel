import UIKit

/// Wizard actions shared by the UI and the debug deep links.
extension AppModel {
    /// Detects the puzzle frame and moves to the confirm-trim step.
    func startTrim(image: UIImage) async {
        guard let jpeg = ImageUtilities.normalizedJPEG(from: image) else {
            flow.errorMessage = "Could not process the photo."
            return
        }
        flow.errorMessage = nil
        flow.isBusy = true
        defer { flow.isBusy = false }
        do {
            let detection = try await api.detectFrame(jpegData: jpeg)
            flow.phase = .confirmTrim(TrimCandidate(rawJPEG: jpeg, detection: detection))
        } catch {
            flow.errorMessage = error.localizedDescription
        }
    }

    /// Uploads the accepted trimmed image and starts a solve session.
    func acceptTrim(_ candidate: TrimCandidate) async {
        guard let trimmed = candidate.trimmedJPEG else {
            flow.errorMessage = "Could not decode the trimmed image."
            return
        }
        flow.errorMessage = nil
        flow.isBusy = true
        defer { flow.isBusy = false }
        do {
            let response = try await api.uploadPuzzle(jpegData: trimmed)
            flow.phase = .solving(SolveSession(puzzleId: response.puzzleId, trimmedJPEG: trimmed))
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
}
