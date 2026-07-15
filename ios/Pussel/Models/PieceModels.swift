import Foundation

/// Response of POST /api/v1/puzzle/{id}/piece.
struct PieceResponse: Codable, Equatable {
    /// Predicted piece center, normalized to the trimmed puzzle image.
    let position: NormalizedPoint
    let positionConfidence: Double
    /// One of 0, 90, 180, 270 (degrees).
    let rotation: Int
    let rotationConfidence: Double
    /// data:image/png;base64,... with background removed, when available.
    let cleanedImage: String?
}

/// A captured piece moving through the prediction queue.
struct CaptureEntry: Identifiable, Equatable {
    enum Status: Equatable {
        case queued
        case predicting
        case done
        case error(String)
    }

    let id: UUID
    /// JPEG sent to the backend; kept for retry.
    let uploadJPEG: Data
    /// Image shown in the UI — the raw capture until a cleaned PNG replaces it.
    var displayImage: Data
    var status: Status = .queued
    var result: PieceResponse?

    init(jpeg: Data) {
        self.id = UUID()
        self.uploadJPEG = jpeg
        self.displayImage = jpeg
    }
}
