import Foundation

/// A point in [0, 1] coordinates relative to the image it belongs to.
struct NormalizedPoint: Codable, Equatable {
    let x: Double
    let y: Double
}

struct QuadCorners: Codable, Equatable {
    let topLeft: NormalizedPoint
    let topRight: NormalizedPoint
    let bottomRight: NormalizedPoint
    let bottomLeft: NormalizedPoint
}

/// Response of POST /api/v1/puzzle/detect-frame.
struct DetectFrameResponse: Codable, Equatable {
    /// data:image/jpeg;base64,... of the trimmed, perspective-corrected puzzle.
    let trimmedImage: String
    let corners: QuadCorners
    let confidence: Double
}

/// Response of POST /api/v1/puzzle/upload.
struct PuzzleUploadResponse: Codable, Equatable {
    let puzzleId: String
    let imageUrl: String?
}
