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

/// Response of GET /api/v1/puzzle/barcode/{ean} — the Ravensburger box-image
/// lookup for a scanned EAN-13. `found == false` (never an HTTP error) means
/// the code isn't a (known) Ravensburger product.
struct BarcodeLookupResponse: Codable, Equatable {
  let found: Bool
  /// data:image/jpeg;base64,… of the box image; present when found.
  let boxImage: String?
  /// The resolved Ravensburger article number; present when found.
  let articleNumber: String?
  /// Piece count the backend OCR'd off the box shot, for prefilling the
  /// piece-count input; nil when the box couldn't be read confidently.
  let pieceCountEstimate: Int?
}

/// Response of POST /api/v1/puzzle/upload.
struct PuzzleUploadResponse: Codable, Equatable {
  let puzzleId: String
  let imageUrl: String?
  /// Backend-computed grid, decoded for parity but not used for display —
  /// the app always sizes the overlay from its own local `GridEstimator`
  /// estimate (see `AppActions.acceptTrim`).
  let pieceCount: Int?
  let rows: Int?
  let cols: Int?
}
