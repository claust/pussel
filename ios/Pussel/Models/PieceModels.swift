import Foundation

/// Normalized [0,1] fraction of the puzzle's width/height covered by the
/// full piece image frame — the image as sent/displayed, in its own
/// orientation (rotation NOT factored out). Present only when the backend
/// matcher measured the piece (SIFT/NCC); nil for the CNN path or a matcher
/// failure fallback.
struct PieceSpan: Codable, Equatable {
  let width: Double
  let height: Double
}

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
  /// Measured size of the full piece image frame, when available. See
  /// `PieceSpan`.
  let pieceSpan: PieceSpan?
}

/// Bounding box normalized to [0,1] image coordinates, mirroring the
/// backend's `BoundingBox` model.
struct NormalizedBoundingBox: Codable, Equatable {
  let x: Double
  let y: Double
  let width: Double
  let height: Double
}

/// Response of POST /api/v1/piece/preview — live piece-region detection in
/// a downscaled camera frame, streamed from `PiecePreviewStreamer`.
struct PiecePreviewResponse: Codable, Equatable {
  let found: Bool
  /// Outline of the detected region, normalized to the frame that was
  /// submitted. Empty when `found` is false.
  let polygon: [NormalizedPoint]
  let bbox: NormalizedBoundingBox?
  let confidence: Double
  /// Best-effort piece-geometry quality flag from a quick corner-detection
  /// pass; only populated when the request opted in with
  /// `include_quality=true` (see `APIClient.previewPiece`).
  let lockable: Bool?
  /// Whether the quick corner cross-check disagreed; only populated with
  /// `include_quality=true`.
  let cornerDisagreement: Bool?
}

/// A captured piece moving through the prediction queue.
struct CaptureEntry: Identifiable, Equatable {
  enum Status: Equatable {
    case queued
    case predicting
    case done
    /// The backend forgot the puzzle_id (restart) — recoverable via re-upload.
    case expired
    case error(String)

    var isRetryable: Bool {
      switch self {
      case .expired, .error: return true
      case .queued, .predicting, .done: return false
      }
    }
  }

  let id: UUID
  /// JPEG sent to the backend; kept for retry.
  let uploadJPEG: Data
  /// Image shown in the UI — the raw capture until a cleaned PNG replaces it.
  /// `didSet` keeps `trimmedDisplayImage` in sync when the cleaned PNG
  /// arrives (`SolveSession.process`); init assignments don't fire it, so
  /// both inits set the trimmed copy explicitly.
  var displayImage: Data {
    didSet { trimmedDisplayImage = ImageUtilities.alphaTrimmedPNG(from: displayImage) }
  }
  /// `displayImage` cropped to its alpha bounding box, trimming the
  /// backend's ~8% transparent margin around a cleaned piece PNG. Computed
  /// once whenever `displayImage` changes, rather than per render —
  /// PuzzleOverlayView reads it on every layout pass. Equal to
  /// `displayImage` itself for images with no transparent margin to trim
  /// (e.g. the raw JPEG capture before a cleaned PNG replaces it).
  var trimmedDisplayImage: Data
  var status: Status = .queued
  var result: PieceResponse?
  /// The geometry store's piece id when this entry was captured by the M10
  /// scan-and-lock flow (nil for shutter/library captures). Links the entry
  /// to the scan gallery, which uses it to restore thumbnails for pieces
  /// enrolled in an earlier scanner visit. Cleared on `reupload` — the
  /// backend's geometry store died with the old puzzle id, so the ids it
  /// minted no longer name anything.
  var scanPieceId: String?

  init(jpeg: Data, scanPieceId: String? = nil) {
    self.id = UUID()
    self.uploadJPEG = jpeg
    self.displayImage = jpeg
    self.trimmedDisplayImage = ImageUtilities.alphaTrimmedPNG(from: jpeg)
    self.scanPieceId = scanPieceId
  }

  /// Rehydrates an entry loaded from disk (see PuzzleStore.loadSession).
  init(
    id: UUID, uploadJPEG: Data, displayImage: Data, status: Status, result: PieceResponse?,
    scanPieceId: String? = nil
  ) {
    self.id = id
    self.uploadJPEG = uploadJPEG
    self.displayImage = displayImage
    self.trimmedDisplayImage = ImageUtilities.alphaTrimmedPNG(from: displayImage)
    self.status = status
    self.result = result
    self.scanPieceId = scanPieceId
  }
}
