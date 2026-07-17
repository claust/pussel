import Foundation

// Models for POST .../piece/geometry and GET .../piece/geometry, mirroring
// backend/app/models/piece_geometry_model.py. The shared decoder uses
// .convertFromSnakeCase, so snake_case JSON keys map automatically to
// camelCase Swift properties. Extra JSON keys (dominant_dev, polyline,
// corners, corner_confidences, contour, n_large_components, border_touching,
// area_ratio, solidity, …) are silently ignored — we only decode what the
// iOS scan-lock UI needs.

// MARK: - Edge type

/// Puzzle-edge profile: a tab protrudes into the neighbour, a blank receives
/// one, and a flat side sits on the puzzle border.
enum GeometryEdgeType: String, Codable, Equatable {
  case tab
  case blank
  case flat

  /// Single-character glyph used in the gallery's edge-type summary line
  /// (e.g. "T·B·F·T"). Compact and language-independent.
  var glyph: String {
    switch self {
    case .tab: return "T"
    case .blank: return "B"
    case .flat: return "F"
    }
  }
}

// MARK: - Quality

/// Quality verdict for a piece contour. Only the two boolean flags matter to
/// the scan-lock flow; the numeric metrics (area_ratio, solidity, …) are left
/// on the backend and not decoded here.
struct GeometryQuality: Codable, Equatable {
  /// The contour passed all quality gates (component count, area, solidity).
  let isClean: Bool
  /// Whether the polydp and curvature corner detectors disagreed beyond the
  /// allowed tolerance; nil when corner detection did not run (the backend
  /// only sets this when a clean contour was found).
  let cornerDisagreement: Bool?
}

// MARK: - Edge summary (list-only; no polyline)

/// Condensed form of one classified edge, sufficient for the gallery badge.
/// `dominant_dev` and `polyline` are intentionally omitted — the app only
/// needs the type label, not the shape detail.
struct GeometryEdgeSummary: Codable, Equatable {
  let type: GeometryEdgeType
}

// MARK: - Record summary (upload response)

/// The subset of `PieceGeometryRecordResponse` the app consumes: the four
/// classified edges in contour-traversal order. Corners, polylines, and the
/// optional full contour are not decoded.
struct PieceGeometryRecordSummary: Codable, Equatable {
  let edges: [GeometryEdgeSummary]
}

// MARK: - Upload response

/// Match status returned by the dedupe store for each uploaded geometry.
enum PieceGeometryStatus: String, Codable, Equatable {
  /// First time this shape has been seen for the puzzle.
  case new
  /// Shape matched an already-enrolled piece within the lock threshold.
  case matched
  /// Shape is within the scan-lock z-score band but below the hard match
  /// threshold — may be the same piece in worse lighting.
  case uncertain
}

/// Response of POST .../piece/geometry.
struct PieceGeometryUploadResponse: Codable, Equatable {
  /// Stable id for this piece (matched or newly enrolled), or nil when the
  /// status is `uncertain` and the backend chose not to enroll.
  let pieceId: String?
  let status: PieceGeometryStatus
  /// Id of the closest already-enrolled piece when a comparison was made
  /// (present for `matched` and `uncertain`; nil for `new`).
  let matchPieceId: String?
  /// Combined shape+colour z-score of the closest match; nil when no
  /// comparison was made (first piece ever, status `new`).
  let zScore: Double?
  /// True when the contour is clean, corner detectors agree, and all four
  /// edges were classified — the minimum bar for scan-lock auto-capture.
  let lockable: Bool
  let quality: GeometryQuality
  let record: PieceGeometryRecordSummary

  /// Flat array of edge types in contour-traversal order, for quick display.
  var edgeTypes: [GeometryEdgeType] { record.edges.map(\.type) }
}

// MARK: - List response

/// One enrolled piece's summary, returned by GET .../piece/geometry.
struct PieceGeometrySummary: Codable, Equatable, Identifiable {
  let pieceId: String
  let edgeTypes: [GeometryEdgeType]
  let isClean: Bool
  /// Always present in list responses (backend always stores a resolved
  /// boolean after enrollment, unlike the nullable upload response field).
  let cornerDisagreement: Bool

  /// `Identifiable` conformance — pieceId is the durable backend key.
  var id: String { pieceId }
}

/// Response of GET .../piece/geometry for a puzzle.
struct PieceGeometryListResponse: Codable, Equatable {
  let puzzleId: String
  let pieces: [PieceGeometrySummary]
}
