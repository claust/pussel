import XCTest

@testable import Pussel

/// Decoding tests against the backend's snake_case JSON shapes
/// (backend/app/models/puzzle_model.py and app/auth models).
final class ModelDecodingTests: XCTestCase {
  private let decoder: JSONDecoder = {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
  }()

  func testDecodeAuthResponse() throws {
    let json = """
      {
        "access_token": "jwt-token-value",
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {
          "id": "12345",
          "email": "claus@example.com",
          "name": "Claus",
          "picture": null,
          "created_at": "2026-07-15T10:00:00.000000"
        }
      }
      """
    let response = try decoder.decode(AuthResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.accessToken, "jwt-token-value")
    XCTAssertEqual(response.expiresIn, 3600)
    XCTAssertEqual(response.user.email, "claus@example.com")
    XCTAssertNil(response.user.picture)
  }

  func testDecodeDetectFrameResponse() throws {
    let json = """
      {
        "trimmed_image": "data:image/jpeg;base64,aGVsbG8=",
        "corners": {
          "top_left": {"x": 0.1, "y": 0.1},
          "top_right": {"x": 0.9, "y": 0.1},
          "bottom_right": {"x": 0.9, "y": 0.9},
          "bottom_left": {"x": 0.1, "y": 0.9}
        },
        "confidence": 0.87
      }
      """
    let response = try decoder.decode(DetectFrameResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.confidence, 0.87)
    XCTAssertEqual(response.corners.topLeft, NormalizedPoint(x: 0.1, y: 0.1))
    XCTAssertEqual(response.corners.bottomRight, NormalizedPoint(x: 0.9, y: 0.9))
    XCTAssertEqual(ImageUtilities.decodeDataURL(response.trimmedImage), Data("hello".utf8))
  }

  func testDecodePuzzleUploadResponse() throws {
    let json = """
      {"puzzle_id": "3f2b9c", "image_url": null}
      """
    let response = try decoder.decode(PuzzleUploadResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.puzzleId, "3f2b9c")
    XCTAssertNil(response.imageUrl)
    XCTAssertNil(response.pieceCount)
    XCTAssertNil(response.rows)
    XCTAssertNil(response.cols)
  }

  func testDecodePuzzleUploadResponseWithGrid() throws {
    let json = """
      {"puzzle_id": "3f2b9c", "image_url": null, "piece_count": 500, "rows": 22, "cols": 22}
      """
    let response = try decoder.decode(PuzzleUploadResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.pieceCount, 500)
    XCTAssertEqual(response.rows, 22)
    XCTAssertEqual(response.cols, 22)
  }

  func testDecodePieceResponse() throws {
    let json = """
      {
        "position": {"x": 0.25, "y": 0.75},
        "position_confidence": 0.91,
        "rotation": 270,
        "rotation_confidence": 0.66,
        "cleaned_image": "data:image/png;base64,cGll"
      }
      """
    let response = try decoder.decode(PieceResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.position, NormalizedPoint(x: 0.25, y: 0.75))
    XCTAssertEqual(response.rotation, 270)
    XCTAssertEqual(response.positionConfidence, 0.91)
    XCTAssertEqual(ImageUtilities.decodeDataURL(response.cleanedImage!), Data("pie".utf8))
    // Grid snap fields absent entirely (old backend): nil, and display falls
    // back to the raw prediction.
    XCTAssertNil(response.gridRow)
    XCTAssertNil(response.gridCol)
    XCTAssertNil(response.snappedPosition)
    XCTAssertEqual(response.displayPosition, response.position)
  }

  func testDecodePieceResponseWithGridSnap() throws {
    let json = """
      {
        "position": {"x": 1.02, "y": -0.03},
        "position_confidence": 0.91,
        "rotation": 0,
        "rotation_confidence": 0.66,
        "cleaned_image": null,
        "grid_row": 0,
        "grid_col": 5,
        "snapped_position": {"x": 0.9166, "y": 0.125}
      }
      """
    let response = try decoder.decode(PieceResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.gridRow, 0)
    XCTAssertEqual(response.gridCol, 5)
    XCTAssertEqual(response.snappedPosition, NormalizedPoint(x: 0.9166, y: 0.125))
    // Display uses the snapped center; the raw out-of-bounds prediction is kept.
    XCTAssertEqual(response.displayPosition, NormalizedPoint(x: 0.9166, y: 0.125))
    XCTAssertEqual(response.position, NormalizedPoint(x: 1.02, y: -0.03))
  }

  func testDecodePieceResponseWithNullGridSnap() throws {
    // The backend's shape when the puzzle's grid is unknown: keys present, null.
    let json = """
      {
        "position": {"x": 0.25, "y": 0.75},
        "position_confidence": 0.91,
        "rotation": 0,
        "rotation_confidence": 0.66,
        "cleaned_image": null,
        "grid_row": null,
        "grid_col": null,
        "snapped_position": null
      }
      """
    let response = try decoder.decode(PieceResponse.self, from: Data(json.utf8))
    XCTAssertNil(response.snappedPosition)
    XCTAssertEqual(response.displayPosition, NormalizedPoint(x: 0.25, y: 0.75))
  }

  func testDecodePieceResponseWithoutCleanedImage() throws {
    let json = """
      {
        "position": {"x": 0.5, "y": 0.5},
        "position_confidence": 0.0,
        "rotation": 0,
        "rotation_confidence": 0.0,
        "cleaned_image": null
      }
      """
    let response = try decoder.decode(PieceResponse.self, from: Data(json.utf8))
    XCTAssertNil(response.cleanedImage)
  }

  func testDecodePieceResponseWithPieceSpan() throws {
    let json = """
      {
        "position": {"x": 0.25, "y": 0.75},
        "position_confidence": 0.91,
        "rotation": 270,
        "rotation_confidence": 0.66,
        "cleaned_image": null,
        "piece_span": {"width": 0.34, "height": 0.25}
      }
      """
    let response = try decoder.decode(PieceResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.pieceSpan, PieceSpan(width: 0.34, height: 0.25))
  }

  func testDecodePieceResponseWithNullPieceSpan() throws {
    let json = """
      {
        "position": {"x": 0.25, "y": 0.75},
        "position_confidence": 0.91,
        "rotation": 270,
        "rotation_confidence": 0.66,
        "cleaned_image": null,
        "piece_span": null
      }
      """
    let response = try decoder.decode(PieceResponse.self, from: Data(json.utf8))
    XCTAssertNil(response.pieceSpan)
  }

  func testDecodePiecePreviewResponseFound() throws {
    let json = """
      {
        "found": true,
        "polygon": [
          {"x": 0.1, "y": 0.2},
          {"x": 0.8, "y": 0.2},
          {"x": 0.8, "y": 0.9},
          {"x": 0.1, "y": 0.9}
        ],
        "bbox": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.7},
        "confidence": 0.82,
        "lockable": true,
        "corner_disagreement": false
      }
      """
    let response = try decoder.decode(PiecePreviewResponse.self, from: Data(json.utf8))
    XCTAssertTrue(response.found)
    XCTAssertEqual(response.polygon.count, 4)
    XCTAssertEqual(response.polygon.first, NormalizedPoint(x: 0.1, y: 0.2))
    XCTAssertEqual(
      response.bbox, NormalizedBoundingBox(x: 0.1, y: 0.2, width: 0.7, height: 0.7))
    XCTAssertEqual(response.confidence, 0.82)
    XCTAssertEqual(response.lockable, true)
    XCTAssertEqual(response.cornerDisagreement, false)
  }

  func testDecodePiecePreviewResponseNotFound() throws {
    // The backend's actual shape when found=false: an empty polygon, no
    // bbox, confidence defaulted to 0.0.
    let json = """
      {"found": false, "polygon": [], "bbox": null, "confidence": 0.0}
      """
    let response = try decoder.decode(PiecePreviewResponse.self, from: Data(json.utf8))
    XCTAssertFalse(response.found)
    XCTAssertTrue(response.polygon.isEmpty)
    XCTAssertNil(response.bbox)
    XCTAssertEqual(response.confidence, 0.0)
    XCTAssertNil(response.lockable)
    XCTAssertNil(response.cornerDisagreement)
  }

  func testDecodePiecePreviewResponseWithoutQualityFields() throws {
    // include_quality=false (or omitted): lockable/corner_disagreement are
    // present but null, matching the backend's default response shape.
    let json = """
      {
        "found": true,
        "polygon": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
        "confidence": 0.5,
        "lockable": null,
        "corner_disagreement": null
      }
      """
    let response = try decoder.decode(PiecePreviewResponse.self, from: Data(json.utf8))
    XCTAssertTrue(response.found)
    XCTAssertNil(response.lockable)
    XCTAssertNil(response.cornerDisagreement)
  }

  func testDecodeBareBase64DataURL() {
    XCTAssertEqual(ImageUtilities.decodeDataURL("aGVsbG8="), Data("hello".utf8))
    XCTAssertNil(ImageUtilities.decodeDataURL("data:image/png;base64,%%%invalid%%%"))
  }

  // MARK: - PieceGeometryUploadResponse

  func testDecodePieceGeometryUploadResponseMatchedStatus() throws {
    // Realistic backend payload for a matched piece: includes fields the app
    // ignores (dominant_dev, polyline, corners, corner_confidences, contour,
    // n_large_components, border_touching, area_ratio, solidity) to confirm
    // the decoder tolerates extra keys gracefully.
    let json = """
      {
        "piece_id": "piece-abc",
        "status": "matched",
        "match_piece_id": "piece-xyz",
        "z_score": 1.42,
        "lockable": true,
        "quality": {
          "is_clean": true,
          "corner_disagreement": false,
          "n_large_components": 1,
          "border_touching": false,
          "area_ratio": 0.87,
          "solidity": 0.94
        },
        "record": {
          "corners": [
            {"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1},
            {"x": 0.9, "y": 0.9}, {"x": 0.1, "y": 0.9}
          ],
          "corner_confidences": [0.9, 0.85, 0.92, 0.88],
          "edges": [
            {"type": "tab",   "dominant_dev": 0.12, "polyline": []},
            {"type": "blank", "dominant_dev": -0.11, "polyline": []},
            {"type": "flat",  "dominant_dev": 0.0, "polyline": []},
            {"type": "tab",   "dominant_dev": 0.09, "polyline": []}
          ],
          "contour": null
        }
      }
      """
    let response = try decoder.decode(PieceGeometryUploadResponse.self, from: Data(json.utf8))

    XCTAssertEqual(response.pieceId, "piece-abc")
    XCTAssertEqual(response.status, .matched)
    XCTAssertEqual(response.matchPieceId, "piece-xyz")
    XCTAssertEqual(response.zScore, 1.42)
    XCTAssertTrue(response.lockable)
    XCTAssertTrue(response.quality.isClean)
    XCTAssertEqual(response.quality.cornerDisagreement, false)
    XCTAssertEqual(
      response.edgeTypes, [.tab, .blank, .flat, .tab],
      "edgeTypes convenience var must mirror record.edges[].type in order")
  }

  func testDecodePieceGeometryUploadResponseNewPieceNilOptionals() throws {
    // A brand-new piece: no match, no z_score, no piece_id yet for uncertain.
    let json = """
      {
        "piece_id": "piece-001",
        "status": "new",
        "match_piece_id": null,
        "z_score": null,
        "lockable": true,
        "quality": {
          "is_clean": true,
          "corner_disagreement": null,
          "n_large_components": 1,
          "border_touching": false,
          "area_ratio": 0.91,
          "solidity": 0.97
        },
        "record": {
          "corners": [],
          "corner_confidences": [],
          "edges": [
            {"type": "flat",  "dominant_dev": 0.0,  "polyline": []},
            {"type": "tab",   "dominant_dev": 0.15, "polyline": []},
            {"type": "flat",  "dominant_dev": 0.0,  "polyline": []},
            {"type": "blank", "dominant_dev": -0.08, "polyline": []}
          ],
          "contour": null
        }
      }
      """
    let response = try decoder.decode(PieceGeometryUploadResponse.self, from: Data(json.utf8))

    XCTAssertEqual(response.pieceId, "piece-001")
    XCTAssertEqual(response.status, .new)
    XCTAssertNil(response.matchPieceId)
    XCTAssertNil(response.zScore)
    XCTAssertNil(response.quality.cornerDisagreement)
    XCTAssertEqual(response.edgeTypes, [.flat, .tab, .flat, .blank])
  }

  func testDecodePieceGeometryUploadResponseUncertainNilPieceId() throws {
    // Uncertain with on_uncertain=report (default): piece_id stays nil.
    let json = """
      {
        "piece_id": null,
        "status": "uncertain",
        "match_piece_id": "piece-xyz",
        "z_score": 2.71,
        "lockable": false,
        "quality": {
          "is_clean": false,
          "corner_disagreement": true,
          "n_large_components": 2,
          "border_touching": true,
          "area_ratio": 0.55,
          "solidity": 0.72
        },
        "record": {
          "corners": [],
          "corner_confidences": [],
          "edges": [],
          "contour": null
        }
      }
      """
    let response = try decoder.decode(PieceGeometryUploadResponse.self, from: Data(json.utf8))

    XCTAssertNil(response.pieceId)
    XCTAssertEqual(response.status, .uncertain)
    XCTAssertFalse(response.lockable)
    XCTAssertFalse(response.quality.isClean)
    XCTAssertEqual(response.quality.cornerDisagreement, true)
    XCTAssertTrue(response.edgeTypes.isEmpty)
  }

  // MARK: - PieceGeometryListResponse

  func testDecodePieceGeometryListResponse() throws {
    let json = """
      {
        "puzzle_id": "puzzle-99",
        "pieces": [
          {
            "piece_id": "piece-001",
            "edge_types": ["tab", "blank", "flat", "tab"],
            "is_clean": true,
            "corner_disagreement": false
          },
          {
            "piece_id": "piece-002",
            "edge_types": ["flat", "flat", "tab", "blank"],
            "is_clean": false,
            "corner_disagreement": true
          }
        ]
      }
      """
    let response = try decoder.decode(PieceGeometryListResponse.self, from: Data(json.utf8))

    XCTAssertEqual(response.puzzleId, "puzzle-99")
    XCTAssertEqual(response.pieces.count, 2)

    let first = response.pieces[0]
    XCTAssertEqual(first.id, "piece-001")
    XCTAssertEqual(first.edgeTypes, [.tab, .blank, .flat, .tab])
    XCTAssertTrue(first.isClean)
    XCTAssertFalse(first.cornerDisagreement)

    let second = response.pieces[1]
    XCTAssertEqual(second.id, "piece-002")
    XCTAssertEqual(second.edgeTypes, [.flat, .flat, .tab, .blank])
    XCTAssertFalse(second.isClean)
    XCTAssertTrue(second.cornerDisagreement)
  }

  func testDecodePieceGeometryListResponseEmptyPieces() throws {
    let json = """
      {"puzzle_id": "puzzle-empty", "pieces": []}
      """
    let response = try decoder.decode(PieceGeometryListResponse.self, from: Data(json.utf8))
    XCTAssertEqual(response.puzzleId, "puzzle-empty")
    XCTAssertTrue(response.pieces.isEmpty)
  }

  // MARK: - GeometryEdgeType.glyph

  func testGeometryEdgeTypeGlyphs() {
    XCTAssertEqual(GeometryEdgeType.tab.glyph, "T")
    XCTAssertEqual(GeometryEdgeType.blank.glyph, "B")
    XCTAssertEqual(GeometryEdgeType.flat.glyph, "F")
  }
}
