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

  func testDecodeBareBase64DataURL() {
    XCTAssertEqual(ImageUtilities.decodeDataURL("aGVsbG8="), Data("hello".utf8))
    XCTAssertNil(ImageUtilities.decodeDataURL("data:image/png;base64,%%%invalid%%%"))
  }
}
