import XCTest

@testable import Pussel

/// URLProtocol stub so APIClient tests never hit the network. URLProtocol
/// callbacks run on URLSession's internal threads, so the shared state is
/// guarded by a lock.
final class StubURLProtocol: URLProtocol {
  private static let lock = NSLock()
  nonisolated(unsafe) private static var _handler: ((URLRequest) -> (Int, Data))?
  nonisolated(unsafe) private static var _receivedRequests: [URLRequest] = []

  static var handler: ((URLRequest) -> (Int, Data))? {
    get { lock.withLock { _handler } }
    set { lock.withLock { _handler = newValue } }
  }

  static var receivedRequests: [URLRequest] {
    get { lock.withLock { _receivedRequests } }
    set { lock.withLock { _receivedRequests = newValue } }
  }

  // These override URLProtocol's class members, so `static` isn't an option
  // (it can't be combined with `override`); silence static_over_final_class.
  // swiftlint:disable static_over_final_class
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  // swiftlint:enable static_over_final_class

  override func startLoading() {
    let handler = Self.lock.withLock {
      Self._receivedRequests.append(request)
      return Self._handler
    }
    guard let handler else {
      client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
      return
    }
    let (status, data) = handler(request)
    let response = HTTPURLResponse(
      url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: data)
    client?.urlProtocolDidFinishLoading(self)
  }

  override func stopLoading() {}
}

@MainActor
final class APIClientTests: XCTestCase {
  private var authStore: AuthStore!
  private var client: APIClient!

  override func setUp() {
    super.setUp()
    StubURLProtocol.handler = nil
    StubURLProtocol.receivedRequests = []
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubURLProtocol.self]
    authStore = AuthStore()
    client = APIClient(
      baseURL: URL(string: "http://stub.local")!,
      session: URLSession(configuration: configuration),
      authStore: authStore
    )
  }

  func testMultipartRequestBuilding() throws {
    let request = client.makeMultipartRequest(
      path: "api/v1/puzzle/abc/piece",
      queryItems: [URLQueryItem(name: "remove_bg", value: "true")],
      jpegData: Data("JPEG".utf8),
      filename: "piece.jpg"
    )
    XCTAssertEqual(request.httpMethod, "POST")
    XCTAssertEqual(
      request.url?.absoluteString, "http://stub.local/api/v1/puzzle/abc/piece?remove_bg=true")
    let contentType = request.value(forHTTPHeaderField: "Content-Type") ?? ""
    XCTAssertTrue(contentType.hasPrefix("multipart/form-data; boundary="))
    let body = try XCTUnwrap(String(bytes: request.httpBody ?? Data(), encoding: .utf8))
    XCTAssertTrue(body.contains("name=\"file\"; filename=\"piece.jpg\""))
    XCTAssertTrue(body.contains("Content-Type: image/jpeg"))
  }

  func testMultipartRequestBuildingIncludesTextFields() throws {
    let request = client.makeMultipartRequest(
      path: "api/v1/puzzle/upload",
      jpegData: Data("JPEG".utf8),
      filename: "puzzle.jpg",
      fields: ["piece_count": "500"]
    )
    let body = try XCTUnwrap(String(bytes: request.httpBody ?? Data(), encoding: .utf8))
    XCTAssertTrue(
      body.contains("Content-Disposition: form-data; name=\"piece_count\"\r\n\r\n500\r\n"))
    XCTAssertTrue(body.contains("name=\"file\"; filename=\"puzzle.jpg\""))
  }

  func testAuthorizationHeaderAttached() async throws {
    authStore.backendToken = "token123"
    StubURLProtocol.handler = { _ in
      (200, Data(#"{"puzzle_id": "p1", "image_url": null}"#.utf8))
    }
    _ = try await client.uploadPuzzle(jpegData: Data("JPEG".utf8), pieceCount: 48)
    let auth = StubURLProtocol.receivedRequests.first?.value(forHTTPHeaderField: "Authorization")
    XCTAssertEqual(auth, "Bearer token123")
  }

  func testErrorMappingUsesDetail() async {
    StubURLProtocol.handler = { _ in
      (404, Data(#"{"detail": "Puzzle not found"}"#.utf8))
    }
    do {
      _ = try await client.processPiece(puzzleId: "dead", jpegData: Data("J".utf8))
      XCTFail("Expected APIError")
    } catch let error as APIError {
      XCTAssertEqual(error.status, 404)
      XCTAssertEqual(error.message, "Puzzle not found")
    } catch {
      XCTFail("Unexpected error type: \(error)")
    }
  }

  func test401TriggersReauthAndRetriesOnce() async throws {
    authStore.backendToken = "stale"
    StubURLProtocol.handler = { request in
      if request.value(forHTTPHeaderField: "Authorization") == "Bearer fresh" {
        return (200, Data(#"{"puzzle_id": "p2", "image_url": null}"#.utf8))
      }
      return (401, Data(#"{"detail": "Invalid or expired token"}"#.utf8))
    }
    client.reauthenticator = { [authStore] in
      authStore?.backendToken = "fresh"
      return true
    }
    let response = try await client.uploadPuzzle(jpegData: Data("J".utf8), pieceCount: 48)
    XCTAssertEqual(response.puzzleId, "p2")
    // The lock-guarded request log stands in for a hand-rolled counter,
    // which would race with URLSession's internal threads.
    XCTAssertEqual(StubURLProtocol.receivedRequests.count, 2)
  }

  func test401WithoutReauthSurfacesError() async {
    authStore.backendToken = "stale"
    authStore.user = UserDTO(
      id: "u1", email: "u@example.com", name: "U", picture: nil, createdAt: nil)
    authStore.avatarURL = URL(string: "https://lh3.googleusercontent.com/a/avatar=s96")
    StubURLProtocol.handler = { _ in
      (401, Data(#"{"detail": "Invalid or expired token"}"#.utf8))
    }
    do {
      _ = try await client.uploadPuzzle(jpegData: Data("J".utf8), pieceCount: 48)
      XCTFail("Expected APIError")
    } catch let error as APIError {
      XCTAssertEqual(error.status, 401)
      XCTAssertEqual(error.message, "Authentication required. Please sign in.")
      // The dead session must be dropped in full so the UI returns to
      // sign-in with nothing of the old session left behind.
      XCTAssertNil(authStore.backendToken)
      XCTAssertNil(authStore.user)
      XCTAssertNil(authStore.avatarURL)
    } catch {
      XCTFail("Unexpected error type: \(error)")
    }
  }
}
