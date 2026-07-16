import Foundation

/// Async client for the Pussel FastAPI backend, mirroring frontend/src/lib/api.ts.
@MainActor
final class APIClient {
  private let baseURL: URL
  private let session: URLSession
  private let authStore: AuthStore
  private let decoder: JSONDecoder

  /// Set by AuthService: silently refresh the backend JWT after a 401.
  /// Returns true when a fresh token was stored and the request can retry.
  var reauthenticator: (() async -> Bool)?

  init(baseURL: URL = Config.apiBaseURL, session: URLSession = .shared, authStore: AuthStore) {
    self.baseURL = baseURL
    self.session = session
    self.authStore = authStore
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    self.decoder = decoder
  }

  // MARK: - Endpoints

  func checkHealth() async -> Bool {
    let request = URLRequest(url: baseURL.appending(path: "health"))
    guard let (_, response) = try? await session.data(for: request),
      let http = response as? HTTPURLResponse
    else { return false }
    return http.statusCode == 200
  }

  func signInWithGoogle(idToken: String) async throws -> AuthResponse {
    var request = URLRequest(url: baseURL.appending(path: "api/v1/auth/google"))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONEncoder().encode(["id_token": idToken])
    let data = try await send(request, authenticated: false)
    return try decoder.decode(AuthResponse.self, from: data)
  }

  func detectFrame(jpegData: Data) async throws -> DetectFrameResponse {
    let request = makeMultipartRequest(
      path: "api/v1/puzzle/detect-frame", jpegData: jpegData, filename: "puzzle.jpg")
    return try await sendDecoding(request)
  }

  func uploadPuzzle(jpegData: Data, pieceCount: Int) async throws -> PuzzleUploadResponse {
    let request = makeMultipartRequest(
      path: "api/v1/puzzle/upload",
      jpegData: jpegData,
      filename: "puzzle.jpg",
      fields: ["piece_count": String(pieceCount)]
    )
    return try await sendDecoding(request)
  }

  func processPiece(puzzleId: String, jpegData: Data, removeBg: Bool = true) async throws
    -> PieceResponse
  {
    let request = makeMultipartRequest(
      path: "api/v1/puzzle/\(puzzleId)/piece",
      queryItems: [URLQueryItem(name: "remove_bg", value: removeBg ? "true" : "false")],
      jpegData: jpegData,
      filename: "piece.jpg"
    )
    return try await sendDecoding(request)
  }

  // MARK: - Request building & sending

  func makeMultipartRequest(
    path: String, queryItems: [URLQueryItem] = [], jpegData: Data, filename: String,
    fields: [String: String] = [:]
  )
    -> URLRequest
  {
    var url = baseURL.appending(path: path)
    if !queryItems.isEmpty {
      url.append(queryItems: queryItems)
    }
    var form = MultipartFormData()
    for (name, value) in fields {
      form.appendField(name: name, value: value)
    }
    form.appendFile(name: "file", filename: filename, mimeType: "image/jpeg", data: jpegData)
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
    request.httpBody = form.encoded()
    return request
  }

  private func sendDecoding<T: Decodable>(_ request: URLRequest) async throws -> T {
    let data = try await send(request, authenticated: true)
    return try decoder.decode(T.self, from: data)
  }

  private func send(_ request: URLRequest, authenticated: Bool, allowRetry: Bool = true)
    async throws -> Data
  {
    var request = request
    if authenticated, let token = authStore.backendToken {
      request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
    let (data, response) = try await session.data(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw APIError(message: "Invalid response from server.", status: nil)
    }
    if http.statusCode == 401, authenticated, allowRetry, await reauthenticator?() == true {
      return try await send(request, authenticated: true, allowRetry: false)
    }
    guard (200..<300).contains(http.statusCode) else {
      if http.statusCode == 401, authenticated {
        // Silent re-auth failed or the retry 401ed again — drop the
        // session so RootView falls back to SignInView instead of
        // leaving an "authenticated" UI whose every call fails.
        authStore.backendToken = nil
        authStore.user = nil
      }
      throw APIError.from(data: data, status: http.statusCode)
    }
    return data
  }
}
