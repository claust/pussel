import Foundation

struct APIError: Error, LocalizedError, Equatable {
  let message: String
  let status: Int?

  var errorDescription: String? { message }

  /// Maps a non-2xx backend response to an error, mirroring frontend/src/lib/api.ts.
  static func from(data: Data, status: Int) -> APIError {
    if status == 401 {
      return APIError(message: "Authentication required. Please sign in.", status: status)
    }
    struct Body: Decodable {
      let detail: String?
    }
    let detail = (try? JSONDecoder().decode(Body.self, from: data))?.detail
    return APIError(message: detail ?? "Request failed with status \(status)", status: status)
  }
}
