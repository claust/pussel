import Foundation

struct UserDTO: Codable, Equatable {
    let id: String
    let email: String
    let name: String
    let picture: String?
    let createdAt: String?
}

/// Response of POST /api/v1/auth/google.
struct AuthResponse: Codable, Equatable {
    let accessToken: String
    let tokenType: String
    let expiresIn: Int
    let user: UserDTO
}
