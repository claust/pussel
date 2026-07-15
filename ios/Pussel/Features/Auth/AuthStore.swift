import Foundation
import Observation

@Observable
@MainActor
final class AuthStore {
  var user: UserDTO?
  /// Backend JWT, in memory only — reminted from the persisted Google
  /// session on launch and after 401s (60 min expiry, no refresh endpoint).
  var backendToken: String?
  var isSigningIn = false
  var errorMessage: String?

  var isAuthenticated: Bool { backendToken != nil }
}
