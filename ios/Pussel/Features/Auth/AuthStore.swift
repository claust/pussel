import Foundation
import Observation

@Observable
@MainActor
final class AuthStore {
  var user: UserDTO?
  /// The signed-in user's Google avatar, resolved by the Sign-In SDK rather
  /// than from `user.picture` (ID tokens carry no `picture` claim).
  var avatarURL: URL?
  /// Backend JWT, in memory only — reminted from the persisted Google
  /// session on launch and after 401s (60 min expiry, no refresh endpoint).
  var backendToken: String?
  var isSigningIn = false
  var errorMessage: String?

  var isAuthenticated: Bool { backendToken != nil }

  /// Drops every part of the signed-in session at once. Both the explicit
  /// sign-out and the 401 session-drop go through here so neither can miss a
  /// field as the session gains state.
  func clearSession() {
    user = nil
    avatarURL = nil
    backendToken = nil
  }
}
