import Foundation
import GoogleSignIn
import UIKit

/// Owns the Google Sign-In flow and the exchange of Google ID tokens for
/// backend JWTs (POST /api/v1/auth/google). The backend JWT is never
/// persisted; GoogleSignIn's Keychain session is the durable credential.
@MainActor
final class AuthService {
  private let authStore: AuthStore
  private let apiClient: APIClient

  init(authStore: AuthStore, apiClient: APIClient) {
    self.authStore = authStore
    self.apiClient = apiClient
    apiClient.reauthenticator = { [weak self] in
      await self?.refreshBackendToken() ?? false
    }
  }

  func configure() {
    guard Config.isGoogleSignInConfigured else { return }
    GIDSignIn.sharedInstance.configuration = GIDConfiguration(
      clientID: Config.googleIOSClientID,
      serverClientID: Config.googleServerClientID
    )
  }

  func handle(url: URL) {
    _ = GIDSignIn.sharedInstance.handle(url)
  }

  /// Called at launch: restores a previous Google session silently, if any.
  func restoreSession() async {
    #if DEBUG
      if applyDebugToken() { return }
    #endif
    guard Config.isGoogleSignInConfigured, GIDSignIn.sharedInstance.hasPreviousSignIn() else {
      return
    }
    _ = await refreshBackendToken()
  }

  func signIn() async {
    guard Config.isGoogleSignInConfigured else {
      authStore.errorMessage =
        "Google Sign-In is not configured. Fill in ios/Config/Secrets.xcconfig."
      return
    }
    guard let presenter = Self.topViewController() else {
      authStore.errorMessage =
        "Could not find a screen to present Google Sign-In from. Please try again."
      return
    }
    authStore.isSigningIn = true
    authStore.errorMessage = nil
    defer { authStore.isSigningIn = false }
    do {
      let result = try await GIDSignIn.sharedInstance.signIn(withPresenting: presenter)
      guard let idToken = result.user.idToken?.tokenString else {
        throw APIError(message: "Google did not return an ID token.", status: nil)
      }
      try await exchange(idToken: idToken)
    } catch let error as GIDSignInError where error.code == .canceled {
      // User dismissed the sheet.
    } catch {
      authStore.errorMessage = error.localizedDescription
    }
  }

  func signOut() {
    GIDSignIn.sharedInstance.signOut()
    authStore.user = nil
    authStore.backendToken = nil
  }

  private func exchange(idToken: String) async throws {
    let response = try await apiClient.signInWithGoogle(idToken: idToken)
    authStore.backendToken = response.accessToken
    authStore.user = response.user
  }

  /// Silent re-auth used on launch and after 401s.
  private func refreshBackendToken() async -> Bool {
    guard Config.isGoogleSignInConfigured else { return false }
    do {
      let user = try await restorePreviousSignIn()
      let refreshed = try await refreshTokens(for: user)
      guard let idToken = refreshed.idToken?.tokenString else { return false }
      try await exchange(idToken: idToken)
      return true
    } catch {
      return false
    }
  }

  private func restorePreviousSignIn() async throws -> GIDGoogleUser {
    try await withCheckedThrowingContinuation { continuation in
      GIDSignIn.sharedInstance.restorePreviousSignIn { user, error in
        if let user {
          continuation.resume(returning: user)
        } else {
          continuation.resume(
            throwing: error ?? APIError(message: "No previous Google sign-in.", status: nil))
        }
      }
    }
  }

  private func refreshTokens(for user: GIDGoogleUser) async throws -> GIDGoogleUser {
    try await withCheckedThrowingContinuation { continuation in
      user.refreshTokensIfNeeded { user, error in
        if let user {
          continuation.resume(returning: user)
        } else {
          continuation.resume(
            throwing: error ?? APIError(message: "Could not refresh Google tokens.", status: nil))
        }
      }
    }
  }

  private static func topViewController() -> UIViewController? {
    let window = UIApplication.shared.connectedScenes
      .compactMap { $0 as? UIWindowScene }
      .flatMap(\.windows)
      .first { $0.isKeyWindow }
    var top = window?.rootViewController
    while let presented = top?.presentedViewController {
      top = presented
    }
    return top
  }

  #if DEBUG
    /// Agent/CI escape hatch: launch with SIMCTL_CHILD_PUSSEL_DEBUG_TOKEN
    /// set to a JWT from backend/scripts/generate_test_token.py to skip
    /// Google Sign-In entirely (Simulator testing without OAuth setup).
    private func applyDebugToken() -> Bool {
      guard let token = ProcessInfo.processInfo.environment["PUSSEL_DEBUG_TOKEN"], !token.isEmpty
      else {
        return false
      }
      authStore.backendToken = token
      authStore.user = UserDTO(
        id: "debug", email: "debug@example.com", name: "Debug User", picture: nil, createdAt: nil)
      return true
    }
  #endif
}
