import SwiftUI

@Observable
@MainActor
final class AppModel {
  let auth: AuthStore
  let flow: AppFlowStore
  let store: PuzzleStore
  let backend: BackendSelection
  @ObservationIgnored let api: APIClient
  @ObservationIgnored let authService: AuthService

  init() {
    let auth = AuthStore()
    let backend = BackendSelection()
    let api = APIClient(baseURL: backend.baseURL, authStore: auth)
    self.auth = auth
    self.flow = AppFlowStore()
    self.store = PuzzleStore()
    self.backend = backend
    self.api = api
    self.authService = AuthService(authStore: auth, apiClient: api)
    authService.configure()
  }

  /// Repoints the app at the other backend (Debug builds only — see
  /// `BackendSelection`). The JWT and every `puzzle_id` in flight were issued
  /// by the backend being left behind, so the session and the wizard are
  /// dropped and a fresh token minted from the new one; the persisted Google
  /// sign-in makes that silent.
  func useLocalBackend(_ useLocal: Bool) {
    guard backend.isSwitchable, backend.usesLocalBackend != useLocal else { return }
    backend.setUsesLocalBackend(useLocal)
    api.baseURL = backend.baseURL
    auth.clearSession()
    flow.reset()
    Task { await authService.restoreSession() }
  }
}

@main
struct PusselApp: App {
  @State private var model = AppModel()

  var body: some Scene {
    WindowGroup {
      RootView()
        .environment(model)
        .onOpenURL { model.authService.handle(url: $0) }
        #if DEBUG
          .task { DebugDriver.start(model) }
        #endif
    }
  }
}
