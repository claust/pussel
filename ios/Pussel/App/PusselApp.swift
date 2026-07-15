import SwiftUI

@Observable
@MainActor
final class AppModel {
    let auth: AuthStore
    let flow: AppFlowStore
    let store: PuzzleStore
    @ObservationIgnored let api: APIClient
    @ObservationIgnored let authService: AuthService

    init() {
        let auth = AuthStore()
        let api = APIClient(authStore: auth)
        self.auth = auth
        self.flow = AppFlowStore()
        self.store = PuzzleStore()
        self.api = api
        self.authService = AuthService(authStore: auth, apiClient: api)
        authService.configure()
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
