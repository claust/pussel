import SwiftUI

@Observable
@MainActor
final class AppModel {
    let auth: AuthStore
    let flow: AppFlowStore
    @ObservationIgnored let api: APIClient
    @ObservationIgnored let authService: AuthService

    init() {
        let auth = AuthStore()
        let api = APIClient(authStore: auth)
        self.auth = auth
        self.flow = AppFlowStore()
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
                .onOpenURL { url in
                    #if DEBUG
                        if model.handleDebugURL(url) { return }
                    #endif
                    model.authService.handle(url: url)
                }
                #if DEBUG
                    .task { DebugDriver.start(model) }
                #endif
        }
    }
}
