import SwiftUI

struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Group {
            if model.auth.isAuthenticated {
                AppFlowView()
            } else {
                SignInView()
            }
        }
        .task {
            await model.authService.restoreSession()
        }
    }
}

/// Switches between the wizard phases once signed in.
struct AppFlowView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        NavigationStack {
            Group {
                switch model.flow.phase {
                case .capturePuzzle:
                    CapturePuzzleView()
                case .confirmTrim(let candidate):
                    ConfirmTrimView(candidate: candidate)
                case .solving(let session):
                    SolvingView(session: session)
                }
            }
            .navigationTitle("Pussel")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        if let user = model.auth.user {
                            Text(user.email)
                        }
                        Button("Sign Out", role: .destructive) {
                            model.authService.signOut()
                            model.flow.reset()
                        }
                    } label: {
                        Image(systemName: "person.crop.circle")
                    }
                }
            }
        }
    }
}
