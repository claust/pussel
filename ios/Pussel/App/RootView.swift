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
            #if DEBUG
              BackendMenuSection()
            #endif
            Button("Sign Out", role: .destructive) {
              model.authService.signOut()
              model.flow.reset()
            }
          } label: {
            ProfileIconView(pictureURL: model.auth.avatarURL)
          }
          .accessibilityLabel("Account")
        }
      }
    }
  }
}

#if DEBUG
  /// Debug-only backend switch inside the account menu: a checkmark row that
  /// flips between the local FastAPI server and the deployed home server, with
  /// the current URL underneath it. Release builds don't show it — they only
  /// ever have the one backend (see `BackendSelection`).
  private struct BackendMenuSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
      Section("Backend") {
        if model.backend.isSwitchable {
          Toggle(
            "Use Local Backend",
            isOn: Binding(
              get: { model.backend.usesLocalBackend },
              set: { model.useLocalBackend($0) }
            )
          )
        }
        Text(model.backend.baseURL.absoluteString)
      }
    }
  }
#endif

/// The account menu icon: the user's Google profile picture when available
/// (fetched via AsyncImage), otherwise the generic person symbol.
private struct ProfileIconView: View {
  let pictureURL: URL?

  private static let iconSize: CGFloat = 28

  var body: some View {
    Group {
      if let pictureURL {
        AsyncImage(url: pictureURL) { image in
          image
            .resizable()
            .scaledToFill()
        } placeholder: {
          fallbackIcon
        }
      } else {
        fallbackIcon
      }
    }
    .frame(width: Self.iconSize, height: Self.iconSize)
    .clipShape(Circle())
  }

  private var fallbackIcon: some View {
    Image(systemName: "person.crop.circle")
      .resizable()
      .scaledToFit()
  }
}
