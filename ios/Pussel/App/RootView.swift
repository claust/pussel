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
            ProfileIconView(pictureURL: model.auth.user?.picture.flatMap(URL.init(string:)))
          }
          .accessibilityLabel("Account")
        }
      }
    }
  }
}

/// The account menu icon: the user's Google profile picture when available
/// (fetched via AsyncImage, cached by URLCache), otherwise the generic symbol.
struct ProfileIconView: View {
  let pictureURL: URL?

  private static let iconSize: CGFloat = 28

  var body: some View {
    if let pictureURL {
      AsyncImage(url: pictureURL) { image in
        image
          .resizable()
          .scaledToFill()
          .frame(width: Self.iconSize, height: Self.iconSize)
          .clipShape(Circle())
      } placeholder: {
        fallbackIcon
      }
    } else {
      fallbackIcon
    }
  }

  private var fallbackIcon: some View {
    Image(systemName: "person.crop.circle")
  }
}
