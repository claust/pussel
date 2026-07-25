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
      // Every bar item the wizard has is declared here, in one toolbar that
      // outlives the phase switch above — see `backButton` for why the phases
      // must not bring their own.
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          backButton
        }
        // The bar draws each item's glass background itself, behind the item's
        // content — so the hidden chevron below left an empty capsule sitting
        // in the corner of the home screen. Dropping the shared background is
        // what actually takes the button out of sight; it does not unmount the
        // item, so `backButton`'s reason for staying put still holds.
        .sharedBackgroundVisibility(model.flow.canGoBack ? .automatic : .hidden)
        ToolbarItem(placement: .topBarTrailing) {
          Menu {
            #if DEBUG
              BackendMenuSection()
            #endif
            // The email is the section's header rather than a row of its own:
            // headers render in a smaller font, so a full address fits on one
            // line where a body row wraps, and it reads as context for the
            // button it sits above.
            Section {
              Button("Sign Out", role: .destructive) {
                model.authService.signOut()
                model.flow.reset()
              }
            } header: {
              if let user = model.auth.user {
                Text(user.email)
              }
            }
          } label: {
            ProfileIconView(pictureURL: model.auth.avatarURL)
          }
          .accessibilityLabel("Account")
        }
      }
    }
  }

  /// The wizard's back chevron.
  ///
  /// Deliberately one button that is always in the bar, dimmed and disabled on
  /// the phases it doesn't apply to, rather than a `.toolbar` inside each
  /// phase's own view. A phase-owned toolbar item is inserted into the
  /// navigation bar only once SwiftUI has committed the new phase's content,
  /// and the bar crossfades it in from the previous phase's (empty) leading
  /// slot. Taps that land in that window hit the bar itself and are dropped —
  /// which is exactly what a tap right after opening a puzzle does, because
  /// reading and decoding that puzzle's images blocks the main thread first
  /// and pushes the item's arrival out past the tap. Reopening the same puzzle
  /// hits warm file and image caches, so the window is short enough to miss.
  ///
  /// Keeping the item mounted for the whole flow means only its content
  /// changes across a phase switch: the button is already installed and
  /// hit-testable the instant `canGoBack` turns true, so there is no window
  /// left to tap into.
  @ViewBuilder
  private var backButton: some View {
    let canGoBack = model.flow.canGoBack
    Button {
      model.flow.goBack()
    } label: {
      Image(systemName: "chevron.left")
    }
    .accessibilityLabel("Back to puzzles")
    // Hidden rather than omitted, so the item itself stays put. Disabled and
    // hidden from VoiceOver with it, so an invisible chevron is never a live
    // control on the home screen. The item's glass background is hidden
    // alongside it, up where the toolbar item is declared.
    .opacity(canGoBack ? 1 : 0)
    .disabled(!canGoBack)
    .accessibilityHidden(!canGoBack)
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
        Text(model.backend.displayName)
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
