import SwiftUI

struct SignInView: View {
  @Environment(AppModel.self) private var model
  @State private var backendHealthy: Bool?

  var body: some View {
    VStack(spacing: 16) {
      Spacer()
      Image(systemName: "puzzlepiece.extension.fill")
        .font(.system(size: 64))
        .foregroundStyle(.tint)
      Text("Pussel")
        .font(.largeTitle.bold())
      Text("Photograph your puzzle, then each piece,\nand see where it belongs.")
        .multilineTextAlignment(.center)
        .foregroundStyle(.secondary)
      Spacer()
      if model.auth.isSigningIn {
        ProgressView()
      } else {
        Button {
          Task { await model.authService.signIn() }
        } label: {
          Label("Sign in with Google", systemImage: "person.crop.circle")
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(!Config.isGoogleSignInConfigured)
        if !Config.isGoogleSignInConfigured {
          Text(
            "Google Sign-In isn't configured yet.\nFill in ios/Config/Secrets.xcconfig and rebuild."
          )
          .font(.footnote)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
        }
      }
      if let error = model.auth.errorMessage {
        Text(error)
          .font(.footnote)
          .foregroundStyle(.red)
          .multilineTextAlignment(.center)
      }
      statusFooter
    }
    .padding(24)
    .task {
      backendHealthy = await model.api.checkHealth()
    }
  }

  private var statusFooter: some View {
    VStack(spacing: 2) {
      Text(Config.apiBaseURL.absoluteString)
      Text("Backend: \(backendHealthy.map { $0 ? "healthy" : "unreachable" } ?? "checking…")")
    }
    .font(.caption.monospaced())
    .foregroundStyle(.secondary)
    .padding(.top, 8)
  }
}
