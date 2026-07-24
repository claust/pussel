import Foundation
import Observation

/// Which backend the app talks to, and whether that is still up for debate.
///
/// Release builds are fixed to the deployed backend baked into the build.
/// Debug builds carry both URLs — the local FastAPI server (`API_BASE_URL`,
/// `make start-backend`) and the deployed one (`RELEASE_API_BASE_URL`) — and
/// the account menu flips between them without a rebuild, so the same install
/// can check a home-server deploy and then go back to local work. The choice
/// persists across launches in UserDefaults.
@Observable
@MainActor
final class BackendSelection {
  private static let defaultsKey = "usesLocalBackend"

  /// True when requests go to the local backend rather than the deployed one.
  /// Always false in Release builds. Change it through `AppModel`, which also
  /// repoints the live `APIClient` and drops the now-foreign session.
  private(set) var usesLocalBackend: Bool

  @ObservationIgnored private let defaults: UserDefaults

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    #if DEBUG
      // Debug builds keep pointing at the local backend by default — the
      // deployed one is opt-in, and only offered when the build knows it.
      let stored = defaults.object(forKey: Self.defaultsKey) as? Bool
      usesLocalBackend = Config.remoteAPIBaseURL == nil || (stored ?? true)
    #else
      usesLocalBackend = false
    #endif
  }

  /// The backend every request is currently addressed to.
  var baseURL: URL {
    #if DEBUG
      if !usesLocalBackend, let remote = Config.remoteAPIBaseURL {
        return remote
      }
    #endif
    return Config.apiBaseURL
  }

  /// False when there is nothing to switch to, so the account menu can leave
  /// the row out entirely: Release builds, and Debug builds whose
  /// Secrets.xcconfig defines no `RELEASE_API_BASE_URL`.
  var isSwitchable: Bool {
    #if DEBUG
      return Config.remoteAPIBaseURL != nil
    #else
      return false
    #endif
  }

  /// Records the choice. Callers go through `AppModel.useLocalBackend(_:)`,
  /// which pairs this with the session and wizard reset the switch requires.
  func setUsesLocalBackend(_ useLocal: Bool) {
    guard isSwitchable else { return }
    usesLocalBackend = useLocal
    defaults.set(useLocal, forKey: Self.defaultsKey)
  }
}
