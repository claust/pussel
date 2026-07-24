import Foundation

/// Typed access to build-time configuration injected into Info.plist from
/// Config/*.xcconfig (see project.yml `info.properties`).
enum Config {
  static var apiBaseURL: URL {
    let raw = string(for: "APIBaseURL")
    if let url = backendURL(from: raw) {
      return url
    }
    #if DEBUG
      return URL(string: "http://localhost:8000")!
    #else
      // A silent localhost fallback would mask a broken Release config;
      // fail fast instead.
      fatalError("APIBaseURL is missing or invalid in Info.plist: '\(raw)'")
    #endif
  }

  /// The deployed backend (`RELEASE_API_BASE_URL` in Config/Secrets.xcconfig),
  /// or nil when the build has no valid value for it. Release builds already
  /// resolve `apiBaseURL` to this; Debug builds keep it alongside their local
  /// URL so `BackendSelection` can flip between the two at runtime.
  static var remoteAPIBaseURL: URL? {
    backendURL(from: string(for: "RemoteAPIBaseURL"))
  }

  /// Parses a configured backend URL, rejecting anything requests can't be
  /// built against. A scheme alone is too weak a test: `https://` and `http:/`
  /// both parse with a scheme and no host, and this project's xcconfigs have
  /// to write URLs as `https:/$()/host` to keep `//` from starting a comment —
  /// so a value truncated at the `$()` is exactly the shape that slips
  /// through. Without the host check that yields a Debug build offering a
  /// backend switch whose requests go nowhere, and a Release build making
  /// malformed requests where the `fatalError` above is meant to fire.
  static func backendURL(from raw: String) -> URL? {
    guard let url = URL(string: raw),
      let scheme = url.scheme?.lowercased(),
      scheme == "http" || scheme == "https",
      let host = url.host(), !host.isEmpty
    else {
      return nil
    }
    return url
  }

  static var googleIOSClientID: String { string(for: "GoogleIOSClientID") }

  static var googleServerClientID: String { string(for: "GoogleServerClientID") }

  /// True once Secrets.xcconfig holds real client ids (not the placeholders).
  /// Both are needed: the iOS id for the sign-in sheet, the server id for a
  /// backend-accepted ID-token audience.
  static var isGoogleSignInConfigured: Bool {
    isRealClientID(googleIOSClientID) && isRealClientID(googleServerClientID)
  }

  private static func isRealClientID(_ id: String) -> Bool {
    id.hasSuffix(".apps.googleusercontent.com") && !id.hasPrefix("YOUR_")
  }

  private static func string(for key: String) -> String {
    (Bundle.main.object(forInfoDictionaryKey: key) as? String) ?? ""
  }
}
