import Foundation

/// Typed access to build-time configuration injected into Info.plist from
/// Config/*.xcconfig (see project.yml `info.properties`).
enum Config {
    static var apiBaseURL: URL {
        let raw = string(for: "APIBaseURL")
        if let url = URL(string: raw), url.scheme != nil {
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
