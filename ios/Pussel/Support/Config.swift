import Foundation

/// Typed access to build-time configuration injected into Info.plist from
/// Config/*.xcconfig (see project.yml `info.properties`).
enum Config {
    static var apiBaseURL: URL {
        URL(string: string(for: "APIBaseURL")) ?? URL(string: "http://localhost:8000")!
    }

    static var googleIOSClientID: String { string(for: "GoogleIOSClientID") }

    static var googleServerClientID: String { string(for: "GoogleServerClientID") }

    /// True once Secrets.xcconfig holds a real iOS client id (not the placeholder).
    static var isGoogleSignInConfigured: Bool {
        googleIOSClientID.hasSuffix(".apps.googleusercontent.com")
            && !googleIOSClientID.hasPrefix("YOUR_")
    }

    private static func string(for key: String) -> String {
        (Bundle.main.object(forInfoDictionaryKey: key) as? String) ?? ""
    }
}
