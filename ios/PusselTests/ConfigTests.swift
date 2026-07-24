import XCTest

@testable import Pussel

/// Covers `Config.backendURL(from:)`, the gate both `apiBaseURL` and
/// `remoteAPIBaseURL` run configured values through.
final class ConfigTests: XCTestCase {
  func testAcceptsConfiguredBackendURLs() {
    for raw in [
      "http://localhost:8000",
      "http://192.168.86.63:8000",
      "https://pussel.example.invalid",
      "https://pussel.example.invalid/",
      "HTTPS://pussel.example.invalid",
    ] {
      XCTAssertEqual(Config.backendURL(from: raw)?.absoluteString, raw, "raw=\(raw)")
    }
  }

  /// A scheme-only check would admit the first two: both parse with a scheme
  /// and no host. They are the realistic failure here because xcconfigs must
  /// write `https:/$()/host` to stop `//` starting a comment, so a value
  /// truncated at the `$()` lands exactly on `https://`.
  func testRejectsSchemeWithoutHost() {
    XCTAssertNil(Config.backendURL(from: "https://"))
    XCTAssertNil(Config.backendURL(from: "http:/"))
  }

  func testRejectsMissingOrUnexpandedValues() {
    // Empty when the xcconfig omits the key; the literal when a build setting
    // never got substituted.
    XCTAssertNil(Config.backendURL(from: ""))
    XCTAssertNil(Config.backendURL(from: "$(RELEASE_API_BASE_URL)"))
    XCTAssertNil(Config.backendURL(from: "pussel.example.invalid"))
  }

  /// Requests are built with `URLRequest`, so a non-http(s) scheme could not
  /// reach a backend even though it parses.
  func testRejectsNonHTTPSchemes() {
    XCTAssertNil(Config.backendURL(from: "ftp://pussel.example.invalid"))
    XCTAssertNil(Config.backendURL(from: "pusseldebug://camera"))
  }
}
