import XCTest

@testable import Pussel

/// Covers the account menu's backend label. The menu is only about 250pt
/// wide, so a full URL wraps onto a second line; these lock in the compact
/// form that fits.
final class BackendSelectionTests: XCTestCase {
  func testDropsTheSchemeAndKeepsANonDefaultPort() {
    XCTAssertEqual(
      BackendSelection.displayName(for: URL(string: "http://localhost:8000")!),
      "localhost:8000")
    XCTAssertEqual(
      BackendSelection.displayName(for: URL(string: "http://192.168.86.63:8000")!),
      "192.168.86.63:8000")
  }

  /// A default port is not in the URL, so nothing is appended — the deployed
  /// backend reads as a bare hostname.
  func testOmitsAnImplicitPort() {
    XCTAssertEqual(
      BackendSelection.displayName(for: URL(string: "https://pussel.example.invalid")!),
      "pussel.example.invalid")
    XCTAssertEqual(
      BackendSelection.displayName(for: URL(string: "https://pussel.example.invalid/")!),
      "pussel.example.invalid")
  }

  /// Config.backendURL(from:) already guarantees a host upstream, so this is
  /// only a belt-and-braces fallback rather than a case the menu can hit.
  func testFallsBackToTheWholeURLWithoutAHost() {
    XCTAssertEqual(
      BackendSelection.displayName(for: URL(string: "https://")!),
      "https://")
  }
}
