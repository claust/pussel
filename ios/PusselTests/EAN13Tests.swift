import XCTest

@testable import Pussel

final class EAN13Tests: XCTestCase {
  func testValidChecksumAccepted() {
    // The Frozen II 2x12 box (article 05009) and a known adult-line code.
    XCTAssertTrue(EAN13.isValidChecksum("4005556050093"))
    XCTAssertTrue(EAN13.isValidChecksum("4005555006220"))
  }

  func testInvalidCheckDigitRejected() {
    XCTAssertFalse(EAN13.isValidChecksum("4005556050094"))
  }

  func testWrongLengthRejected() {
    XCTAssertFalse(EAN13.isValidChecksum("400555605009"))
    XCTAssertFalse(EAN13.isValidChecksum("40055560500933"))
    XCTAssertFalse(EAN13.isValidChecksum(""))
  }

  func testNonDigitInputRejected() {
    XCTAssertFalse(EAN13.isValidChecksum("400555605009X"))
    // Multi-scalar characters must not sneak through the count/compactMap.
    XCTAssertFalse(EAN13.isValidChecksum("400555605009🧩"))
  }
}
