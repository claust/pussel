import XCTest

@testable import Pussel

/// Tests for the closed-set count filter — the "no wrong guesses" guarantee
/// the on-device Vision path shares with the Python estimator
/// (`_extract_candidates` in backend/app/services/piece_count_estimator.py).
/// The Vision request itself isn't exercised here; these pin the filtering
/// logic that turns raw OCR tokens into a count or a safe nil.
final class PieceCountReaderTests: XCTestCase {
  /// Convenience: a token tall and confident enough to survive the gates.
  private func token(_ text: String, confidence: Float = 0.9, height: Float = 0.1)
    -> PieceCountToken
  {
    PieceCountToken(text: text, confidence: confidence, height: height)
  }

  // MARK: - Plain counts

  func testReadsEveryKnownPlainCount() {
    for count in PieceCountFilter.plainCounts {
      XCTAssertEqual(
        PieceCountFilter.estimate(from: [token(String(count))]), count, "count=\(count)")
    }
  }

  func testCollapsesThousandsSeparator() {
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("1.000")]), 1000)
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("1,000")]), 1000)
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("2.000")]), 2000)
  }

  func testStripsSurroundingPunctuation() {
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("«500»")]), 500)
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("(1000)")]), 1000)
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("\u{201C}300\u{201D}")]), 300)
  }

  // MARK: - Multipacks

  func testReadsMultipackPerPuzzlePieces() {
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("2x12")]), 12)
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("3x49")]), 49)
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("4x100")]), 100)
  }

  func testReadsMultipackWithUnicodeTimesSign() {
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("2×24")]), 24)
  }

  func testRejectsBoxDimensionsPosingAsMultipack() {
    // "70x50" — large N, not a kids multipack; and neither side is a count.
    XCTAssertNil(PieceCountFilter.estimate(from: [token("70x50")]))
    // Plausible N but the pieces side isn't a sold multipack size.
    XCTAssertNil(PieceCountFilter.estimate(from: [token("2x15")]))
  }

  // MARK: - Rejections (no wrong guesses)

  func testRejectsNumbersOutsideTheClosedSet() {
    // A year, an article number, a dimension, an age — all look like counts.
    for noise in ["1891", "12000622", "70", "36", "18", "1200"] {
      XCTAssertNil(PieceCountFilter.estimate(from: [token(noise)]), "noise=\(noise)")
    }
  }

  func testRejectsNonNumericText() {
    XCTAssertNil(PieceCountFilter.estimate(from: [token("Teile")]))
    XCTAssertNil(PieceCountFilter.estimate(from: [token("Ravensburger")]))
  }

  func testRejectsTokensBelowConfidenceFloor() {
    XCTAssertNil(PieceCountFilter.estimate(from: [token("1000", confidence: 0.49)]))
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("1000", confidence: 0.5)]), 1000)
  }

  // MARK: - Tallest token wins

  func testPrefersTheTallestMatchingToken() {
    // The fine-print "1000 Teile" contents line (short) must lose to the
    // big front-of-box "500" badge (tall) when both are on the box.
    let tokens = [
      token("1000", height: 0.04),
      token("500", height: 0.18),
    ]
    XCTAssertEqual(PieceCountFilter.estimate(from: tokens), 500)
  }

  func testPicksTheCountWordOutOfAMultiWordLine() {
    // Vision returns "1000 Teile" as one observation; the numeral still reads.
    XCTAssertEqual(PieceCountFilter.estimate(from: [token("1000 Teile")]), 1000)
  }

  func testEmptyTokensYieldNil() {
    XCTAssertNil(PieceCountFilter.estimate(from: []))
  }

  // MARK: - Undecodable input

  func testReadReturnsNilOnUndecodableBytes() {
    XCTAssertNil(PieceCountReader.read(jpegData: Data([0x00, 0x01, 0x02])))
  }
}
