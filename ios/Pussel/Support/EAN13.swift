import Foundation

/// Pure EAN-13 checksum validation. Vision (and AVFoundation) occasionally
/// misread a digit, so every scanned payload is re-verified locally before
/// it's allowed to trigger a backend lookup.
enum EAN13 {
  /// Whether `code` is exactly 13 digits with a valid GS1 mod-10 check digit
  /// (weights 1,3,1,3,… over the first 12 digits).
  static func isValidChecksum(_ code: String) -> Bool {
    guard code.count == 13 else { return false }
    let digits = code.compactMap(\.wholeNumberValue)
    guard digits.count == 13 else { return false }
    let weightedSum = digits[0..<12].enumerated().reduce(0) { sum, pair in
      sum + pair.element * (pair.offset % 2 == 0 ? 1 : 3)
    }
    return (10 - weightedSum % 10) % 10 == digits[12]
  }
}
