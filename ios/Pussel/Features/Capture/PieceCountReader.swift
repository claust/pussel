import CoreGraphics
import Foundation
import ImageIO
import Vision

/// One OCR text line reduced to what the count filter needs: the recognized
/// string, Vision's confidence, and the observation's normalized bounding-box
/// height. Height is the "tallest token wins" tie-breaker — the piece count is
/// the biggest number on the box, which is what beats look-alike numerals like
/// the "Since 1891" tagline, age badges, and centimetre dimensions.
struct PieceCountToken: Equatable {
  let text: String
  let confidence: Float
  /// Normalized ([0,1]) height of the text observation's bounding box.
  let height: Float
}

/// The closed-set count filter, ported verbatim from the Python estimator
/// (`_extract_candidates` in backend/app/services/piece_count_estimator.py) so
/// the "no wrong guesses" guarantee is identical here: only tokens that exactly
/// match a piece count Ravensburger actually sells — plus "N x M" kids
/// multipacks, where the per-puzzle M is the count — survive, and the tallest
/// survivor wins.
///
/// Kept separate from the Vision request so it is testable without a running
/// Vision context: it operates on plain `PieceCountToken`s.
enum PieceCountFilter {
  /// Piece counts Ravensburger sells as single-puzzle boxes. Deliberately a
  /// closed set: exact-match filtering is what keeps OCR noise (years, article
  /// numbers, box dimensions) from ever becoming an estimate.
  static let plainCounts: Set<Int> = [
    24, 35, 49, 54, 60, 80, 100, 125, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000,
  ]
  /// Kids multipacks are printed "N x M" (e.g. "2x12", "3x49"); N is the
  /// number of puzzles in the box and M — the count — the pieces per puzzle.
  /// The small-N requirement rejects box dimensions like "70x50".
  static let multipackPuzzles: Set<Int> = [2, 3, 4]
  static let multipackPieces: Set<Int> = [12, 20, 24, 49, 100]

  /// Vision confidence floor. Matches the investigation benchmark
  /// (`vision_bench.py`), where every correct badge read scored well above it
  /// and this cutoff turned would-be wrong guesses into safe Nones.
  static let minConfidence: Float = 0.5

  /// Leading/trailing punctuation Vision sometimes glues onto a numeral.
  private static let stripCharacters = CharacterSet(charactersIn: "\"'.,;:()|«»\u{201C}\u{201D}*")

  /// Picks the most likely piece count from a photo's OCR tokens, or nil when
  /// nothing on the box confidently matches a known count — never a
  /// low-confidence guess.
  static func estimate(from tokens: [PieceCountToken]) -> Int? {
    var best: (value: Int, height: Float)?
    for token in tokens where token.confidence >= minConfidence {
      // A single observation can hold several words ("1000 Teile"); the height
      // is the line's, so every word in it competes on the same scale — as the
      // Python tokens (tesseract words) and the Vision benchmark both do.
      for word in token.text.split(whereSeparator: { $0.isWhitespace }) {
        guard let value = countValue(from: String(word)) else { continue }
        if best == nil || token.height > best!.height {
          best = (value, token.height)
        }
      }
    }
    return best?.value
  }

  /// Reduces one OCR word to a known piece count, or nil if it isn't one.
  static func countValue(from word: String) -> Int? {
    var token = word.trimmingCharacters(in: stripCharacters)
    token = token.replacingOccurrences(of: "×", with: "x")
    token = collapseThousandsSeparator(token)
    if let pieces = multipackValue(token) {
      return pieces
    }
    // Plain count: 2–4 digits (`^\d{2,4}$`) that hits the closed set.
    if (2...4).contains(token.count), token.allSatisfy(\.isNumber), let value = Int(token),
      plainCounts.contains(value)
    {
      return value
    }
    return nil
  }

  /// The per-puzzle piece count of an "N x M" multipack token, or nil.
  private static func multipackValue(_ token: String) -> Int? {
    let lowered = token.lowercased()
    guard let separator = lowered.firstIndex(of: "x") else { return nil }
    let left = String(lowered[lowered.startIndex..<separator])
    let right = String(lowered[lowered.index(after: separator)...])
    guard let puzzles = Int(left), let pieces = Int(right),
      left.allSatisfy(\.isNumber), right.allSatisfy(\.isNumber),
      multipackPuzzles.contains(puzzles), multipackPieces.contains(pieces)
    else {
      return nil
    }
    return pieces
  }

  /// Collapses a thousands separator inside a count ("1.000" / "1,000" → "1000"),
  /// matching the Python `_SEP_RE` — a single leading digit, a `.`/`,`, then
  /// exactly three digits. Anything else is returned unchanged.
  private static func collapseThousandsSeparator(_ token: String) -> String {
    let characters = Array(token)
    guard characters.count == 5, characters[0].isNumber,
      characters[1] == "." || characters[1] == ",",
      characters[2].isNumber, characters[3].isNumber, characters[4].isNumber
    else {
      return token
    }
    return String([characters[0], characters[2], characters[3], characters[4]])
  }
}

/// On-device piece-count reader for the detect-frame photo path: OCRs the
/// user's photo of a puzzle box with Apple's Vision text recognizer and returns
/// the piece count printed on it, filtered through `PieceCountFilter`.
///
/// Replaces the backend tesseract OCR (`Layout.PHOTO` in
/// backend/app/services/piece_count_estimator.py) that used to prefill the
/// confirm screen. Vision reads real box photos far better than tesseract —
/// including the dim, sideways, uncropped shots the backend path missed (33/38
/// with zero wrong reads across the investigation benchmark, versus 14/28 for
/// the tesseract photo path) — and it reads the raw photo directly, so no crop
/// or orientation-normalization step is needed. Running it locally also drops
/// it out of the detect-frame round trip: the count is read in parallel with
/// the upload, not waited on after it.
///
/// The barcode-lookup path keeps its backend read (the CDN product shot, where
/// tesseract holds its own and Vision misses one).
///
/// Stateless and thread-safe: each call builds its own request. The recognizer
/// is CPU-heavy, so callers run it off the main actor (see `AppModel.startTrim`).
enum PieceCountReader {
  /// Reads the count from a decoded box photo. `orientation` describes how the
  /// pixels sit relative to upright; the normalized upload JPEG bakes EXIF
  /// orientation in, so its default `.up` is correct for that path.
  static func read(cgImage: CGImage, orientation: CGImagePropertyOrientation = .up) -> Int? {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    // Off: a puzzle count is a bare numeral, and language correction would
    // "helpfully" reshape stray digits toward words.
    request.usesLanguageCorrection = false
    let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
    do {
      try handler.perform([request])
    } catch {
      // A failed recognizer is a missing estimate, not an error to surface —
      // the confirm screen simply opens with an empty field.
      return nil
    }
    let tokens = (request.results ?? []).compactMap { observation -> PieceCountToken? in
      guard let candidate = observation.topCandidates(1).first else { return nil }
      return PieceCountToken(
        text: candidate.string,
        confidence: candidate.confidence,
        height: Float(observation.boundingBox.height))
    }
    return PieceCountFilter.estimate(from: tokens)
  }

  /// Reads the count from JPEG bytes, decoding them first. Returns nil on
  /// undecodable bytes.
  static func read(jpegData: Data) -> Int? {
    guard let source = CGImageSourceCreateWithData(jpegData as CFData, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      return nil
    }
    return read(cgImage: cgImage)
  }
}
