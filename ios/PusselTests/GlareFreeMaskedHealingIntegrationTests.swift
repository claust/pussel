import UIKit
import XCTest

@testable import Pussel

/// Env-gated integration test: runs the real masked-healing `compose()`
/// against a real on-device capture dump pulled to the Mac (see
/// `GlareFreeDump.swift`'s doc comment for the pull recipe), and writes the
/// result as `swift_masked.jpg` so `scripts/stitch_quality/score_stitch.py`
/// can score it against the Python `--mode masked` reference the same way
/// it scores the app's own `composite.jpg`. Skipped unless `GLARE_DUMP_DIR`
/// is set — this isn't part of the regular suite, since it needs a real
/// capture dump that isn't (and shouldn't be) committed to the repo.
///
/// Simulator tests share the host Mac's filesystem, so an absolute Mac path
/// in `GLARE_DUMP_DIR` works directly. `xcodebuild test` strips a
/// `TEST_RUNNER_` prefix off environment variables before the test process
/// sees them, so set it on the invocation:
///
///   TEST_RUNNER_GLARE_DUMP_DIR=/path/to/dump \
///     xcodebuild test -project ios/Pussel.xcodeproj -scheme Pussel \
///     -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
///     -only-testing:PusselTests/GlareFreeMaskedHealingIntegrationTests
///
/// `GLARE_DUMP_OUT` (optional, same `TEST_RUNNER_` prefix) redirects the
/// written file elsewhere; it defaults to alongside the dump itself.
final class GlareFreeMaskedHealingIntegrationTests: XCTestCase {
  /// Mirrors `GlareFreeDump`'s private `Metadata` type just enough to read
  /// back `expectedShifts` — the same black-box decode approach
  /// `GlareFreeDumpTests` uses, so this test doesn't need access to
  /// `GlareFreeDump`'s internals.
  private struct DecodedMetadata: Decodable {
    let expectedShifts: [CGSize?]?
  }

  func testComposeRealDumpWritesSwiftMasked() throws {
    guard let dumpDirPath = ProcessInfo.processInfo.environment["GLARE_DUMP_DIR"] else {
      throw XCTSkip(
        "GLARE_DUMP_DIR not set -- skipping the real-dump masked-healing integration test")
    }
    let dumpDir = URL(fileURLWithPath: dumpDirPath, isDirectory: true)
    let outDir =
      ProcessInfo.processInfo.environment["GLARE_DUMP_OUT"].map {
        URL(fileURLWithPath: $0, isDirectory: true)
      }
      ?? dumpDir

    let reference = try loadImage(dumpDir.appendingPathComponent("reference.jpg"))
    var others: [UIImage] = []
    var presentIndices: [Int] = []
    for index in 1...4 {
      let path = dumpDir.appendingPathComponent("corner_\(index).jpg")
      guard FileManager.default.fileExists(atPath: path.path) else { continue }
      others.append(try loadImage(path))
      presentIndices.append(index)
    }
    XCTAssertFalse(others.isEmpty, "dump at \(dumpDirPath) should have at least one corner_N.jpg")

    // `expectedShifts` in `metadata.json` is indexed like `others` here
    // (one entry per `corner_N.jpg` in ascending N, dropped entries for
    // missing corners) -- mirrors how the capture controller builds it in
    // production (`GlareFreeCaptureController.expectedShift(step:)`).
    var expectedShifts: [CGSize?]?
    let metadataPath = dumpDir.appendingPathComponent("metadata.json")
    if let metadataData = try? Data(contentsOf: metadataPath),
      let metadata = try? JSONDecoder().decode(DecodedMetadata.self, from: metadataData),
      let allShifts = metadata.expectedShifts
    {
      expectedShifts = presentIndices.map { index in
        index - 1 < allShifts.count ? allShifts[index - 1] : nil
      }
    }

    guard
      let result = GlareFreeComposer.compose(
        reference: reference, others: others, expectedShifts: expectedShifts)
    else {
      return XCTFail("compose returned nil for the real dump at \(dumpDirPath)")
    }
    print(
      "GlareFreeMaskedHealingIntegrationTests: \(dumpDirPath) alignedFrameCount="
        + "\(result.alignedFrameCount)/\(others.count)")

    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let outputPath = outDir.appendingPathComponent("swift_masked.jpg")
    guard let data = result.image.jpegData(compressionQuality: 0.95) else {
      return XCTFail("could not JPEG-encode the composite for \(dumpDirPath)")
    }
    try data.write(to: outputPath, options: .atomic)
    print("GlareFreeMaskedHealingIntegrationTests: wrote \(outputPath.path)")
  }

  private func loadImage(_ url: URL) throws -> UIImage {
    let data = try Data(contentsOf: url)
    return try XCTUnwrap(UIImage(data: data), "\(url.path) should decode as an image")
  }
}
