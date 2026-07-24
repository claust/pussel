import UIKit
import XCTest

@testable import Pussel

/// Tests for the DEBUG-only glare-free diagnostic dump. Verifies the on-disk
/// contract (`GlareFreeDump.swift`'s doc comment) against a temp directory
/// rather than the real Documents folder, and decodes `metadata.json`
/// through a locally-defined mirror of its schema — a black-box check that
/// doesn't reach into `GlareFreeDump`'s private `Metadata` type.
final class GlareFreeDumpTests: XCTestCase {
  private struct DecodedImage: Decodable {
    let filename: String
    let width: Int
    let height: Int
  }

  private struct DecodedMetadata: Decodable {
    let timestamp: String
    let images: [DecodedImage]
    let expectedShifts: [CGSize?]?
    let alignedFrameCount: Int
    let appVersion: String
    let appBuild: String
  }

  private var tempBaseDirectory: URL!

  override func setUp() {
    super.setUp()
    tempBaseDirectory = FileManager.default.temporaryDirectory
      .appendingPathComponent("GlareFreeDumpTests-\(UUID().uuidString)", isDirectory: true)
  }

  override func tearDown() {
    try? FileManager.default.removeItem(at: tempBaseDirectory)
    tempBaseDirectory = nil
    super.tearDown()
  }

  /// A tiny solid-color image, sized so its width and height are
  /// distinguishable (round-tripping a square would hide a swapped
  /// width/height bug).
  private func solidImage(width: Int, height: Int, color: UIColor) -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
      .image { context in
        color.setFill()
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
      }
  }

  func testRecordWritesAllFilesAndDecodableMetadata() async throws {
    let reference = solidImage(width: 12, height: 8, color: .systemRed)
    let others = [
      solidImage(width: 12, height: 8, color: .systemBlue),
      solidImage(width: 12, height: 8, color: .systemGreen),
      solidImage(width: 12, height: 8, color: .systemYellow),
      solidImage(width: 12, height: 8, color: .systemPurple),
    ]
    let composite = solidImage(width: 12, height: 8, color: .black)
    let expectedShifts: [CGSize?] = [
      CGSize(width: 0.25, height: 0.18), nil, CGSize(width: -0.25, height: -0.18), nil,
    ]

    let directory = await GlareFreeDump.record(
      reference: reference, others: others, expectedShifts: expectedShifts, composite: composite,
      alignedFrameCount: 3, baseDirectory: tempBaseDirectory)
    let dumpDirectory = try XCTUnwrap(directory, "record should report the directory it wrote")

    // All seven files exist: the five captures, the composite, and the
    // metadata sidecar.
    let expectedFilenames = [
      "reference.jpg", "corner_1.jpg", "corner_2.jpg", "corner_3.jpg", "corner_4.jpg",
      "composite.jpg", "metadata.json",
    ]
    for filename in expectedFilenames {
      XCTAssertTrue(
        FileManager.default.fileExists(atPath: dumpDirectory.appendingPathComponent(filename).path),
        "\(filename) should have been written")
    }
    let onDisk = try FileManager.default.contentsOfDirectory(atPath: dumpDirectory.path)
    XCTAssertEqual(Set(onDisk), Set(expectedFilenames))

    // The JPEGs round-trip: decoding each back gives the same pixel size as
    // the image that went in.
    let allImages =
      [("reference.jpg", reference)]
      + zip(1...4, others).map {
        ("corner_\($0).jpg", $1)
      } + [("composite.jpg", composite)]
    for (filename, original) in allImages {
      let path = dumpDirectory.appendingPathComponent(filename).path
      let roundTripped = try XCTUnwrap(
        UIImage(contentsOfFile: path), "\(filename) should decode back to an image")
      XCTAssertEqual(roundTripped.size, original.size, "\(filename) pixel size should round-trip")
    }

    // metadata.json decodes and matches what was passed in.
    let metadataData = try Data(
      contentsOf: dumpDirectory.appendingPathComponent("metadata.json"))
    let metadata = try JSONDecoder().decode(DecodedMetadata.self, from: metadataData)
    XCTAssertNotNil(
      ISO8601DateFormatter().date(from: metadata.timestamp), "timestamp should be ISO8601")
    XCTAssertEqual(metadata.images.count, 6)
    XCTAssertEqual(Set(metadata.images.map(\.filename)), Set(expectedFilenames.dropLast()))
    for image in metadata.images {
      XCTAssertEqual(image.width, 12)
      XCTAssertEqual(image.height, 8)
    }
    XCTAssertEqual(metadata.alignedFrameCount, 3)
    XCTAssertFalse(metadata.appVersion.isEmpty)
    XCTAssertFalse(metadata.appBuild.isEmpty)

    // Null entries in expectedShifts survive the round trip in position —
    // collapsing them to zero would misrepresent "no seed hypothesis" as
    // "seeded with zero shift".
    let shifts = try XCTUnwrap(metadata.expectedShifts)
    XCTAssertEqual(shifts.count, 4)
    XCTAssertEqual(shifts[0], CGSize(width: 0.25, height: 0.18))
    XCTAssertNil(shifts[1])
    XCTAssertEqual(shifts[2], CGSize(width: -0.25, height: -0.18))
    XCTAssertNil(shifts[3])
  }

  func testRecordCreatesATimestampedSubdirectoryUnderBaseDirectory() async throws {
    let image = solidImage(width: 4, height: 4, color: .white)
    let directory = await GlareFreeDump.record(
      reference: image, others: [], expectedShifts: nil, composite: image, alignedFrameCount: 0,
      baseDirectory: tempBaseDirectory)
    let dumpDirectory = try XCTUnwrap(directory)
    XCTAssertEqual(dumpDirectory.deletingLastPathComponent(), tempBaseDirectory)
    XCTAssertNotEqual(dumpDirectory.lastPathComponent, "")
  }
}
