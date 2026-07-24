import Foundation
import UIKit
import os

/// DEBUG-only diagnostic dump of one glare-free capture burst: the raw
/// reference and corner shots, the composed output, and enough metadata to
/// reproduce the composite offline. Written under
/// `Documents/GlareFreeDumps/<yyyyMMdd-HHmmss-SSS>/`; pull it over USB with
/// `pymobiledevice3 apps pull dk.delectosoft.pussel /Documents/GlareFreeDumps <dest>`
/// (container access works for dev-signed builds without exposing Documents
/// through Finder file sharing).
/// There is no in-app UI for this; it exists purely so a bad composite seen
/// on a test device can be reproduced and debugged on a Mac afterwards.
///
/// A no-op in Release builds, and any failure here (disk full, encode
/// error) is logged and swallowed rather than thrown — diagnostics must
/// never be able to break the capture flow they're observing.
enum GlareFreeDump {
  #if DEBUG
    private static let log = Logger(subsystem: "dk.delectosoft.pussel", category: "glare-dump")
  #endif

  /// One dumped image's filename and pixel size, so `metadata.json` can be
  /// cross-checked against the files it sits beside without re-decoding
  /// them.
  private struct ImageInfo: Codable {
    let filename: String
    let width: Int
    let height: Int
  }

  private struct Metadata: Codable {
    let timestamp: String
    let images: [ImageInfo]
    /// Mirrors `GlareFreeComposer.compose`'s `expectedShifts` verbatim —
    /// null entries preserved, so a missing seed hypothesis for a given
    /// corner is visible rather than silently collapsed to zero.
    let expectedShifts: [CGSize?]?
    let alignedFrameCount: Int
    let appVersion: String
    let appBuild: String
  }

  /// `Documents/GlareFreeDumps` — the parent of every timestamped dump.
  static var defaultBaseDirectory: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("GlareFreeDumps", isDirectory: true)
  }

  /// Writes one burst's frames, composite, and metadata under
  /// `baseDirectory/<yyyyMMdd-HHmmss-SSS>/`. `reference`/`others` must be the
  /// original captures handed to `GlareFreeComposer.compose` — not its
  /// internally downscaled working copies — so the dump preserves the full
  /// resolution the on-device pipeline actually saw.
  ///
  /// Not actor-isolated: awaiting this from the `@MainActor` capture
  /// controller already runs the encode-and-write work off the main actor,
  /// since Swift schedules a nonisolated `async` function's body on the
  /// cooperative thread pool rather than the caller's actor. Callers that
  /// don't want to wait for it (the capture flow shouldn't stall on disk
  /// I/O) should fire it from an unstructured `Task` instead of awaiting it
  /// inline.
  ///
  /// Returns the dump directory on success, so tests can locate the files
  /// without duplicating the timestamp formatting; `nil` in Release builds
  /// or when a write failed partway through.
  @discardableResult
  static func record(
    reference: UIImage,
    others: [UIImage],
    expectedShifts: [CGSize?]?,
    composite: UIImage,
    alignedFrameCount: Int,
    baseDirectory: URL = defaultBaseDirectory
  ) async -> URL? {
    #if DEBUG
      let now = Date()
      let directoryFormatter = DateFormatter()
      directoryFormatter.locale = Locale(identifier: "en_US_POSIX")
      // Millisecond suffix so two bursts landing in the same second can't
      // silently share (and mix) a directory — createDirectory doesn't fail
      // on an existing path.
      directoryFormatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
      let directory = baseDirectory.appendingPathComponent(
        directoryFormatter.string(from: now), isDirectory: true)

      let fileManager = FileManager.default
      do {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
      } catch {
        log.error("could not create dump directory: \(error.localizedDescription)")
        return nil
      }

      var images: [ImageInfo] = []
      // Full resolution, no downscaling — these are the raw inputs an
      // offline re-run of the composer needs, so the whole point of the
      // dump would be lost if they were shrunk.
      guard write(reference, as: "reference.jpg", into: directory, recording: &images) else {
        return nil
      }
      for (index, other) in others.enumerated() {
        guard write(other, as: "corner_\(index + 1).jpg", into: directory, recording: &images)
        else { return nil }
      }
      guard write(composite, as: "composite.jpg", into: directory, recording: &images) else {
        return nil
      }

      let metadata = Metadata(
        timestamp: ISO8601DateFormatter().string(from: now),
        images: images,
        expectedShifts: expectedShifts,
        alignedFrameCount: alignedFrameCount,
        appVersion: bundleString("CFBundleShortVersionString"),
        appBuild: bundleString("CFBundleVersion"))
      do {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(metadata)
        try data.write(to: directory.appendingPathComponent("metadata.json"), options: .atomic)
      } catch {
        log.error("could not write metadata.json: \(error.localizedDescription)")
        return nil
      }
      return directory
    #else
      return nil
    #endif
  }

  #if DEBUG
    /// Encodes `image` as JPEG and writes it into `directory`, appending its
    /// filename and pixel size to `images` on success. Returns whether the
    /// write succeeded, so callers can bail out of the burst rather than
    /// leave `metadata.json` describing files that don't exist.
    private static func write(
      _ image: UIImage, as filename: String, into directory: URL,
      recording images: inout [ImageInfo]
    ) -> Bool {
      guard let data = image.jpegData(compressionQuality: 0.95) else {
        log.error("could not JPEG-encode \(filename)")
        return false
      }
      do {
        try data.write(to: directory.appendingPathComponent(filename), options: .atomic)
      } catch {
        log.error("could not write \(filename): \(error.localizedDescription)")
        return false
      }
      // Pixel size, not point size — matches how `GlareFreeComposer` reads
      // an image's actual resolution.
      let pixelSize = CGSize(
        width: image.size.width * image.scale, height: image.size.height * image.scale)
      images.append(
        ImageInfo(
          filename: filename, width: Int(pixelSize.width.rounded()),
          height: Int(pixelSize.height.rounded())))
      return true
    }

    private static func bundleString(_ key: String) -> String {
      (Bundle.main.object(forInfoDictionaryKey: key) as? String) ?? ""
    }
  #endif
}
