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

  /// One shot's camera geometry, flattened for JSON. Matrices are written
  /// **row-major** — simd stores them by column, and a row-major dump is
  /// what reads naturally next to the pinhole algebra in
  /// `PlaneRectification` and in the offline tools that re-run it.
  private struct GeometryInfo: Codable {
    /// Camera → world, 16 entries.
    let cameraTransform: [Float]
    /// Intrinsics for the **upright** still (not the sensor's landscape
    /// buffer), 9 entries.
    let intrinsics: [Float]
    /// The upright still's pixel size, which is the space `intrinsics`
    /// describes.
    let imageWidth: Double
    let imageHeight: Double
    /// World y of the puzzle's surface, shared by every shot in the burst.
    let planeHeight: Float

    init(_ geometry: PlaneCaptureGeometry) {
      let transform = geometry.cameraTransform
      cameraTransform = (0..<4).flatMap { row in
        [
          transform.columns.0[row], transform.columns.1[row],
          transform.columns.2[row], transform.columns.3[row],
        ]
      }
      let matrix = geometry.intrinsics
      intrinsics = (0..<3).flatMap { row in
        [matrix.columns.0[row], matrix.columns.1[row], matrix.columns.2[row]]
      }
      imageWidth = Double(geometry.imageSize.width)
      imageHeight = Double(geometry.imageSize.height)
      planeHeight = geometry.planeHeight
    }
  }

  private struct Metadata: Codable {
    let timestamp: String
    let images: [ImageInfo]
    /// Mirrors `GlareFreeComposer.compose`'s `expectedShifts` verbatim —
    /// null entries preserved, so a missing seed hypothesis for a given
    /// corner is visible rather than silently collapsed to zero.
    let expectedShifts: [CGSize?]?
    /// The burst's camera geometry, reference first then one per corner
    /// shot, in `images` order. Null entries (and a null array) mean the
    /// composite was *not* plane-rectified — the Simulator path, or a
    /// device burst that never found the surface — which is exactly what an
    /// offline re-run needs to know before blaming the geometry.
    let geometries: [GeometryInfo?]?
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
  /// Not actor-isolated, but don't rely on that alone to keep the encode-
  /// and-write work off the main actor: where a nonisolated `async`
  /// function's body runs is a language-version detail (SE-0338 hops it to
  /// the cooperative pool today; Swift 6.2's `nonisolated(nonsending)`
  /// direction runs it on the caller's actor), and this function has no
  /// suspension points of its own. Call it from an unstructured background
  /// `Task` (as the capture controller does) rather than awaiting it inline
  /// from `@MainActor` code — the capture flow shouldn't stall on disk I/O
  /// either way.
  ///
  /// Returns the dump directory on success, so tests can locate the files
  /// without duplicating the timestamp formatting; `nil` in Release builds
  /// or when a write failed partway through.
  @discardableResult
  static func record(
    reference: UIImage,
    others: [UIImage],
    expectedShifts: [CGSize?]?,
    geometries: [PlaneCaptureGeometry?]? = nil,
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
        geometries: geometries?.map { $0.map(GeometryInfo.init) },
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
