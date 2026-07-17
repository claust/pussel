import UIKit
import XCTest

@testable import Pussel

/// Round-trip tests for the on-device puzzle persistence (PuzzleStore).
/// Each test uses a unique puzzle id and deletes it afterwards so the shared
/// Documents directory is left clean.
@MainActor
final class PuzzleStoreTests: XCTestCase {
  /// A real (decodable) JPEG so thumbnail generation exercises ImageIO.
  /// `color` distinguishes one image's bytes from another's, so a test can tell
  /// which file it got back.
  private func tinyJPEG(_ color: UIColor = .systemBlue) -> Data {
    let renderer = UIGraphicsImageRenderer(size: CGSize(width: 8, height: 8))
    return renderer.image { context in
      color.setFill()
      context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
    }.jpegData(compressionQuality: 0.9)!
  }

  // Mirrors PuzzleStore's on-disk layout so tests can assert against the
  // actual files (the store's URL helpers are private).
  private var puzzlesRoot: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Puzzles", isDirectory: true)
  }

  private func puzzleDir(_ id: UUID) -> URL {
    puzzlesRoot.appendingPathComponent(id.uuidString, isDirectory: true)
  }

  private func uploadFile(_ puzzle: UUID, _ piece: UUID) -> URL {
    puzzleDir(puzzle).appendingPathComponent("pieces/\(piece.uuidString)-upload.jpg")
  }

  private func displayFile(_ puzzle: UUID) -> URL {
    puzzleDir(puzzle).appendingPathComponent("display.jpg")
  }

  private func makeSession(
    store: PuzzleStore, name: String = "Test", displayJPEG: Data? = nil
  ) -> SolveSession {
    SolveSession(
      id: UUID(),
      name: name,
      puzzleId: "server-123",
      trimmedJPEG: tinyJPEG(),
      displayJPEG: displayJPEG,
      targetPieceCount: 48,
      rows: 6,
      cols: 8,
      store: store
    )
  }

  /// An APIClient wired to `StubURLProtocol` (defined in APIClientTests) so
  /// `remove(id:api:)`'s geometry un-enroll never touches the network. Tests
  /// that care about the request assert on `StubURLProtocol.receivedRequests`.
  private func stubAPI() -> APIClient {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubURLProtocol.self]
    return APIClient(
      baseURL: URL(string: "http://stub.local")!,
      session: URLSession(configuration: configuration),
      authStore: AuthStore()
    )
  }

  // MARK: Zoom copy (display.jpg)

  func testZoomCopyIsPersistedAndReloaded() throws {
    let store = PuzzleStore()
    // Real JPEG bytes, as the zoom viewer decodes these for display — and a
    // different colour from the trimmed image the session is built with, so
    // the assertions below can tell the two files apart rather than passing on
    // a fallback.
    let displayJPEG = tinyJPEG(.systemRed)
    let session = makeSession(store: store, displayJPEG: displayJPEG)
    XCTAssertNotEqual(displayJPEG, session.trimmedJPEG)
    addTeardownBlock { @MainActor in store.delete(session.id) }

    session.persist()
    XCTAssertTrue(FileManager.default.fileExists(atPath: displayFile(session.id).path))

    let reloaded = try XCTUnwrap(store.loadSession(id: session.id))
    // Byte-for-byte: the copy is stored as given, never re-encoded.
    XCTAssertEqual(reloaded.displayJPEG, displayJPEG)
    XCTAssertNotNil(UIImage(data: try XCTUnwrap(reloaded.displayJPEG)))
    // The viewer draws the sharp copy when there is one.
    XCTAssertEqual(reloaded.zoomJPEG, displayJPEG)
  }

  func testPuzzleWithoutZoomCopyReloadsAndFallsBackToTrimmed() throws {
    // The compatibility path for every puzzle saved before the zoom copy
    // existed, and for a capture whose re-warp failed: no display.jpg on disk,
    // and the viewer falls back to the trimmed image rather than showing
    // nothing.
    let store = PuzzleStore()
    let session = makeSession(store: store, displayJPEG: nil)
    addTeardownBlock { @MainActor in store.delete(session.id) }

    session.persist()
    XCTAssertFalse(FileManager.default.fileExists(atPath: displayFile(session.id).path))

    let reloaded = try XCTUnwrap(store.loadSession(id: session.id))
    XCTAssertNil(reloaded.displayJPEG)
    XCTAssertEqual(reloaded.zoomJPEG, reloaded.trimmedJPEG)
  }

  func testSavePersistsSummaryAndSurvivesReload() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store, name: "Frosty field")
    addTeardownBlock { @MainActor in store.delete(session.id) }

    session.entries = [
      CaptureEntry(
        id: UUID(),
        uploadJPEG: Data("piece-upload".utf8),
        displayImage: Data("piece-display".utf8),
        status: .done,
        result: PieceResponse(
          position: NormalizedPoint(x: 0.34, y: 0.4),
          positionConfidence: 0.77,
          rotation: 90,
          rotationConfidence: 0.77,
          cleanedImage: nil,
          pieceSpan: PieceSpan(width: 0.34, height: 0.25)
        ),
        scanPieceId: "p001"
      )
    ]
    session.persist()

    let summary = try XCTUnwrap(store.puzzles.first { $0.id == session.id })
    XCTAssertEqual(summary.name, "Frosty field")
    XCTAssertEqual(summary.pieceCount, 1)
    XCTAssertEqual(summary.placedCount, 1)
    XCTAssertEqual(summary.targetPieceCount, 48)
    XCTAssertEqual(summary.rows, 6)
    XCTAssertEqual(summary.cols, 8)
    XCTAssertNotNil(summary.thumbnail)

    // A brand-new store instance reads purely from disk.
    let reloaded = try XCTUnwrap(PuzzleStore().loadSession(id: session.id))
    XCTAssertEqual(reloaded.name, "Frosty field")
    XCTAssertEqual(reloaded.puzzleId, "server-123")
    XCTAssertEqual(reloaded.trimmedJPEG, session.trimmedJPEG)
    XCTAssertEqual(reloaded.targetPieceCount, 48)
    XCTAssertEqual(reloaded.rows, 6)
    XCTAssertEqual(reloaded.cols, 8)
    XCTAssertEqual(reloaded.entries.count, 1)
    let entry = try XCTUnwrap(reloaded.entries.first)
    XCTAssertEqual(entry.status, .done)
    XCTAssertEqual(entry.displayImage, Data("piece-display".utf8))
    XCTAssertEqual(entry.result?.position, NormalizedPoint(x: 0.34, y: 0.4))
    XCTAssertEqual(entry.result?.rotation, 90)
    XCTAssertEqual(entry.result?.pieceSpan, PieceSpan(width: 0.34, height: 0.25))
    // The scan-and-lock link survives the round trip, so the scan gallery
    // can restore this entry's photo as its thumbnail on a later visit.
    XCTAssertEqual(entry.scanPieceId, "p001")
  }

  func testUnpredictedPieceReloadsAsQueued() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }

    // Captured but never predicted: no separate display file, no result.
    let raw = Data("raw-capture".utf8)
    session.entries = [
      CaptureEntry(id: UUID(), uploadJPEG: raw, displayImage: raw, status: .queued, result: nil)
    ]
    session.persist()

    let reloaded = try XCTUnwrap(PuzzleStore().loadSession(id: session.id))
    let entry = try XCTUnwrap(reloaded.entries.first)
    XCTAssertEqual(entry.status, .queued)
    XCTAssertNil(entry.result)
    // Display falls back to the upload bytes when no cleaned image was stored.
    XCTAssertEqual(entry.displayImage, raw)
  }

  func testRemovedPieceIsPrunedFromDisk() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }

    let keep = CaptureEntry(
      id: UUID(), uploadJPEG: Data("keep".utf8), displayImage: Data("keep".utf8), status: .queued,
      result: nil)
    let drop = CaptureEntry(
      id: UUID(), uploadJPEG: Data("drop".utf8), displayImage: Data("drop".utf8), status: .queued,
      result: nil)
    session.entries = [keep, drop]
    session.persist()

    let fileManager = FileManager.default
    XCTAssertTrue(fileManager.fileExists(atPath: uploadFile(session.id, drop.id).path))

    session.remove(id: drop.id, api: stubAPI())

    // The dropped piece's image file is pruned; the kept one remains.
    XCTAssertFalse(fileManager.fileExists(atPath: uploadFile(session.id, drop.id).path))
    XCTAssertTrue(fileManager.fileExists(atPath: uploadFile(session.id, keep.id).path))

    let reloaded = try XCTUnwrap(PuzzleStore().loadSession(id: session.id))
    XCTAssertEqual(reloaded.entries.map(\.id), [keep.id])
  }

  func testRemovingScannedPieceUnenrollsItFromTheGeometryStore() async throws {
    // Without this DELETE, the scanner's gallery pre-fill (loadEnrolled) reads
    // the piece back off the server on the next visit and shows it as a
    // thumbnail-less tile, because its local photo is gone.
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }

    StubURLProtocol.receivedRequests = []
    StubURLProtocol.handler = { _ in (204, Data()) }
    defer { StubURLProtocol.handler = nil }

    let scanned = CaptureEntry(
      id: UUID(), uploadJPEG: Data("scanned".utf8), displayImage: Data("scanned".utf8),
      status: .queued, result: nil, scanPieceId: "p007")
    session.entries = [scanned]
    session.persist()

    session.remove(id: scanned.id, api: stubAPI())

    // The un-enroll is fire-and-forget — remove() spawns the request in an
    // unawaited Task — so wait for it rather than assuming it landed by the
    // time remove() returned.
    try await waitForRequest(matching: "/api/v1/puzzle/server-123/piece/geometry/p007")
    XCTAssertEqual(StubURLProtocol.receivedRequests.last?.httpMethod, "DELETE")
  }

  func testRemovingUnscannedPieceSkipsTheGeometryDelete() async throws {
    // A hand-captured piece was never enrolled, so there is nothing to
    // un-enroll — and no piece id that a DELETE could safely name.
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }

    StubURLProtocol.receivedRequests = []
    StubURLProtocol.handler = { _ in (204, Data()) }
    defer { StubURLProtocol.handler = nil }

    let entry = CaptureEntry(
      id: UUID(), uploadJPEG: Data("manual".utf8), displayImage: Data("manual".utf8),
      status: .queued, result: nil, scanPieceId: nil)
    session.entries = [entry]
    session.persist()

    session.remove(id: entry.id, api: stubAPI())

    // Give any (incorrectly) spawned request a chance to land before asserting
    // that none did.
    try await Task.sleep(nanoseconds: 100_000_000)
    XCTAssertTrue(session.entries.isEmpty)
    XCTAssertTrue(StubURLProtocol.receivedRequests.isEmpty)
  }

  /// Polls until a request for `path` shows up, or fails the test after ~2s.
  private func waitForRequest(matching path: String) async throws {
    for _ in 0..<40 {
      if StubURLProtocol.receivedRequests.contains(where: { $0.url?.path == path }) { return }
      try await Task.sleep(nanoseconds: 50_000_000)
    }
    XCTFail("No request for \(path) within the timeout")
  }

  func testDeleteRemovesPuzzle() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store)
    // Clean up even if an assertion fails before the delete below.
    addTeardownBlock { @MainActor in store.delete(session.id) }
    session.persist()
    XCTAssertTrue(store.puzzles.contains { $0.id == session.id })
    XCTAssertTrue(FileManager.default.fileExists(atPath: puzzleDir(session.id).path))

    store.delete(session.id)
    XCTAssertFalse(store.puzzles.contains { $0.id == session.id })
    XCTAssertNil(store.loadSession(id: session.id))
    XCTAssertFalse(FileManager.default.fileExists(atPath: puzzleDir(session.id).path))
  }

  func testDeleteWithUndoHidesPuzzleButKeepsFilesDuringWindow() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }
    session.persist()

    store.deleteWithUndo(session.id)

    // Gone from the list, still whole on disk — the point of the undo window.
    XCTAssertFalse(store.puzzles.contains { $0.id == session.id })
    XCTAssertEqual(store.pendingDelete?.id, session.id)
    XCTAssertTrue(FileManager.default.fileExists(atPath: puzzleDir(session.id).path))

    // A rescan must not resurrect the row just because the folder is there.
    store.refresh()
    XCTAssertFalse(store.puzzles.contains { $0.id == session.id })
  }

  func testUndoDeleteRestoresPuzzle() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }
    session.persist()

    store.deleteWithUndo(session.id)
    store.undoDelete()

    XCTAssertNil(store.pendingDelete)
    XCTAssertTrue(store.puzzles.contains { $0.id == session.id })
    XCTAssertNotNil(store.loadSession(id: session.id))
    XCTAssertTrue(FileManager.default.fileExists(atPath: puzzleDir(session.id).path))
  }

  func testCommitPendingDeleteRemovesFilesForGood() throws {
    let store = PuzzleStore()
    let session = makeSession(store: store)
    addTeardownBlock { @MainActor in store.delete(session.id) }
    session.persist()

    store.deleteWithUndo(session.id)
    store.commitPendingDelete()

    XCTAssertNil(store.pendingDelete)
    XCTAssertFalse(store.puzzles.contains { $0.id == session.id })
    XCTAssertFalse(FileManager.default.fileExists(atPath: puzzleDir(session.id).path))

    // Undoing after the window has closed must not resurrect a deleted row.
    store.undoDelete()
    XCTAssertFalse(store.puzzles.contains { $0.id == session.id })
  }

  func testSecondDeleteCommitsTheFirst() throws {
    let store = PuzzleStore()
    let first = makeSession(store: store, name: "First")
    let second = makeSession(store: store, name: "Second")
    addTeardownBlock { @MainActor in
      store.delete(first.id)
      store.delete(second.id)
    }
    first.persist()
    second.persist()

    store.deleteWithUndo(first.id)
    store.deleteWithUndo(second.id)

    // Only the newest delete is undoable; the first is already gone for good,
    // otherwise its files would linger with no way left to reach them.
    XCTAssertEqual(store.pendingDelete?.id, second.id)
    XCTAssertFalse(FileManager.default.fileExists(atPath: puzzleDir(first.id).path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: puzzleDir(second.id).path))

    store.undoDelete()
    XCTAssertTrue(store.puzzles.contains { $0.id == second.id })
    XCTAssertFalse(store.puzzles.contains { $0.id == first.id })
  }
}
