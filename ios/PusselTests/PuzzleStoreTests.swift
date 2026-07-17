import UIKit
import XCTest

@testable import Pussel

/// Round-trip tests for the on-device puzzle persistence (PuzzleStore).
/// Each test uses a unique puzzle id and deletes it afterwards so the shared
/// Documents directory is left clean.
@MainActor
final class PuzzleStoreTests: XCTestCase {
  /// A real (decodable) JPEG so thumbnail generation exercises ImageIO.
  private func tinyJPEG() -> Data {
    let renderer = UIGraphicsImageRenderer(size: CGSize(width: 8, height: 8))
    return renderer.image { context in
      UIColor.systemBlue.setFill()
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

  private func makeSession(store: PuzzleStore, name: String = "Test") -> SolveSession {
    SolveSession(
      id: UUID(),
      name: name,
      puzzleId: "server-123",
      trimmedJPEG: tinyJPEG(),
      targetPieceCount: 48,
      rows: 6,
      cols: 8,
      store: store
    )
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

    session.remove(id: drop.id)

    // The dropped piece's image file is pruned; the kept one remains.
    XCTAssertFalse(fileManager.fileExists(atPath: uploadFile(session.id, drop.id).path))
    XCTAssertTrue(fileManager.fileExists(atPath: uploadFile(session.id, keep.id).path))

    let reloaded = try XCTUnwrap(PuzzleStore().loadSession(id: session.id))
    XCTAssertEqual(reloaded.entries.map(\.id), [keep.id])
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
