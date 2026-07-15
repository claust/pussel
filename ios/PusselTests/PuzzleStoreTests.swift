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

    private func makeSession(store: PuzzleStore, name: String = "Test") -> SolveSession {
        SolveSession(
            id: UUID(),
            name: name,
            puzzleId: "server-123",
            trimmedJPEG: tinyJPEG(),
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
                    cleanedImage: nil
                )
            )
        ]
        session.persist()

        let summary = try XCTUnwrap(store.puzzles.first { $0.id == session.id })
        XCTAssertEqual(summary.name, "Frosty field")
        XCTAssertEqual(summary.pieceCount, 1)
        XCTAssertEqual(summary.placedCount, 1)
        XCTAssertNotNil(summary.thumbnail)

        // A brand-new store instance reads purely from disk.
        let reloaded = try XCTUnwrap(PuzzleStore().loadSession(id: session.id))
        XCTAssertEqual(reloaded.name, "Frosty field")
        XCTAssertEqual(reloaded.puzzleId, "server-123")
        XCTAssertEqual(reloaded.trimmedJPEG, session.trimmedJPEG)
        XCTAssertEqual(reloaded.entries.count, 1)
        let entry = try XCTUnwrap(reloaded.entries.first)
        XCTAssertEqual(entry.status, .done)
        XCTAssertEqual(entry.displayImage, Data("piece-display".utf8))
        XCTAssertEqual(entry.result?.position, NormalizedPoint(x: 0.34, y: 0.4))
        XCTAssertEqual(entry.result?.rotation, 90)
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

        let keep = CaptureEntry(id: UUID(), uploadJPEG: Data("keep".utf8), displayImage: Data("keep".utf8), status: .queued, result: nil)
        let drop = CaptureEntry(id: UUID(), uploadJPEG: Data("drop".utf8), displayImage: Data("drop".utf8), status: .queued, result: nil)
        session.entries = [keep, drop]
        session.persist()

        session.remove(id: drop.id)

        let reloaded = try XCTUnwrap(PuzzleStore().loadSession(id: session.id))
        XCTAssertEqual(reloaded.entries.map(\.id), [keep.id])
    }

    func testDeleteRemovesPuzzle() throws {
        let store = PuzzleStore()
        let session = makeSession(store: store)
        session.persist()
        XCTAssertTrue(store.puzzles.contains { $0.id == session.id })

        store.delete(session.id)
        XCTAssertFalse(store.puzzles.contains { $0.id == session.id })
        XCTAssertNil(store.loadSession(id: session.id))
    }
}
