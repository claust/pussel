import Foundation
import Observation

/// A lightweight row for the home-screen list — enough to render a card
/// without loading every piece image of every stored puzzle.
struct PuzzleSummary: Identifiable, Equatable {
    let id: UUID
    let name: String
    let createdAt: Date
    let updatedAt: Date
    let pieceCount: Int
    let placedCount: Int
    /// Bytes of the trimmed puzzle image, used as the card thumbnail.
    let thumbnail: Data?
}

/// On-disk manifest for one puzzle. Image bytes live in sibling files, so the
/// JSON stays small; the piece's `cleanedImage` base64 is never persisted here
/// (the display image is stored as a `<id>-display.jpg` file instead).
private struct PuzzleManifest: Codable {
    let id: UUID
    var serverPuzzleId: String
    var name: String
    let createdAt: Date
    var updatedAt: Date
    var pieces: [StoredPiece]
}

private struct StoredPiece: Codable {
    let id: UUID
    /// nil while the piece is captured but not yet predicted.
    var result: StoredResult?
}

private struct StoredResult: Codable {
    let position: NormalizedPoint
    let positionConfidence: Double
    let rotation: Int
    let rotationConfidence: Double
}

/// Local, on-device persistence for solved/in-progress puzzles. Everything is
/// kept under `Documents/Puzzles/<uuid>/` — one folder per puzzle with a
/// `manifest.json`, the `trimmed.jpg` picture, and a `pieces/` directory of
/// `<pieceId>-upload.jpg` (and optional `<pieceId>-display.jpg`) files. No
/// server storage: reopening a puzzle rehydrates purely from disk.
@Observable
@MainActor
final class PuzzleStore {
    /// Home-screen rows, newest activity first. Refreshed after every mutation.
    private(set) var puzzles: [PuzzleSummary] = []

    @ObservationIgnored private let fileManager = FileManager.default
    @ObservationIgnored private let encoder: JSONEncoder
    @ObservationIgnored private let decoder: JSONDecoder

    init() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.encoder = encoder
        self.decoder = decoder
        try? fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        refresh()
    }

    // MARK: Public API

    /// Persists the current state of a session: writes any missing image files,
    /// prunes files for removed pieces, and rewrites the manifest. Idempotent —
    /// safe to call on every change (enqueue, prediction done, remove, reupload).
    func save(_ session: SolveSession) {
        let pieces = piecesDir(session.id)
        // Creating the pieces dir with intermediates also creates its parent
        // puzzle dir, so no separate puzzle-dir creation is needed.
        try? fileManager.createDirectory(at: pieces, withIntermediateDirectories: true)

        let trimmed = trimmedURL(session.id)
        if !fileManager.fileExists(atPath: trimmed.path) {
            try? session.trimmedJPEG.write(to: trimmed, options: .atomic)
        }
        // The trimmed image is immutable, so its downsampled thumbnail is
        // generated once and reused for the home-screen list.
        let thumbnailData = loadOrCreateThumbnail(for: session.id, from: session.trimmedJPEG)

        var keep = Set<String>()
        for entry in session.entries {
            keep.insert(entry.id.uuidString)
            let upload = uploadURL(session.id, entry.id)
            if !fileManager.fileExists(atPath: upload.path) {
                try? entry.uploadJPEG.write(to: upload, options: .atomic)
            }
            let display = displayURL(session.id, entry.id)
            // Only store a separate display file when the cleaned image actually
            // differs from the raw capture, otherwise the bytes are duplicated.
            if entry.displayImage != entry.uploadJPEG {
                try? entry.displayImage.write(to: display, options: .atomic)
            }
        }
        pruneOrphanPieceFiles(in: pieces, keeping: keep)

        let now = Date()
        let manifest = PuzzleManifest(
            id: session.id,
            serverPuzzleId: session.puzzleId,
            name: session.name,
            createdAt: session.createdAt,
            updatedAt: now,
            pieces: session.entries.map { entry in
                StoredPiece(
                    id: entry.id,
                    result: entry.result.map {
                        StoredResult(
                            position: $0.position,
                            positionConfidence: $0.positionConfidence,
                            rotation: $0.rotation,
                            rotationConfidence: $0.rotationConfidence
                        )
                    }
                )
            }
        )
        if let data = try? encoder.encode(manifest) {
            try? data.write(to: manifestURL(session.id), options: .atomic)
        }

        // Update just this puzzle's summary from in-memory state rather than
        // rescanning and re-decoding the whole directory on every persist()
        // (which runs on enqueue/prediction/remove/reupload).
        upsert(
            PuzzleSummary(
                id: session.id,
                name: session.name,
                createdAt: session.createdAt,
                updatedAt: now,
                pieceCount: session.entries.count,
                placedCount: session.entries.filter { $0.result != nil }.count,
                thumbnail: thumbnailData
            )
        )
    }

    /// Inserts or replaces a summary in the in-memory list, keeping it sorted
    /// by most-recent activity.
    private func upsert(_ summary: PuzzleSummary) {
        puzzles.removeAll { $0.id == summary.id }
        puzzles.append(summary)
        puzzles.sort { $0.updatedAt > $1.updatedAt }
    }

    /// Returns the cached thumbnail bytes for a puzzle, generating and caching
    /// them from the trimmed image on first use. Falls back to nil (the row
    /// then shows a placeholder) rather than the full-size image.
    private func loadOrCreateThumbnail(for id: UUID, from trimmedJPEG: Data) -> Data? {
        let url = thumbnailURL(id)
        if let existing = try? Data(contentsOf: url) {
            return existing
        }
        guard let generated = ImageUtilities.thumbnailJPEG(from: trimmedJPEG) else { return nil }
        try? generated.write(to: url, options: .atomic)
        return generated
    }

    /// Rehydrates a full working session from disk, or nil if the folder is
    /// missing/corrupt. Pieces with a stored result come back `.done`; pieces
    /// captured but never predicted come back `.queued` so the solve view can
    /// resume them.
    func loadSession(id: UUID) -> SolveSession? {
        guard let data = try? Data(contentsOf: manifestURL(id)),
              let manifest = try? decoder.decode(PuzzleManifest.self, from: data),
              let trimmed = try? Data(contentsOf: trimmedURL(id)) else {
            return nil
        }
        // The folder name is the canonical identity — use it (not manifest.id)
        // so persist() always writes back to the directory we loaded from,
        // even if the manifest was copied or moved.
        let session = SolveSession(
            id: id,
            name: manifest.name,
            puzzleId: manifest.serverPuzzleId,
            trimmedJPEG: trimmed,
            createdAt: manifest.createdAt,
            store: self
        )
        session.entries = manifest.pieces.compactMap { piece in
            guard let upload = try? Data(contentsOf: uploadURL(id, piece.id)) else { return nil }
            let display = (try? Data(contentsOf: displayURL(id, piece.id))) ?? upload
            let result = piece.result.map {
                PieceResponse(
                    position: $0.position,
                    positionConfidence: $0.positionConfidence,
                    rotation: $0.rotation,
                    rotationConfidence: $0.rotationConfidence,
                    cleanedImage: nil
                )
            }
            return CaptureEntry(
                id: piece.id,
                uploadJPEG: upload,
                displayImage: display,
                status: result == nil ? .queued : .done,
                result: result
            )
        }
        return session
    }

    /// Permanently removes a puzzle and all of its images from disk.
    func delete(_ id: UUID) {
        try? fileManager.removeItem(at: puzzleDir(id))
        refresh()
    }

    /// Rebuilds `puzzles` by scanning the store directory.
    func refresh() {
        let dirs = (try? fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey]
        )) ?? []
        puzzles = dirs.compactMap { dir -> PuzzleSummary? in
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true,
                  let data = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
                  let manifest = try? decoder.decode(PuzzleManifest.self, from: data) else {
                return nil
            }
            // Prefer the small cached thumbnail; generate it lazily from the
            // trimmed image if this puzzle predates thumbnail caching.
            let thumbnail = (try? Data(contentsOf: thumbnailURL(manifest.id)))
                ?? (try? Data(contentsOf: trimmedURL(manifest.id))).flatMap { trimmed in
                    loadOrCreateThumbnail(for: manifest.id, from: trimmed)
                }
            return PuzzleSummary(
                id: manifest.id,
                name: manifest.name,
                createdAt: manifest.createdAt,
                updatedAt: manifest.updatedAt,
                pieceCount: manifest.pieces.count,
                placedCount: manifest.pieces.filter { $0.result != nil }.count,
                thumbnail: thumbnail
            )
        }
        .sorted { $0.updatedAt > $1.updatedAt }
    }

    // MARK: File layout

    private var rootURL: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Puzzles", isDirectory: true)
    }

    private func puzzleDir(_ id: UUID) -> URL {
        rootURL.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func piecesDir(_ id: UUID) -> URL {
        puzzleDir(id).appendingPathComponent("pieces", isDirectory: true)
    }

    private func trimmedURL(_ id: UUID) -> URL {
        puzzleDir(id).appendingPathComponent("trimmed.jpg")
    }

    private func thumbnailURL(_ id: UUID) -> URL {
        puzzleDir(id).appendingPathComponent("thumb.jpg")
    }

    private func manifestURL(_ id: UUID) -> URL {
        puzzleDir(id).appendingPathComponent("manifest.json")
    }

    private func uploadURL(_ puzzle: UUID, _ piece: UUID) -> URL {
        piecesDir(puzzle).appendingPathComponent("\(piece.uuidString)-upload.jpg")
    }

    private func displayURL(_ puzzle: UUID, _ piece: UUID) -> URL {
        piecesDir(puzzle).appendingPathComponent("\(piece.uuidString)-display.jpg")
    }

    /// Deletes piece image files whose id is no longer in the session.
    private func pruneOrphanPieceFiles(in dir: URL, keeping keep: Set<String>) {
        let files = (try? fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        for file in files {
            // Filenames are "<uuid>-upload.jpg" / "<uuid>-display.jpg".
            let id = String(file.lastPathComponent.prefix(36))
            if !keep.contains(id) {
                try? fileManager.removeItem(at: file)
            }
        }
    }
}
