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
  /// Total piece count entered by the user for this puzzle (distinct from
  /// `pieceCount`, which counts captured pieces).
  let targetPieceCount: Int
  let rows: Int
  let cols: Int
  /// Bytes of the small downsampled thumbnail (thumb.jpg), used as the card
  /// image — not the full-size trimmed puzzle JPEG.
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
  /// Total piece count entered by the user, and the grid estimated from it.
  var pieceCount: Int
  var rows: Int
  var cols: Int
}

private struct StoredPiece: Codable {
  let id: UUID
  /// nil while the piece is captured but not yet predicted.
  var result: StoredResult?
  /// Geometry-store piece id for M10 scan-and-lock captures; nil for
  /// shutter/library captures and for manifests written before this field
  /// existed (optional Codable decodes missing keys as nil).
  var scanPieceId: String?
}

private struct StoredResult: Codable {
  let position: NormalizedPoint
  let positionConfidence: Double
  let rotation: Int
  let rotationConfidence: Double
  let pieceSpan: PieceSpan?
}

/// Local, on-device persistence for solved/in-progress puzzles. Everything is
/// kept under `Documents/Puzzles/<uuid>/` — one folder per puzzle with a
/// `manifest.json`, the `trimmed.jpg` picture, an optional zoom-quality
/// `display.jpg` of the same crop, and a `pieces/` directory of
/// `<pieceId>-upload.jpg` (and optional `<pieceId>-display.jpg`) files. No
/// server storage: reopening a puzzle rehydrates purely from disk.
@Observable
@MainActor
final class PuzzleStore {
  /// Home-screen rows, newest activity first. Refreshed after every mutation.
  /// Excludes `pendingDelete` — a puzzle waiting out its undo window is gone
  /// from the list even though its files are still on disk.
  private(set) var puzzles: [PuzzleSummary] = []

  /// The puzzle hidden by the most recent `deleteWithUndo`, if its undo window
  /// is still open. Drives the undo snackbar.
  private(set) var pendingDelete: PuzzleSummary?

  /// How long the user has to undo before the files actually go.
  static let undoWindow: Duration = .seconds(5)

  @ObservationIgnored private var purgeTask: Task<Void, Never>?
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
    // The zoom copy is optional and immutable, so it's written once and its
    // absence is a valid state (puzzles from before it was kept, and captures
    // whose re-warp failed) — loadSession falls back to the trimmed image.
    let display = puzzleDisplayURL(session.id)
    if let displayJPEG = session.displayJPEG, !fileManager.fileExists(atPath: display.path) {
      try? displayJPEG.write(to: display, options: .atomic)
    }
    // The trimmed image is immutable, so its downsampled thumbnail is
    // generated once and reused for the home-screen list.
    let thumbnailData = loadOrCreateThumbnail(for: session.id, from: session.trimmedJPEG)

    // Only entries whose upload image is actually on disk go into the
    // manifest, so a failed best-effort write can't leave the manifest
    // referencing a missing file (which loadSession would then drop,
    // diverging from the summary counts). A dropped entry is retried on the
    // next persist(). keep set drives orphan pruning.
    var keep = Set<String>()
    var persistedEntries: [CaptureEntry] = []
    for entry in session.entries {
      let upload = uploadURL(session.id, entry.id)
      if !fileManager.fileExists(atPath: upload.path) {
        try? entry.uploadJPEG.write(to: upload, options: .atomic)
      }
      // Skip entries whose upload image didn't make it to disk; they'll be
      // retried on the next persist() rather than left dangling.
      guard fileManager.fileExists(atPath: upload.path) else { continue }
      keep.insert(entry.id.uuidString)

      let display = displayURL(session.id, entry.id)
      // Only store a separate display file when the cleaned image actually
      // differs from the raw capture, otherwise the bytes are duplicated.
      // The cleaned image is immutable once predicted, so skip the rewrite
      // if it already exists (avoids rewriting every piece on each persist).
      if entry.displayImage != entry.uploadJPEG, !fileManager.fileExists(atPath: display.path) {
        try? entry.displayImage.write(to: display, options: .atomic)
      }
      persistedEntries.append(entry)
    }
    pruneOrphanPieceFiles(in: pieces, keeping: keep)

    let now = Date()
    let manifest = PuzzleManifest(
      id: session.id,
      serverPuzzleId: session.puzzleId,
      name: session.name,
      createdAt: session.createdAt,
      updatedAt: now,
      pieces: persistedEntries.map { entry in
        StoredPiece(
          id: entry.id,
          result: entry.result.map {
            StoredResult(
              position: $0.position,
              positionConfidence: $0.positionConfidence,
              rotation: $0.rotation,
              rotationConfidence: $0.rotationConfidence,
              pieceSpan: $0.pieceSpan
            )
          },
          scanPieceId: entry.scanPieceId
        )
      },
      pieceCount: session.targetPieceCount,
      rows: session.rows,
      cols: session.cols
    )
    do {
      let data = try encoder.encode(manifest)
      try data.write(to: manifestURL(session.id), options: .atomic)
    } catch {
      // Persistence is the whole point — don't advertise a puzzle in the
      // list that won't survive a relaunch. Surface loudly in Debug and
      // leave the in-memory list untouched.
      assertionFailure("Failed to persist puzzle \(session.id): \(error)")
      return
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
        // Counts reflect what was persisted, matching the manifest.
        pieceCount: persistedEntries.count,
        placedCount: persistedEntries.filter { $0.result != nil }.count,
        targetPieceCount: session.targetPieceCount,
        rows: session.rows,
        cols: session.cols,
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
      let trimmed = try? Data(contentsOf: trimmedURL(id))
    else {
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
      displayJPEG: try? Data(contentsOf: puzzleDisplayURL(id)),
      targetPieceCount: manifest.pieceCount,
      rows: manifest.rows,
      cols: manifest.cols,
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
          cleanedImage: nil,
          pieceSpan: $0.pieceSpan
        )
      }
      return CaptureEntry(
        id: piece.id,
        uploadJPEG: upload,
        displayImage: display,
        status: result == nil ? .queued : .done,
        result: result,
        scanPieceId: piece.scanPieceId
      )
    }
    return session
  }

  /// Permanently removes a puzzle and all of its images from disk.
  func delete(_ id: UUID) {
    try? fileManager.removeItem(at: puzzleDir(id))
    refresh()
  }

  /// Hides a puzzle from the list and starts its undo window. Nothing is
  /// touched on disk until the window closes, so quitting or crashing mid-undo
  /// leaves the puzzle intact rather than half-deleted.
  func deleteWithUndo(_ id: UUID) {
    // Only one undo is offered at a time, so a second delete finishes the
    // first — otherwise its files would linger with no way left to reach them.
    commitPendingDelete()
    guard let summary = puzzles.first(where: { $0.id == id }) else { return }
    pendingDelete = summary
    puzzles.removeAll { $0.id == id }
    purgeTask = Task { [weak self] in
      try? await Task.sleep(for: Self.undoWindow)
      guard !Task.isCancelled else { return }
      self?.commitPendingDelete()
    }
  }

  /// Puts the pending puzzle back in the list and calls off the purge.
  func undoDelete() {
    purgeTask?.cancel()
    purgeTask = nil
    guard let summary = pendingDelete else { return }
    pendingDelete = nil
    upsert(summary)
  }

  /// Ends the undo window early and does the real delete. A no-op when nothing
  /// is pending, so it's safe to call from anywhere that supersedes the undo.
  func commitPendingDelete() {
    purgeTask?.cancel()
    purgeTask = nil
    guard let pending = pendingDelete else { return }
    // Cleared first: `delete` calls `refresh`, which would otherwise see the
    // folder mid-removal and count it as still-pending.
    pendingDelete = nil
    delete(pending.id)
  }

  /// Rebuilds `puzzles` by scanning the store directory. A puzzle awaiting undo
  /// is still on disk, so it has to be filtered back out or it reappears.
  func refresh() {
    let dirs =
      (try? fileManager.contentsOfDirectory(
        at: rootURL,
        includingPropertiesForKeys: [.isDirectoryKey]
      )) ?? []
    puzzles = dirs.compactMap { dir -> PuzzleSummary? in
      guard dir.lastPathComponent != pendingDelete?.id.uuidString else { return nil }
      // The folder name is the canonical id (matching loadSession); use it
      // for the summary and all file lookups rather than manifest.id, which
      // could drift if a manifest was copied or moved.
      guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true,
        let id = UUID(uuidString: dir.lastPathComponent),
        let data = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
        let manifest = try? decoder.decode(PuzzleManifest.self, from: data)
      else {
        return nil
      }
      // Prefer the small cached thumbnail; generate it lazily from the
      // trimmed image if this puzzle predates thumbnail caching.
      let thumbnail =
        (try? Data(contentsOf: thumbnailURL(id)))
        ?? (try? Data(contentsOf: trimmedURL(id))).flatMap { trimmed in
          loadOrCreateThumbnail(for: id, from: trimmed)
        }
      return PuzzleSummary(
        id: id,
        name: manifest.name,
        createdAt: manifest.createdAt,
        updatedAt: manifest.updatedAt,
        pieceCount: manifest.pieces.count,
        placedCount: manifest.pieces.filter { $0.result != nil }.count,
        targetPieceCount: manifest.pieceCount,
        rows: manifest.rows,
        cols: manifest.cols,
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

  /// The puzzle's optional zoom-quality overview (`SolveSession.displayJPEG`),
  /// distinct from a piece's `<pieceId>-display.jpg` under `pieces/`.
  private func puzzleDisplayURL(_ id: UUID) -> URL {
    puzzleDir(id).appendingPathComponent("display.jpg")
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
    let files =
      (try? fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
    for file in files {
      // Filenames are "<uuid>-upload.jpg" / "<uuid>-display.jpg".
      let id = String(file.lastPathComponent.prefix(36))
      if !keep.contains(id) {
        try? fileManager.removeItem(at: file)
      }
    }
  }
}
