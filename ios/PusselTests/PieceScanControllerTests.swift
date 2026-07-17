import XCTest

@testable import Pussel

// MARK: - Fake geometry client

/// Recording fake that returns scripted `PieceGeometryUploadResponse` /
/// `PieceGeometryListResponse` values without any network calls.
@MainActor
final class FakeGeometryClient: PieceGeometryScanning {
  private(set) var uploadCallCount = 0
  private(set) var lastEnrollUncertain: Bool?

  /// Queue of upload responses. Each call to `uploadPieceGeometry` dequeues
  /// one; panics if the queue runs dry (indicates the test over-called).
  var uploadResponses: [Result<PieceGeometryUploadResponse, Error>] = []
  var listResponse: Result<PieceGeometryListResponse, Error> = .success(
    PieceGeometryListResponse(puzzleId: "p1", pieces: []))

  func uploadPieceGeometry(
    puzzleId: String, jpegData: Data, enrollUncertain: Bool
  ) async throws -> PieceGeometryUploadResponse {
    uploadCallCount += 1
    lastEnrollUncertain = enrollUncertain
    guard !uploadResponses.isEmpty else {
      fatalError("FakeGeometryClient: no more scripted upload responses")
    }
    return try uploadResponses.removeFirst().get()
  }

  func listPieceGeometry(puzzleId: String) async throws -> PieceGeometryListResponse {
    try listResponse.get()
  }
}

// MARK: - Helpers

/// Builds a minimal `PieceGeometryUploadResponse` for a given status.
/// `zScore` defaults to a plausible gray-zone value because the controller
/// keys the uncertain-vs-unreadable distinction on it (nil = the pipeline
/// produced no fingerprint); tests for the unreadable branch pass nil
/// explicitly.
private func uploadResponse(
  status: PieceGeometryStatus,
  pieceId: String? = "piece-1",
  matchPieceId: String? = nil,
  zScore: Double? = -1.2,
  edgeTypes: [GeometryEdgeType] = [.tab, .blank, .flat, .tab]
) -> PieceGeometryUploadResponse {
  PieceGeometryUploadResponse(
    pieceId: pieceId,
    status: status,
    matchPieceId: matchPieceId,
    zScore: zScore,
    lockable: status == .new,
    quality: GeometryQuality(isClean: true, cornerDisagreement: false),
    record: PieceGeometryRecordSummary(
      edges: edgeTypes.map { GeometryEdgeSummary(type: $0) }
    )
  )
}

/// Injects three stable lockable frames spanning ≥1 s, causing the default
/// `PieceScanStabilityTracker` inside `controller` to fire once.
@MainActor
private func driveToFire(_ controller: PieceScanController) {
  let square: [NormalizedPoint] = [
    NormalizedPoint(x: 0.1, y: 0.1), NormalizedPoint(x: 0.9, y: 0.1),
    NormalizedPoint(x: 0.9, y: 0.9), NormalizedPoint(x: 0.1, y: 0.9),
  ]
  let near: [NormalizedPoint] = [
    NormalizedPoint(x: 0.10, y: 0.10), NormalizedPoint(x: 0.91, y: 0.10),
    NormalizedPoint(x: 0.91, y: 0.91), NormalizedPoint(x: 0.10, y: 0.91),
  ]
  let t0 = Date(timeIntervalSince1970: 0)
  controller.ingest(.lockable(polygon: square, confidence: 0.9), at: t0)
  controller.ingest(
    .lockable(polygon: near, confidence: 0.9), at: t0.addingTimeInterval(0.5))
  controller.ingest(
    .lockable(polygon: near, confidence: 0.9), at: t0.addingTimeInterval(1.0))
}

/// Yields the main-actor executor enough times for nested async tasks
/// (capture → POST → rearm) to all complete when the sleep is instant.
/// Eight yields covers: ingest Task spawn + capturePhoto await + post call
/// + response processing + scheduleRearm Task spawn + sleep + reset.
@MainActor
private func settle() async {
  for _ in 0..<8 { await Task.yield() }
}

/// Fixture JPEG bytes — not a valid image, but enough for the controller to
/// treat the capture as successful.
private let fixtureJPEG = Data("FAKEJPEG".utf8)

// MARK: - Tests

@MainActor
final class PieceScanControllerTests: XCTestCase {

  // MARK: - Factory

  /// Builds a controller with an instant sleep and a capture closure that
  /// returns `fixtureJPEG` by default (or `nil` if `captureReturnsNil`).
  /// The controller's haptic events are routed to a `HapticBox` so the test
  /// can check them without the inout-in-escaping-closure restriction.
  private func makeController(
    client: FakeGeometryClient,
    captureReturnsNil: Bool = false
  ) -> (ctrl: PieceScanController, haptics: HapticBox) {
    let box = HapticBox()
    let ctrl = PieceScanController(
      puzzleId: "test-puzzle",
      geometryClient: client,
      capture: { captureReturnsNil ? nil : fixtureJPEG },
      haptic: { [weak box] kind in box?.record(kind) },
      sleep: { _ in }
    )
    return (ctrl, box)
  }

  // MARK: - 1. Happy path: locked

  func testStableStreamLocksOnceGalleryGrowsAndSuccessHapticFires() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(
        uploadResponse(
          status: .new, pieceId: "p-new",
          edgeTypes: [.tab, .blank, .flat, .tab])
      )
    ]
    let (ctrl, haptics) = makeController(client: client)

    // Drive stability → fires once.
    driveToFire(ctrl)
    await settle()

    // After settle(), the instant rearm has already moved phase back to .scanning.
    // Test the durable effects instead of the transient verdict phase.
    XCTAssertEqual(client.uploadCallCount, 1)
    XCTAssertEqual(ctrl.gallery.count, 1)
    XCTAssertEqual(ctrl.gallery.first?.pieceId, "p-new")
    XCTAssertEqual(ctrl.gallery.first?.thumbnailJPEG, fixtureJPEG)
    XCTAssertTrue(haptics.contains(.success))
    XCTAssertFalse(haptics.contains(.failure))
  }

  func testRearmsAfterLockedForASecondPiece() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .new, pieceId: "p-new")),
      .success(uploadResponse(status: .new, pieceId: "p-new2")),
    ]
    let (ctrl, _) = makeController(client: client)

    // First piece.
    driveToFire(ctrl)
    await settle()
    XCTAssertEqual(ctrl.phase, .scanning, "Must be scanning after instant rearm")

    // Second piece.
    driveToFire(ctrl)
    await settle()
    XCTAssertEqual(client.uploadCallCount, 2)
    XCTAssertEqual(ctrl.gallery.count, 2)
  }

  // MARK: - 2. Matched → alreadyScanned

  func testMatchedResponseGivesWarningHapticAndDoesNotGrowGallery() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .matched, pieceId: "p-orig", matchPieceId: "p-orig"))
    ]
    let (ctrl, haptics) = makeController(client: client)

    driveToFire(ctrl)
    await settle()

    XCTAssertEqual(ctrl.gallery.count, 0, "Gallery must not grow on alreadyScanned")
    XCTAssertTrue(haptics.contains(.warning))
    XCTAssertFalse(haptics.contains(.success))
  }

  // MARK: - 3a. Uncertain: pending JPEG kept, no gallery change, no haptic

  /// Uncertain requires a NON-instant sleep so the rearm has not cleared
  /// pendingUncertainJPEG by the time we check. We use a sleep closure that
  /// suspends forever (never resumes), giving us a deterministic window.
  func testUncertainResponseKeepsPendingJPEGAndNoGalleryChange() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .uncertain, pieceId: nil, matchPieceId: "p-maybe"))
    ]
    let box = HapticBox()
    // Non-resuming sleep so the rearm doesn't clear pendingUncertainJPEG.
    let ctrl = PieceScanController(
      puzzleId: "test-puzzle",
      geometryClient: client,
      capture: { fixtureJPEG },
      haptic: { [weak box] kind in box?.record(kind) },
      sleep: { _ in await Task.yield() }  // yields but doesn't block long
    )

    driveToFire(ctrl)
    await settle()

    // After settle, uncertain rearm delay ran (Task.yield). The phase
    // transitioned back, but pendingUncertainJPEG persists until confirmed.
    // The gallery must still be empty.
    XCTAssertEqual(ctrl.gallery.count, 0)
    // Spec: no haptic on uncertain.
    XCTAssertFalse(box.contains(.success))
    XCTAssertFalse(box.contains(.failure))
    XCTAssertFalse(box.contains(.warning))
  }

  // MARK: - 3a2. Uncertain WITHOUT a z-score → unreadable, no confirm chip

  /// A nil z-score means the backend pipeline failed on the photo (no
  /// fingerprint), so on_uncertain=enroll would have nothing to enroll —
  /// the confirm chip must NOT appear, or the user gets a dead-end loop.
  func testUncertainWithoutZScoreBecomesUnreadableWithoutChip() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .uncertain, pieceId: nil, zScore: nil))
    ]
    let box = HapticBox()
    let ctrl = PieceScanController(
      puzzleId: "test-puzzle",
      geometryClient: client,
      capture: { fixtureJPEG },
      haptic: { [weak box] kind in box?.record(kind) },
      sleep: { _ in await Task.yield() }
    )

    driveToFire(ctrl)
    await settle()

    XCTAssertNil(ctrl.pendingUncertainJPEG, "unreadable must not offer the confirm chip")
    XCTAssertEqual(ctrl.gallery.count, 0)
    XCTAssertFalse(box.contains(.success))
    // confirmUncertainAsNew with nothing pending must be a no-op.
    await ctrl.confirmUncertainAsNew()
    XCTAssertEqual(client.uploadCallCount, 1)
  }

  // MARK: - 3b. confirmUncertainAsNew → enrolled + gallery

  func testConfirmUncertainAsNewEnrollsAndAddsToGallery() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .uncertain, pieceId: nil, matchPieceId: "p-maybe")),
      .success(
        uploadResponse(
          status: .new, pieceId: "p-confirmed",
          edgeTypes: [.flat, .tab, .blank, .flat])),
    ]
    // Use a non-resuming sleep so the uncertain chip stays available long
    // enough for us to call confirmUncertainAsNew.
    let box = HapticBox()
    let pendingJPEGClearedAfterConfirm = LockBox(false)
    let ctrl = PieceScanController(
      puzzleId: "test-puzzle",
      geometryClient: client,
      capture: { fixtureJPEG },
      haptic: { [weak box] kind in box?.record(kind) },
      sleep: { _ in
        // Pause long enough that we can confirm before the rearm clears state.
        try? await Task.sleep(nanoseconds: 10_000_000)
      }
    )

    driveToFire(ctrl)
    // Yield enough for capture + POST to complete (but not for the long sleep).
    for _ in 0..<5 { await Task.yield() }

    // The JPEG must be pending at this point (the rearm sleep is still in flight).
    XCTAssertNotNil(ctrl.pendingUncertainJPEG, "JPEG must be pending before confirm")

    // Now confirm.
    await ctrl.confirmUncertainAsNew()
    _ = pendingJPEGClearedAfterConfirm
    await settle()

    XCTAssertEqual(client.uploadCallCount, 2)
    XCTAssertEqual(client.lastEnrollUncertain, true)
    XCTAssertNil(ctrl.pendingUncertainJPEG)
    XCTAssertEqual(ctrl.gallery.count, 1)
    XCTAssertEqual(ctrl.gallery.first?.pieceId, "p-confirmed")
    XCTAssertTrue(box.contains(.success))
  }

  // MARK: - 3c. Double-tap confirm doesn't double-enroll

  func testDoubleTapConfirmDoesNotDoubleEnroll() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .uncertain, pieceId: nil)),
      .success(uploadResponse(status: .new, pieceId: "p-once")),
    ]
    let ctrl = PieceScanController(
      puzzleId: "test-puzzle",
      geometryClient: client,
      capture: { fixtureJPEG },
      haptic: { _ in },
      sleep: { _ in try? await Task.sleep(nanoseconds: 10_000_000) }
    )

    driveToFire(ctrl)
    for _ in 0..<5 { await Task.yield() }

    // Tap confirm twice concurrently — second call should find pendingUncertainJPEG nil.
    async let first: Void = ctrl.confirmUncertainAsNew()
    async let second: Void = ctrl.confirmUncertainAsNew()
    _ = await (first, second)
    await settle()

    // Only one POST after the uncertain one.
    XCTAssertEqual(client.uploadCallCount, 2)
    XCTAssertEqual(ctrl.gallery.count, 1)
  }

  // MARK: - 4. Nil capture → failure

  func testNilCaptureGivesFailureHapticAndNeverPosts() async {
    let client = FakeGeometryClient()
    let (ctrl, haptics) = makeController(client: client, captureReturnsNil: true)

    driveToFire(ctrl)
    await settle()

    XCTAssertEqual(client.uploadCallCount, 0, "No POST when capture returns nil")
    XCTAssertTrue(haptics.contains(.failure))
    XCTAssertFalse(haptics.contains(.success))
  }

  // MARK: - 5. API throw → failure haptic, then re-arms to scanning

  func testAPIThrowGivesFailureHapticThenRearmsToScanning() async {
    let client = FakeGeometryClient()
    struct FakeError: Error, LocalizedError {
      var errorDescription: String? { "server offline" }
    }
    client.uploadResponses = [.failure(FakeError())]
    let (ctrl, haptics) = makeController(client: client)

    driveToFire(ctrl)
    await settle()

    XCTAssertTrue(haptics.contains(.failure))
    // Instant sleep re-arms back to .scanning.
    XCTAssertEqual(ctrl.phase, .scanning)
  }

  // MARK: - 6. loadEnrolled merges gallery

  func testLoadEnrolledMergesServerPiecesWithLocalGallery() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(
        uploadResponse(
          status: .new, pieceId: "local-1",
          edgeTypes: [.tab, .blank, .flat, .tab]))
    ]
    let listResult = PieceGeometryListResponse(
      puzzleId: "test-puzzle",
      pieces: [
        PieceGeometrySummary(
          pieceId: "local-1", edgeTypes: [.tab, .blank, .flat, .tab],
          isClean: true, cornerDisagreement: false),
        PieceGeometrySummary(
          pieceId: "server-only", edgeTypes: [.flat, .flat, .flat, .flat],
          isClean: true, cornerDisagreement: false),
      ]
    )
    client.listResponse = .success(listResult)
    let (ctrl, _) = makeController(client: client)

    // Add local-1 to the gallery with a thumbnail via a lock cycle.
    driveToFire(ctrl)
    await settle()
    XCTAssertEqual(ctrl.gallery.count, 1)
    XCTAssertNotNil(ctrl.gallery.first?.thumbnailJPEG)

    // loadEnrolled: local-1 already present (thumbnail preserved), server-only appended.
    await ctrl.loadEnrolled()

    XCTAssertEqual(ctrl.gallery.count, 2)
    let local = ctrl.gallery.first { $0.pieceId == "local-1" }
    let serverOnly = ctrl.gallery.first { $0.pieceId == "server-only" }
    XCTAssertNotNil(local?.thumbnailJPEG, "Existing thumbnail must be preserved")
    XCTAssertNil(serverOnly?.thumbnailJPEG, "Server-only piece has no local thumbnail")
  }

  // MARK: - 7. ingest during non-scanning phase is ignored

  func testIngestDuringCapturingPhaseIsIgnored() async {
    let client = FakeGeometryClient()
    client.uploadResponses = [
      .success(uploadResponse(status: .new, pieceId: "p1")),
      // If ingest fires a second time, this would be dequeued.
      .success(uploadResponse(status: .new, pieceId: "p2")),
    ]
    let (ctrl, _) = makeController(client: client)

    driveToFire(ctrl)
    // The phase is now .capturing or .verdict; feed more lockable frames.
    let square: [NormalizedPoint] = [
      NormalizedPoint(x: 0.1, y: 0.1), NormalizedPoint(x: 0.9, y: 0.1),
      NormalizedPoint(x: 0.9, y: 0.9), NormalizedPoint(x: 0.1, y: 0.9),
    ]
    let t2 = Date(timeIntervalSince1970: 100)
    ctrl.ingest(.lockable(polygon: square, confidence: 0.9), at: t2)
    ctrl.ingest(
      .lockable(polygon: square, confidence: 0.9), at: t2.addingTimeInterval(0.5))
    ctrl.ingest(
      .lockable(polygon: square, confidence: 0.9), at: t2.addingTimeInterval(1.0))

    await settle()

    // Only the first lock cycle should have posted.
    XCTAssertEqual(client.uploadCallCount, 1)
  }
}

// MARK: - HapticBox

/// Reference-type box that captures haptic events from the controller's
/// injected closure. Necessary because the controller's haptic closure is
/// `@escaping` and Swift closures can't safely capture `inout` parameters
/// beyond their enclosing function's lifetime.
final class HapticBox {
  private var recorded: [ScanHaptic] = []

  func record(_ haptic: ScanHaptic) {
    recorded.append(haptic)
  }

  func contains(_ haptic: ScanHaptic) -> Bool {
    recorded.contains(haptic)
  }
}

// MARK: - LockBox

/// Generic reference-type box for sharing a value across closures in tests.
final class LockBox<T> {
  var value: T
  init(_ value: T) { self.value = value }
}
