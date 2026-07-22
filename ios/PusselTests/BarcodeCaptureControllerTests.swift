import XCTest

@testable import Pussel

// MARK: - Fake lookup client

/// Recording fake that returns scripted `BarcodeLookupResponse` values
/// without any network calls — mirrors `FakeGeometryClient`.
@MainActor
final class FakeBarcodeLookupClient: BarcodeLookupClient {
  private(set) var lookupCallCount = 0
  private(set) var receivedEANs: [String] = []

  /// Queue of lookup responses. Each call dequeues one; panics if the queue
  /// runs dry (indicates the test over-called).
  var lookupResponses: [Result<BarcodeLookupResponse, Error>] = []

  func lookupBarcode(ean: String) async throws -> BarcodeLookupResponse {
    lookupCallCount += 1
    receivedEANs.append(ean)
    guard !lookupResponses.isEmpty else {
      fatalError("FakeBarcodeLookupClient: no more scripted lookup responses")
    }
    return try lookupResponses.removeFirst().get()
  }
}

// MARK: - Helpers

private let frozenEAN = "4005556050093"
private let boxJPEG = Data("box-jpeg-bytes".utf8)

private func foundResponse(jpeg: Data = boxJPEG) -> BarcodeLookupResponse {
  BarcodeLookupResponse(
    found: true,
    boxImage: "data:image/jpeg;base64,\(jpeg.base64EncodedString())",
    articleNumber: "05009"
  )
}

private let missResponse = BarcodeLookupResponse(found: false, boxImage: nil, articleNumber: nil)

private func detection(_ payload: String) -> EAN13Detection {
  EAN13Detection(payload: payload, boundingBox: CGRect(x: 0.2, y: 0.4, width: 0.5, height: 0.15))
}

/// Lets the controller's unstructured lookup task run to completion —
/// mirrors PieceScanControllerTests' settle helper.
@MainActor
private func settle() async {
  for _ in 0..<8 { await Task.yield() }
}

// MARK: - Tests

@MainActor
final class BarcodeCaptureControllerTests: XCTestCase {
  private func makeController(
    client: FakeBarcodeLookupClient,
    requiredHits: Int = 3
  ) -> (BarcodeCaptureController, capture: () -> [Data]) {
    let controller = BarcodeCaptureController(
      client: client,
      tracker: BarcodeStabilityTracker(requiredConsecutiveHits: requiredHits)
    )
    // Reference type so the closure and the test observe the same array.
    let found = FoundRecorder()
    controller.onFound = { found.jpegs.append($0) }
    return (controller, { found.jpegs })
  }

  private final class FoundRecorder {
    var jpegs: [Data] = []
  }

  func testLookupFiresOnlyAfterStableRead() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [.success(foundResponse())]
    let (controller, _) = makeController(client: client)

    controller.ingest(detection(frozenEAN))
    controller.ingest(detection(frozenEAN))
    await settle()
    XCTAssertEqual(client.lookupCallCount, 0)

    controller.ingest(detection(frozenEAN))
    await settle()
    XCTAssertEqual(client.lookupCallCount, 1)
    XCTAssertEqual(client.receivedEANs, [frozenEAN])
  }

  func testFoundResponseDeliversDecodedJPEG() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [.success(foundResponse())]
    let (controller, capturedJPEGs) = makeController(client: client)

    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()

    XCTAssertEqual(capturedJPEGs(), [boxJPEG])
    XCTAssertEqual(controller.phase, .scanning)
  }

  func testMissBlacklistsEANForTheSession() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [.success(missResponse)]
    let (controller, capturedJPEGs) = makeController(client: client)

    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertEqual(client.lookupCallCount, 1)
    XCTAssertTrue(capturedJPEGs().isEmpty)

    // The same code held in view keeps producing stable reads — none of
    // them may re-fire the lookup.
    for _ in 0..<6 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertEqual(client.lookupCallCount, 1)
  }

  func testMissDoesNotBlacklistOtherEANs() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [.success(missResponse), .success(foundResponse())]
    let (controller, capturedJPEGs) = makeController(client: client)

    for _ in 0..<3 { controller.ingest(detection("4006381333931")) }
    await settle()
    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()

    XCTAssertEqual(client.receivedEANs, ["4006381333931", frozenEAN])
    XCTAssertEqual(capturedJPEGs(), [boxJPEG])
  }

  func testTransportErrorDoesNotBlacklist() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [
      .failure(APIError(message: "offline", status: nil)),
      .success(foundResponse()),
    ]
    let (controller, capturedJPEGs) = makeController(client: client)

    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertEqual(client.lookupCallCount, 1)
    XCTAssertTrue(capturedJPEGs().isEmpty)
    XCTAssertEqual(controller.phase, .scanning)

    // The barcode leaves the frame and comes back: the retry succeeds.
    controller.ingest(nil)
    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertEqual(client.lookupCallCount, 2)
    XCTAssertEqual(capturedJPEGs(), [boxJPEG])
  }

  func testUndecodableImageInFoundResponseIsTreatedAsMiss() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [
      .success(BarcodeLookupResponse(found: true, boxImage: nil, articleNumber: "05009"))
    ]
    let (controller, capturedJPEGs) = makeController(client: client)

    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertTrue(capturedJPEGs().isEmpty)

    // Blacklisted — no retry loop against a persistently broken response.
    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertEqual(client.lookupCallCount, 1)
  }

  func testIngestsDuringLookupAreDropped() async {
    let client = FakeBarcodeLookupClient()
    client.lookupResponses = [.success(foundResponse())]
    let (controller, _) = makeController(client: client)

    for _ in 0..<3 { controller.ingest(detection(frozenEAN)) }
    // Immediately feed more stable reads before the lookup task settles;
    // the phase guard must drop them rather than queue a second lookup.
    for _ in 0..<6 { controller.ingest(detection(frozenEAN)) }
    await settle()
    XCTAssertEqual(client.lookupCallCount, 1)
  }
}
