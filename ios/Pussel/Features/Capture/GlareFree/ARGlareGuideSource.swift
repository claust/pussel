import ARKit
import CoreImage
import Observation
import SceneKit
import UIKit
import os
import simd

/// Pure geometry behind the AR guide source, split out so it is testable
/// without an `ARSession`.
enum ARGuideGeometry {
  /// Bilinear interpolation of a world quad at a unit position (top-left
  /// origin). Quad order matches the raycast screen corners: top left,
  /// top right, bottom right, bottom left.
  static func interpolated(quad: [SIMD3<Float>], at unit: CGPoint) -> SIMD3<Float> {
    let across = SIMD3<Float>(repeating: Float(unit.x))
    let top = simd_mix(quad[0], quad[1], across)
    let bottom = simd_mix(quad[3], quad[2], across)
    return simd_mix(top, bottom, SIMD3(repeating: Float(unit.y)))
  }

  /// Whether a world point lies in front of the camera (ARKit cameras look
  /// along their −z axis). Projecting a behind-the-camera point yields a
  /// mirrored garbage position, so callers must filter with this first.
  /// A dot product against the forward axis, not a matrix inverse — this
  /// runs per projected point every AR frame, and a camera transform is
  /// rigid, so the axis read is equivalent.
  static func isInFront(ofCameraAt transform: simd_float4x4, point: SIMD3<Float>) -> Bool {
    let position = SIMD3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
    let forward = -SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
    return simd_dot(point - position, forward) > 0
  }
}

/// ARKit-backed guide source for the glare-free capture flow — the device
/// path (`isSupported`), replacing live image registration with world
/// tracking so the corner targets cannot drift or vanish while the phone
/// tilts.
///
/// Before the center shot it raycasts the four screen corners onto the
/// detected horizontal plane every frame; once all four hit (`isReady`),
/// the shutter is worth pressing. `beginGuiding` freezes that quad as the
/// puzzle outline (the user was told to fit the whole puzzle in frame) and
/// interpolates the controller's step anchors into world targets on the
/// plane. From then on each frame projects outline + targets into view
/// points (`overlay`) and feeds the active step's target to `onUpdate` as
/// the same `GlareGuideUpdate` shape the registration tracker emits — the
/// controller's dwell auto-shutter is unchanged. Stills come from
/// `captureHighResolutionFrame()`, so the composer keeps getting full-
/// resolution photos.
@Observable
@MainActor
final class ARGlareGuideSource: NSObject, GlareGuideSource, ARSessionDelegate {
  /// Whether this device can run the AR guide. False on the Simulator —
  /// there the flow falls back to `GlareGuideTracker`, which keeps the
  /// `pusseldebug://previewloop` E2E path working.
  static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }

  /// Everything the view draws on top of the preview, in view points.
  struct Overlay: Equatable {
    /// The frozen puzzle outline — empty when a corner is behind the
    /// camera (no partial quad: a 3-point "rectangle" reads as a glitch).
    var outline: [CGPoint]
    /// One entry per corner step (steps 1…4), nil when behind the camera.
    var targets: [CGPoint?]
  }

  /// Corner raycasts landing farther than this from the camera are grazing
  /// hits on the infinite plane (phone aimed at the horizon), not a
  /// tabletop — treated as "surface not found".
  private static let maxSurfaceDistance: Float = 3

  private static let log = Logger(subsystem: "dk.delectosoft.pussel", category: "glare-guide")
  private static let ciContext = CIContext()

  /// The preview view. The source owns it so raycasts and projections use
  /// the exact viewport ARKit renders into; `ARGuidePreview` just hosts it.
  let sceneView = ARSCNView()
  private var session: ARSession { sceneView.session }

  @ObservationIgnored var onUpdate: ((GlareGuideUpdate) -> Void)?
  /// Reports a fatal session failure (camera access, tracking) with a
  /// user-facing message — the view surfaces it and dismisses.
  @ObservationIgnored var onFailure: ((String) -> Void)?

  /// Whether the puzzle's surface has been found (pre-shot): all four
  /// screen corners currently raycast onto a horizontal plane. Gates the
  /// reference shutter.
  private(set) var isReady = false
  /// The projected outline + targets, published every frame while guiding.
  private(set) var overlay: Overlay?

  /// The latest pre-shot corner quad, kept fresh by the live scan.
  @ObservationIgnored private var pendingQuad: [SIMD3<Float>]?
  /// The quad snapshotted at the reference photo's own camera pose —
  /// what `beginGuiding` freezes (the live scan may have moved on while
  /// the high-res capture was in flight).
  @ObservationIgnored private var quadAtCapture: [SIMD3<Float>]?
  @ObservationIgnored private var worldQuad: [SIMD3<Float>]?
  @ObservationIgnored private var worldTargets: [SIMD3<Float>]?
  @ObservationIgnored private var activeStep: Int?
  /// Whether the previous frame had a usable fix for the active target, so
  /// only acquired/lost transitions are logged — mirrors the tracker.
  @ObservationIgnored private var wasTracking = false
  /// The surface's world height, remembered from the last successful
  /// center raycast. Estimated-plane raycasts over a glossy puzzle come
  /// and go frame to frame (observed in the field: ready flickering
  /// against misses), but the table itself does not move — once found,
  /// the height keeps the quad available continuously and each success
  /// merely refreshes it. Cleared when the session (re)starts.
  @ObservationIgnored private var surfaceHeight: Float?
  /// Throttle for the once-per-second scanning diagnostics.
  @ObservationIgnored private var lastDiagnostic = Date.distantPast
  /// Why the last `raycastQuad` returned nil — folded into the diagnostic.
  @ObservationIgnored private var lastRaycastFailure = "no raycast yet"

  func run() {
    let configuration = ARWorldTrackingConfiguration()
    configuration.planeDetection = [.horizontal]
    // Video format whose still counterpart is full sensor resolution, so
    // captureHighResolutionFrame() hands the composer ~12 MP photos.
    if let format = ARWorldTrackingConfiguration
      .recommendedVideoFormatForHighResolutionFrameCapturing
    {
      configuration.videoFormat = format
    }
    surfaceHeight = nil
    session.delegate = self
    session.run(configuration)
    Self.log.notice(
      "AR session started, video format \(configuration.videoFormat.imageResolution.debugDescription, privacy: .public)"
    )
  }

  func pause() {
    session.pause()
  }

  /// The current frame at full still resolution, rotated upright. Nil on
  /// capture failure — the controller fails the step and offers a restart.
  func captureStill() async -> UIImage? {
    let size = sceneView.bounds.size
    do {
      let frame = try await session.captureHighResolutionFrame()
      if worldQuad == nil {
        // This is the reference shot: freeze the quad from the *returned*
        // frame's camera pose — the pose at the moment the photo actually
        // fired — not from whatever the live scan measures once the async
        // capture returns. The user may move the phone while the high-res
        // capture is in flight, and the outline must match the photo.
        quadAtCapture =
          Self.quad(
            onPlaneAtHeight: surfaceHeight, camera: frame.camera, viewportSize: size)
          ?? pendingQuad
      }
      return Self.stillImage(from: frame.capturedImage)
    } catch {
      Self.log.error("high-resolution capture failed: \(error.localizedDescription)")
      return nil
    }
  }

  /// The screen-corner quad as seen by `camera`, unprojected onto the
  /// horizontal plane at `height` — nil when the surface is unknown or a
  /// corner ray misses the plane.
  private static func quad(
    onPlaneAtHeight height: Float?, camera: ARCamera, viewportSize size: CGSize
  ) -> [SIMD3<Float>]? {
    guard let height, size.width > 0, size.height > 0 else { return nil }
    var plane = matrix_identity_float4x4
    plane.columns.3 = SIMD4(0, height, 0, 1)
    let corners = [
      CGPoint(x: 0, y: 0),
      CGPoint(x: size.width, y: 0),
      CGPoint(x: size.width, y: size.height),
      CGPoint(x: 0, y: size.height),
    ]
    var quad: [SIMD3<Float>] = []
    for corner in corners {
      guard
        let point = camera.unprojectPoint(
          corner, ontoPlane: plane, orientation: .portrait, viewportSize: size)
      else { return nil }
      quad.append(point)
    }
    return quad
  }

  // MARK: - GlareGuideSource

  func beginGuiding(reference _: UIImage) {
    // The reference image itself is not needed — world tracking replaces
    // image registration. What matters is the quad under the screen at the
    // moment the shot fired, snapshotted by `captureStill`.
    guard let quad = quadAtCapture ?? pendingQuad else {
      Self.log.error("center shot landed without a surface quad — AR guiding cannot start")
      return
    }
    quadAtCapture = nil
    worldQuad = quad
    // The anchors live in unit coordinates of the reference photo; the quad
    // corners are the *screen* corners at the center shot. The preview is
    // an aspect-fill crop of the photo, so the two differ by a small crop —
    // fine for guidance, and the composer re-registers the real offsets.
    worldTargets = GlareFreeCaptureController.steps.dropFirst().map {
      ARGuideGeometry.interpolated(quad: quad, at: $0.anchor)
    }
    wasTracking = false
  }

  func setActiveStep(_ step: Int?) {
    activeStep = step
  }

  func stopGuiding() {
    worldQuad = nil
    worldTargets = nil
    activeStep = nil
    overlay = nil
    wasTracking = false
    // Drop the scanning state too: the restart may happen over a different
    // surface, and reacquiring one takes only a couple of seconds — better
    // than a ready state resting on a stale height.
    pendingQuad = nil
    quadAtCapture = nil
    surfaceHeight = nil
    isReady = false
  }

  // MARK: - ARSessionDelegate

  // ARSession calls its delegate on the main queue (delegateQueue is nil),
  // so hopping back into the actor is an assertion, not a dispatch.
  nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
    MainActor.assumeIsolated { handle(frame) }
  }

  nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
    MainActor.assumeIsolated {
      Self.log.error("AR session failed: \(error.localizedDescription)")
      let message =
        (error as? ARError)?.code == .cameraUnauthorized
        ? "Pussel cannot use the camera. Check camera access in Settings."
        : "The camera's motion tracking failed."
      onFailure?(message)
    }
  }

  private func handle(_ frame: ARFrame) {
    let size = sceneView.bounds.size
    guard size.width > 0, size.height > 0 else {
      logDiagnostic("view not laid out yet", frame: frame)
      return
    }
    let aspect = size.width / size.height
    guard case .normal = frame.camera.trackingState else {
      isReady = false
      pendingQuad = nil
      logDiagnostic("tracking not normal", frame: frame)
      if worldQuad != nil {
        overlay = nil
        emit(offset: nil, aspect: aspect)
      }
      return
    }
    if worldQuad == nil {
      pendingQuad = raycastQuad(frame: frame, size: size)
      isReady = pendingQuad != nil
      logDiagnostic(isReady ? "surface ready" : lastRaycastFailure, frame: frame)
    } else {
      publishGuidance(frame: frame, size: size, aspect: aspect)
    }
  }

  /// One state line per second while the surface is being sought — enough
  /// to tell a stuck session from missing planes from failing raycasts
  /// when scanning misbehaves in the field.
  private func logDiagnostic(_ detail: String, frame: ARFrame) {
    let now = Date()
    guard now.timeIntervalSince(lastDiagnostic) >= 1 else { return }
    lastDiagnostic = now
    let tracking: String
    switch frame.camera.trackingState {
    case .normal: tracking = "normal"
    case .notAvailable: tracking = "notAvailable"
    case .limited(let reason): tracking = "limited(\(String(describing: reason)))"
    }
    let planes = frame.anchors.count(where: { $0 is ARPlaneAnchor })
    Self.log.notice(
      "AR scan: \(detail, privacy: .public); tracking=\(tracking, privacy: .public) planes=\(planes)"
    )
  }

  /// The four screen corners projected onto the puzzle's surface plane —
  /// nil until the surface has been found.
  ///
  /// Only the screen *center* is raycast (that is where the puzzle is —
  /// the one spot guaranteed to have scene coverage; the extreme corners
  /// of the field of view often have none, and corner raycasts were
  /// observed to fail indefinitely in the field). The hit fixes the
  /// surface's height; the world is gravity-aligned, so the surface is the
  /// horizontal plane at that height, and each corner is its screen ray
  /// intersected with that plane analytically.
  private func raycastQuad(frame: ARFrame, size: CGSize) -> [SIMD3<Float>]? {
    let camera = SIMD3(
      frame.camera.transform.columns.3.x,
      frame.camera.transform.columns.3.y,
      frame.camera.transform.columns.3.z)
    if let surface = raycastSurfacePoint(from: CGPoint(x: size.width / 2, y: size.height / 2)),
      simd_distance(surface, camera) < Self.maxSurfaceDistance
    {
      surfaceHeight = surface.y
    }
    guard let height = surfaceHeight else { return nil }
    let corners = [
      CGPoint(x: 0, y: 0),
      CGPoint(x: size.width, y: 0),
      CGPoint(x: size.width, y: size.height),
      CGPoint(x: 0, y: size.height),
    ]
    var quad: [SIMD3<Float>] = []
    for corner in corners {
      guard let point = intersection(ofScreenPoint: corner, withHorizontalPlaneAtY: height),
        simd_distance(point, camera) < Self.maxSurfaceDistance
      else {
        lastRaycastFailure = "corner ray misses surface at \(Int(corner.x)),\(Int(corner.y))"
        return nil
      }
      quad.append(point)
    }
    return quad
  }

  /// The center's hit on the puzzle's surface: a detected plane when one
  /// exists, otherwise ARKit's estimated plane from the current frame's
  /// scene geometry. The fallback is load-bearing — over a glossy puzzle
  /// filling the view, plane *detection* can take arbitrarily long or
  /// never converge (observed in the field: tracking normal, zero plane
  /// anchors for 30+ s), while the estimate is available almost at once.
  /// The quad is only frozen once at the shutter, so estimate jitter does
  /// not move anchored targets.
  private func raycastSurfacePoint(from point: CGPoint) -> SIMD3<Float>? {
    let attempts: [(ARRaycastQuery.Target, ARRaycastQuery.TargetAlignment)] = [
      (.existingPlaneInfinite, .horizontal),
      (.estimatedPlane, .any),
    ]
    for (target, alignment) in attempts {
      guard
        let query = sceneView.raycastQuery(from: point, allowing: target, alignment: alignment),
        let hit = session.raycast(query).first
      else { continue }
      return SIMD3(
        hit.worldTransform.columns.3.x,
        hit.worldTransform.columns.3.y,
        hit.worldTransform.columns.3.z)
    }
    lastRaycastFailure = "no surface hit at \(Int(point.x)),\(Int(point.y))"
    return nil
  }

  /// Where the ray through a screen point meets the horizontal plane at
  /// world height `planeY` — nil when the ray points away from it.
  private func intersection(
    ofScreenPoint point: CGPoint, withHorizontalPlaneAtY planeY: Float
  ) -> SIMD3<Float>? {
    let near = sceneView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 0))
    let far = sceneView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 1))
    let origin = SIMD3(near.x, near.y, near.z)
    let direction = SIMD3(far.x - near.x, far.y - near.y, far.z - near.z)
    guard abs(direction.y) > .ulpOfOne else { return nil }
    let along = (planeY - origin.y) / direction.y
    guard along > 0 else { return nil }
    return origin + along * direction
  }

  /// Projects the frozen outline and targets into view points and feeds
  /// the active step's target to the controller.
  private func publishGuidance(frame: ARFrame, size: CGSize, aspect: CGFloat) {
    guard let worldQuad, let worldTargets else { return }
    let cameraTransform = frame.camera.transform
    func projected(_ world: SIMD3<Float>) -> CGPoint? {
      guard ARGuideGeometry.isInFront(ofCameraAt: cameraTransform, point: world) else {
        return nil
      }
      return frame.camera.projectPoint(world, orientation: .portrait, viewportSize: size)
    }
    let outline = worldQuad.compactMap(projected)
    let targets = worldTargets.map(projected)
    overlay = Overlay(
      outline: outline.count == worldQuad.count ? outline : [],
      targets: targets)
    guard let step = activeStep, step >= 1, step - 1 < targets.count else { return }
    let offset: CGSize?
    if let view = targets[step - 1] {
      // Synthesized so the controller's `anchor + offset` lands on the
      // projected view-unit position; with frameAspect equal to the view's
      // aspect, the view's aspect-fill mapping becomes the identity.
      let anchor = GlareFreeCaptureController.steps[step].anchor
      offset = CGSize(
        width: view.x / size.width - anchor.x,
        height: view.y / size.height - anchor.y)
    } else {
      offset = nil
    }
    emit(offset: offset, aspect: aspect)
  }

  private func emit(offset: CGSize?, aspect: CGFloat) {
    let isTracking = offset != nil
    if isTracking != wasTracking {
      wasTracking = isTracking
      Self.log.notice("AR guidance \(isTracking ? "acquired" : "lost")")
    }
    onUpdate?(GlareGuideUpdate(offset: offset, frameAspect: aspect))
  }

  /// The captured buffer rotated upright: AR frames arrive in the sensor's
  /// landscape orientation, and the app is portrait-only, so `.right` is
  /// the one rotation in play — same convention as the camera sessions'
  /// `videoRotationAngle = 90`.
  private static func stillImage(from buffer: CVPixelBuffer) -> UIImage? {
    let upright = CIImage(cvPixelBuffer: buffer).oriented(.right)
    guard let cgImage = ciContext.createCGImage(upright, from: upright.extent) else { return nil }
    return UIImage(cgImage: cgImage)
  }
}
