import Observation
import UIKit

/// One step of the guided five-shot glare-free capture: a name for the
/// progress line and the spot on the puzzle the guide dot is pinned to.
struct GlareFreeStep: Equatable {
  let title: String
  /// The dot's anchor in unit coordinates *of the center reference shot*
  /// (0…1, top-left origin). For the first step there is no reference yet,
  /// so the anchor doubles as a static screen position; for the corner
  /// steps the live tracker moves the dot so it stays glued to this spot
  /// on the puzzle while the phone moves.
  let anchor: CGPoint
}

/// State machine for the glare-free capture flow: a manual center shot,
/// then four corner shots that fire automatically — the guide dot is
/// anchored to a fixed spot on the puzzle (`ingestGuide`), and once the
/// user has steered it into the screen-center ring and held it there, the
/// step's photo is taken and the next anchor lights up. A background
/// composition pass (`GlareFreeComposer`) then fuses the burst.
/// Dependency-injected capture and compose closures keep it unit-testable
/// without a camera — mirrors `PieceScanController`.
@Observable
@MainActor
final class GlareFreeCaptureController {
  enum Phase: Equatable {
    case capturing(step: Int)
    case composing
    case done
    case failed(String)
  }

  /// The capture sequence. The center shot comes first because it becomes
  /// the composite's reference frame — the output keeps its viewpoint. The
  /// corner anchors are modest offsets (~a quarter of the frame): steering
  /// one to the screen center shifts the camera by that same fraction,
  /// which keeps the puzzle fully in frame while still displacing a
  /// specular reflection a long way.
  static let steps: [GlareFreeStep] = [
    GlareFreeStep(title: "Center", anchor: CGPoint(x: 0.5, y: 0.5)),
    GlareFreeStep(title: "Top left", anchor: CGPoint(x: 0.25, y: 0.32)),
    GlareFreeStep(title: "Top right", anchor: CGPoint(x: 0.75, y: 0.32)),
    GlareFreeStep(title: "Bottom right", anchor: CGPoint(x: 0.75, y: 0.68)),
    GlareFreeStep(title: "Bottom left", anchor: CGPoint(x: 0.25, y: 0.68)),
  ]

  /// How close (unit distance) the tracked dot must be to the screen
  /// center for the auto-shutter's dwell to count.
  static let aimTolerance: CGFloat = 0.06

  private(set) var phase: Phase = .capturing(step: 0)
  private(set) var isCapturing = false
  /// The finished composite; set as `phase` becomes `.done`.
  private(set) var composite: GlareFreeComposer.Composite?
  /// The center shot, once taken — the view hands it to the guide tracker
  /// as the registration reference for the corner steps.
  private(set) var referenceShot: UIImage?
  /// The latest tracker measurement (corner steps only). Nil before the
  /// first measurement and after each step advance; a non-nil update with
  /// a nil offset means tracking is currently lost.
  private(set) var guide: GlareGuideUpdate?

  private let capture: () async -> UIImage?
  private let compose: (UIImage, [UIImage], [CGSize?]) async -> GlareFreeComposer.Composite?
  private var captured: [UIImage] = []
  private var aim = GlareAimStabilityTracker()

  #if DEBUG
    /// The controller of the currently presented glare-free screen, so
    /// `pusseldebug://glareshot` can drive the shutter on the Simulator —
    /// mirrors `PieceScanController.debugActive`. Registered by
    /// `GlareFreeCaptureView` while it is presented.
    static weak var debugActive: GlareFreeCaptureController?
  #endif

  init(
    capture: @escaping () async -> UIImage?,
    compose: @escaping (UIImage, [UIImage], [CGSize?]) async -> GlareFreeComposer.Composite? =
      { reference, others, expectedShifts in
        await Task.detached(priority: .userInitiated) {
          GlareFreeComposer.compose(
            reference: reference, others: others, expectedShifts: expectedShifts)
        }.value
      }
  ) {
    self.capture = capture
    self.compose = compose
  }

  var currentStep: GlareFreeStep? {
    guard case .capturing(let index) = phase else { return nil }
    return Self.steps[index]
  }

  var capturedCount: Int { captured.count }

  /// Where the guide dot currently sits, in unit coordinates of the live
  /// frame: the screen center while aiming the reference shot, and the
  /// tracked anchor position afterwards — nil there until tracking has a
  /// fix, so the view can show "looking for the puzzle" instead of a dot
  /// pinned to a stale spot.
  var dotUnitPosition: CGPoint? {
    guard case .capturing(let index) = phase else { return nil }
    guard index > 0 else { return CGPoint(x: 0.5, y: 0.5) }
    guard let offset = guide?.offset else { return nil }
    let anchor = Self.steps[index].anchor
    return CGPoint(x: anchor.x + offset.width, y: anchor.y + offset.height)
  }

  /// The content displacement expected of step `index`'s shot relative to
  /// the reference (unit coordinates, top-left origin): the shot fires
  /// with the step's anchor steered to the frame center, so the content
  /// has moved by exactly that steering distance. Seeds the composer's
  /// registration.
  static func expectedShift(step index: Int) -> CGSize? {
    guard index > 0, index < steps.count else { return nil }
    let anchor = steps[index].anchor
    return CGSize(width: 0.5 - anchor.x, height: 0.5 - anchor.y)
  }

  /// Feeds one tracker measurement: remembers it for the view's dot and,
  /// when the dot has dwelt on the screen center long enough, fires the
  /// step's photo. Corner steps only — the reference shot is manual.
  func ingestGuide(_ update: GlareGuideUpdate, at date: Date = Date()) {
    guard case .capturing(let index) = phase, index > 0, !isCapturing else { return }
    guide = update
    let onTarget: Bool
    if let dot = dotUnitPosition {
      onTarget = hypot(dot.x - 0.5, dot.y - 0.5) <= Self.aimTolerance
    } else {
      onTarget = false
    }
    guard aim.ingest(onTarget: onTarget, at: date) else { return }
    Task { await self.captureShot() }
  }

  /// Takes the current step's photo and advances; the last step triggers
  /// composition. Fired by the shutter button on the reference step and by
  /// the aim dwell on the corner steps; shots while one is already
  /// developing are ignored, so each step captures exactly once.
  func captureShot() async {
    guard case .capturing(let index) = phase, !isCapturing else { return }
    isCapturing = true
    defer { isCapturing = false }
    guard let image = await capture() else {
      phase = .failed("Could not take that photo.")
      return
    }
    captured.append(image)
    if index == 0 {
      referenceShot = image
    }
    guard index + 1 < Self.steps.count else {
      await composeCaptured()
      return
    }
    // Re-arm the aim for the next anchor; the stale dot would otherwise
    // still sit at the screen center and instantly re-fire.
    aim.reset()
    guide = nil
    phase = .capturing(step: index + 1)
  }

  private func composeCaptured() async {
    phase = .composing
    let reference = captured[0]
    let others = Array(captured.dropFirst())
    let shifts = (1..<Self.steps.count).map(Self.expectedShift(step:))
    // A composer failure still leaves the reference shot — degrade to a
    // normal single-photo capture rather than dead-ending the flow. The
    // view surfaces the degradation via `alignedFrameCount == 0`.
    composite =
      await compose(reference, others, shifts)
      ?? GlareFreeComposer.Composite(image: reference, alignedFrameCount: 0)
    phase = .done
  }

  /// Restarts the five-shot sequence (offered after a failed capture).
  func restart() {
    guard !isCapturing else { return }
    captured = []
    composite = nil
    referenceShot = nil
    guide = nil
    aim.reset()
    phase = .capturing(step: 0)
  }
}
