import UIKit

/// The guidance backend behind the glare-free capture screen: something
/// that, once the center reference shot is taken, keeps telling the
/// controller where the current step's puzzle-anchored target sits in the
/// live view (as a `GlareGuideUpdate`).
///
/// Two implementations: `ARGlareGuideSource` (world-tracking ARKit — the
/// device path, targets can never be "lost" while tracking holds) and
/// `GlareGuideTracker` (Vision image registration — the Simulator/E2E
/// fallback, driven by `pusseldebug://previewloop`).
@MainActor
protocol GlareGuideSource: AnyObject {
  /// Called on the main actor with every guidance measurement. Set once by
  /// the owning view before guiding starts.
  var onUpdate: ((GlareGuideUpdate) -> Void)? { get set }

  /// The center shot just landed — start guiding the corner steps.
  /// `reference` is that shot (the registration tracker aligns live frames
  /// against it; the AR source freezes its world quad instead).
  func beginGuiding(reference: UIImage)

  /// The corner step currently being aimed (index into
  /// `GlareFreeCaptureController.steps`), nil when none is.
  func setActiveStep(_ step: Int?)

  /// The flow restarted — stop guiding until a new reference is taken.
  func stopGuiding()
}

extension GlareGuideTracker: GlareGuideSource {
  @MainActor func beginGuiding(reference: UIImage) {
    setReference(reference)
  }

  @MainActor func setActiveStep(_ step: Int?) {
    setExpectedOffset(step.flatMap { GlareFreeCaptureController.expectedShift(step: $0) })
  }

  @MainActor func stopGuiding() {
    clearReference()
  }
}
