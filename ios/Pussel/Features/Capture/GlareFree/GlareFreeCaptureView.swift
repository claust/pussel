import SwiftUI
import UIKit

/// Guided five-shot glare-free capture of the full puzzle, presented from
/// CapturePuzzleView. The user photographs the puzzle once from the
/// center; after that the guide dot is anchored to a fixed spot *on the
/// puzzle* (live-tracked by `GlareGuideTracker`), and each corner shot
/// fires automatically once the user has steered that spot into the
/// screen-center ring and held it there. `GlareFreeComposer` then fuses
/// the burst into one glare-free image that continues down the normal
/// detect-frame path via `onImage`.
///
/// Reuses `BoxCameraSession` with the tracker attached as its live frame
/// consumer — the "third live-camera screen" the session's extraction
/// note anticipated.
struct GlareFreeCaptureView: View {
  /// The composite (or, if no frames registered, the center shot) — routed
  /// to the existing detect-frame path.
  let onImage: (UIImage) -> Void

  @Environment(AppModel.self) private var model
  @Environment(\.dismiss) private var dismiss
  @State private var camera = BoxCameraSession()
  @State private var tracker = GlareGuideTracker()
  @State private var controller: GlareFreeCaptureController?

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()
      BoxCameraPreview(session: camera.session)
        .ignoresSafeArea()
      guidanceOverlay
    }
    .overlay(alignment: .top) { topBar }
    .overlay(alignment: .bottom) { bottomBar }
    .task { await startSession() }
    .onDisappear {
      camera.stop()
      #if DEBUG
        if GlareFreeCaptureController.debugActive === controller {
          GlareFreeCaptureController.debugActive = nil
        }
      #endif
    }
    .onChange(of: controller?.phase) { _, phase in
      switch phase {
      case .capturing(step: 0):
        // Back at the start (restart after a failure) — stop tracking
        // until a new reference is taken.
        tracker.clearReference()
      case .capturing(let step):
        // The center shot just landed — it becomes the tracking reference
        // for the corner steps.
        if step == 1, let reference = controller?.referenceShot {
          tracker.setReference(reference)
        }
        // Tell the tracker where this step steers the content, so it can
        // use the expectation as a registration prior.
        tracker.setExpectedOffset(GlareFreeCaptureController.expectedShift(step: step))
      case .done:
        guard let composite = controller?.composite else { return }
        dismiss()
        if composite.alignedFrameCount == 0 {
          model.flow.errorMessage =
            "Could not align the extra shots — used the center photo as-is."
        }
        onImage(composite.image)
      default:
        break
      }
    }
    .keepsScreenAwake()
  }

  private func startSession() async {
    let controller =
      self.controller
      ?? GlareFreeCaptureController(capture: { [camera] in
        await camera.capturePhoto()
      })
    self.controller = controller
    #if DEBUG
      GlareFreeCaptureController.debugActive = controller
    #endif
    tracker.onUpdate = { update in
      controller.ingestGuide(update)
    }
    camera.attachFrameConsumer(tracker)
    let started = await camera.start()
    // Dismissed while the permission prompt was up — the user left on
    // purpose, so there is nothing to complain about.
    guard !Task.isCancelled else { return }
    guard started else {
      #if DEBUG
        // The Simulator has no camera; stay on screen over the black
        // background so `pusseldebug://previewloop` + `glareshot` can
        // drive the flow. A real device failure still reports and
        // dismisses — mirrors BoxCameraView.
        if !BoxCameraSession.isCameraAvailable { return }
      #endif
      model.flow.errorMessage =
        "Pussel cannot use the camera. Check camera access in Settings."
      dismiss()
      return
    }
    camera.setStreamingEnabled(true)
  }

  /// The guide graphics over the preview. While aiming the reference shot
  /// the pulsing dot simply marks the screen center; on the corner steps a
  /// fixed ring marks the target and the dot moves with the puzzle — the
  /// tracker pins it to the current step's anchor spot, so steering it
  /// into the ring is steering the phone.
  private var guidanceOverlay: some View {
    GeometryReader { proxy in
      if let controller, case .capturing(let step) = controller.phase {
        let center = CGPoint(x: proxy.size.width / 2, y: proxy.size.height / 2)
        if step == 0 {
          GuideDot(onTarget: false)
            .position(center)
        } else {
          TargetRing(onTarget: dotIsOnTarget)
            .position(center)
          if let dot = controller.dotUnitPosition,
            let aspect = controller.guide?.frameAspect
          {
            GuideDot(onTarget: dotIsOnTarget)
              .position(
                clamped(
                  mapped(dot, frameAspect: aspect, into: proxy.size),
                  within: proxy.size))
          }
        }
      }
    }
    .allowsHitTesting(false)
    .ignoresSafeArea()
  }

  /// Whether the tracked dot currently sits inside the aim tolerance — for
  /// tinting only; the controller makes the actual shutter decision.
  private var dotIsOnTarget: Bool {
    guard let dot = controller?.dotUnitPosition else { return false }
    return hypot(dot.x - 0.5, dot.y - 0.5) <= GlareFreeCaptureController.aimTolerance
  }

  /// Maps a unit point of the camera frame into view coordinates through
  /// the preview's aspect-fill: the frame is scaled to cover the view and
  /// centered, exactly as `AVCaptureVideoPreviewLayer` renders it.
  private func mapped(_ unit: CGPoint, frameAspect: CGFloat, into size: CGSize) -> CGPoint {
    guard size.width > 0, size.height > 0, frameAspect > 0 else { return .zero }
    let filled =
      size.width / size.height > frameAspect
      ? CGSize(width: size.width, height: size.width / frameAspect)
      : CGSize(width: size.height * frameAspect, height: size.height)
    return CGPoint(
      x: (size.width - filled.width) / 2 + unit.x * filled.width,
      y: (size.height - filled.height) / 2 + unit.y * filled.height)
  }

  /// Keeps the dot visible even when its anchor is outside the view — a
  /// dot pinned to the edge still tells the user which way to move.
  private func clamped(_ point: CGPoint, within size: CGSize) -> CGPoint {
    let inset: CGFloat = 44
    return CGPoint(
      x: min(max(point.x, inset), size.width - inset),
      y: min(max(point.y, inset), size.height - inset))
  }

  private var topBar: some View {
    HStack {
      Button("Cancel") { dismiss() }
        .foregroundStyle(.white)
      Spacer()
      if let controller, case .capturing(let step) = controller.phase {
        Text("Photo \(step + 1) of \(GlareFreeCaptureController.steps.count)")
          .font(.callout.weight(.semibold))
          .foregroundStyle(.white)
      }
    }
    .padding()
  }

  private var bottomBar: some View {
    VStack(spacing: 16) {
      switch controller?.phase {
      case .capturing(let step):
        instructionBanner(step: step)
        if step == 0 {
          shutterButton
        }
      case .composing:
        progressBanner
      case .failed(let message):
        failureBanner(message)
      case .done, nil:
        EmptyView()
      }
    }
    .padding(.bottom, 24)
  }

  private func instructionBanner(step: Int) -> some View {
    let text: String
    if step == 0 {
      text = "Fit the whole puzzle in the frame, then tap the shutter."
    } else if controller?.guide?.offset == nil {
      text = "Looking for the puzzle — keep it fully in view."
    } else {
      text = "Move the phone until the dot sits in the ring — the photo takes itself."
    }
    return banner {
      Text(text)
        .font(.callout.weight(.semibold))
        .foregroundStyle(.white)
        .multilineTextAlignment(.center)
    }
  }

  private var progressBanner: some View {
    banner {
      HStack(spacing: 10) {
        ProgressView()
          .tint(.white)
        Text("Removing glare…")
          .font(.callout.weight(.semibold))
          .foregroundStyle(.white)
      }
    }
  }

  private func failureBanner(_ message: String) -> some View {
    banner {
      VStack(spacing: 10) {
        Text(message)
          .font(.callout.weight(.semibold))
          .foregroundStyle(.white)
        Button("Start Over") { controller?.restart() }
          .buttonStyle(.borderedProminent)
      }
    }
  }

  /// Capsule chrome shared by the banners — styled after BoxCameraView's
  /// lookup banner.
  private func banner(@ViewBuilder content: () -> some View) -> some View {
    content()
      .padding(.horizontal, 16)
      .padding(.vertical, 10)
      .background(RoundedRectangle(cornerRadius: 16).fill(.ultraThinMaterial.opacity(0.9)))
      .shadow(radius: 4)
      .padding(.horizontal, 24)
  }

  private var shutterButton: some View {
    Button {
      Task { await controller?.captureShot() }
    } label: {
      ZStack {
        Circle().fill(.white).frame(width: 58, height: 58)
        Circle().strokeBorder(.white, lineWidth: 3).frame(width: 70, height: 70)
      }
    }
    .disabled(controller?.isCapturing ?? true)
    .accessibilityLabel("Take the reference shot")
  }
}

/// The pulsing guide dot. On the corner steps it is pinned to a spot on
/// the puzzle and turns green as it enters the target ring.
private struct GuideDot: View {
  let onTarget: Bool
  @State private var pulsing = false

  var body: some View {
    ZStack {
      Circle()
        .stroke(onTarget ? .green : .white, lineWidth: 3)
        .frame(width: 56, height: 56)
        .scaleEffect(pulsing ? 1.15 : 0.9)
        .opacity(pulsing ? 0.5 : 1)
        .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true), value: pulsing)
      Circle()
        .fill(onTarget ? .green : .white)
        .frame(width: 14, height: 14)
    }
    .shadow(radius: 4)
    .onAppear { pulsing = true }
  }
}

/// The fixed screen-center ring the tracked dot must be steered into.
private struct TargetRing: View {
  let onTarget: Bool

  var body: some View {
    Circle()
      .stroke(
        onTarget ? Color.green : Color.white.opacity(0.85),
        style: StrokeStyle(lineWidth: 3, dash: [7, 6])
      )
      .frame(width: 84, height: 84)
      .shadow(radius: 4)
  }
}
