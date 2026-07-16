import SwiftUI

struct ConfirmTrimView: View {
  /// Mirrors LOW_CONFIDENCE_THRESHOLD in frontend/src/app/real/page.tsx.
  private static let lowConfidence = 0.4

  @Environment(AppModel.self) private var model
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @State private var quarterTurns = 0
  @State private var hintOpacity = 0.0
  @State private var hintTurns = 0
  @State private var didPlayHint = false
  let candidate: TrimCandidate
  /// Decoded once at init rather than in `body`, which re-runs on every
  /// rotate tap (base64-decoding the data URL and the JPEG each time would be
  /// wasteful during the rotation animation).
  private let previewImage: UIImage?

  init(candidate: TrimCandidate) {
    self.candidate = candidate
    self.previewImage = candidate.trimmedJPEG.flatMap(UIImage.init(data:))
  }

  var body: some View {
    VStack(spacing: 16) {
      Text("Does this look right?")
        .font(.title2.bold())
      if candidate.detection.confidence < Self.lowConfidence {
        Label(
          "Detection looks uncertain — consider retaking.",
          systemImage: "exclamationmark.triangle.fill"
        )
        .font(.footnote)
        .foregroundStyle(.orange)
      }
      if let image = previewImage {
        // Square bounding box so the image stays fully visible at every
        // angle while it spins (a rotating landscape/portrait photo would
        // otherwise overflow its frame at the 90°/270° positions).
        Color.clear
          .aspectRatio(1, contentMode: .fit)
          .overlay {
            Image(uiImage: image)
              .resizable()
              .scaledToFit()
              .clipShape(RoundedRectangle(cornerRadius: 12))
              .rotationEffect(.degrees(Double(quarterTurns) * 90))
          }
          .frame(maxHeight: 420)
          .overlay { tapHint }
          .contentShape(Rectangle())
          .onTapGesture { rotate() }
          .accessibilityElement()
          .accessibilityLabel("Trimmed puzzle photo")
          .accessibilityHint("Double tap to rotate 90 degrees clockwise.")
          .accessibilityAddTraits(.isButton)
          .accessibilityAction { rotate() }
          .task { await playTapHint() }
      } else {
        ContentUnavailableView("Could not decode the trimmed image", systemImage: "xmark.octagon")
      }
      Text("Detection confidence: \(Int(candidate.detection.confidence * 100))%")
        .font(.footnote)
        .foregroundStyle(.secondary)
      Spacer()
      if model.flow.isBusy {
        ProgressView("Uploading puzzle…")
      } else {
        HStack(spacing: 12) {
          Button {
            model.flow.errorMessage = nil
            model.flow.pendingRetake = candidate.source
            model.flow.phase = .capturePuzzle
          } label: {
            Label("Retake", systemImage: "arrow.counterclockwise")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.bordered)
          .controlSize(.large)

          Button {
            Task { await model.acceptTrim(candidate, quarterTurns: quarterTurns) }
          } label: {
            Label("Use This", systemImage: "checkmark")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.borderedProminent)
          .controlSize(.large)
          // Gate on the cached preview (not the base64-decoding
          // `trimmedJPEG` computed property) so this doesn't re-decode
          // on every rotate tap and stays disabled exactly when the
          // preview can't be shown.
          .disabled(previewImage == nil)
        }
      }
      if let error = model.flow.errorMessage {
        Text(error)
          .font(.footnote)
          .foregroundStyle(.red)
          .multilineTextAlignment(.center)
      }
    }
    .padding(24)
  }

  /// One-shot "tap here to rotate" demo drawn over the photo: a tapping finger
  /// beside a rotate glyph that turns a quarter clockwise. Purely decorative —
  /// hidden from VoiceOver (which gets the accessibility hint instead) and
  /// non-interactive so it never swallows the tap it is advertising.
  private var tapHint: some View {
    ZStack(alignment: .bottomTrailing) {
      Image(systemName: "rotate.right")
        .font(.system(size: 52, weight: .semibold))
        .rotationEffect(.degrees(Double(hintTurns) * 90))
      Image(systemName: "hand.tap.fill")
        .font(.system(size: 26, weight: .semibold))
        .offset(x: 12, y: 10)
    }
    .foregroundStyle(.white)
    .padding(24)
    .background(.black.opacity(0.55), in: Circle())
    .opacity(hintOpacity)
    .allowsHitTesting(false)
    .accessibilityHidden(true)
  }

  private func playTapHint() async {
    guard previewImage != nil, !didPlayHint else { return }
    didPlayHint = true
    guard await pause(for: .milliseconds(500)) else { return }
    withAnimation(.easeIn(duration: 0.25)) { hintOpacity = 1 }
    guard await pause(for: .milliseconds(550)) else { return }
    withAnimation(reduceMotion ? nil : .snappy(duration: 0.45)) { hintTurns = 1 }
    guard await pause(for: .milliseconds(950)) else { return }
    withAnimation(.easeOut(duration: 0.45)) { hintOpacity = 0 }
  }

  /// Sleeps, reporting whether the wait finished — a cancelled `.task` (the
  /// view went away mid-demo) must stop the sequence rather than run the rest
  /// of its animation steps back to back.
  private func pause(for duration: Duration) async -> Bool {
    do {
      try await Task.sleep(for: duration)
      return true
    } catch {
      return false
    }
  }

  private func rotate() {
    guard !model.flow.isBusy else { return }
    // The demo has served its purpose once the user taps; leaving it to finish
    // would have it spinning over an image that is already spinning.
    withAnimation(.easeOut(duration: 0.2)) { hintOpacity = 0 }
    // Always increment (never wrap to 0) so the animation turns clockwise on
    // every tap instead of unwinding 270° → 0°.
    withAnimation(.snappy(duration: 0.35)) {
      quarterTurns += 1
    }
  }
}
