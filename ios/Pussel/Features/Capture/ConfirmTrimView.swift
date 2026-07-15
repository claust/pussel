import SwiftUI

struct ConfirmTrimView: View {
  /// Mirrors LOW_CONFIDENCE_THRESHOLD in frontend/src/app/real/page.tsx.
  private static let lowConfidence = 0.4

  @Environment(AppModel.self) private var model
  let candidate: TrimCandidate

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
      if let data = candidate.trimmedJPEG, let image = UIImage(data: data) {
        Image(uiImage: image)
          .resizable()
          .scaledToFit()
          .clipShape(RoundedRectangle(cornerRadius: 12))
          .frame(maxHeight: 420)
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
            Task { await model.acceptTrim(candidate) }
          } label: {
            Label("Use This", systemImage: "checkmark")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.borderedProminent)
          .controlSize(.large)
          .disabled(candidate.trimmedJPEG == nil)
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
}
