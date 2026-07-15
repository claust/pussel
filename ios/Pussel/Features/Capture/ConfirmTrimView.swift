import SwiftUI

struct ConfirmTrimView: View {
    /// Mirrors LOW_CONFIDENCE_THRESHOLD in frontend/src/app/real/page.tsx.
    private static let lowConfidence = 0.4

    @Environment(AppModel.self) private var model
    @State private var quarterTurns = 0
    let candidate: TrimCandidate

    var body: some View {
        VStack(spacing: 16) {
            Text("Does this look right?")
                .font(.title2.bold())
            if candidate.detection.confidence < Self.lowConfidence {
                Label("Detection looks uncertain — consider retaking.", systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
            if let data = candidate.trimmedJPEG, let image = UIImage(data: data) {
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
                Button {
                    // Always increment (never wrap to 0) so the animation turns
                    // clockwise on every tap instead of unwinding 270° → 0°.
                    withAnimation(.snappy(duration: 0.35)) {
                        quarterTurns += 1
                    }
                } label: {
                    Label("Rotate", systemImage: "rotate.right")
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                .disabled(model.flow.isBusy)
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
