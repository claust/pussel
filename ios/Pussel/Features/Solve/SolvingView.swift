import SwiftUI

struct SolvingView: View {
    @Environment(AppModel.self) private var model
    let session: SolveSession
    @State private var isReuploading = false

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                if session.puzzleExpired {
                    expiredBanner
                }
                PuzzleOverlayView(session: session)
                PieceCaptureView()
                if !session.entries.isEmpty {
                    PieceQueueView(session: session)
                }
                if let error = session.errorMessage {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
                Button(role: .destructive) {
                    model.flow.reset()
                } label: {
                    Label("New Puzzle", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(.bordered)
                .padding(.top, 8)
            }
            .padding()
        }
    }

    /// Shown when the backend restarted and forgot the puzzle_id — the kept
    /// trimmed image can be re-uploaded for a fresh id without re-shooting.
    private var expiredBanner: some View {
        VStack(spacing: 8) {
            Label("This puzzle session expired on the server.", systemImage: "exclamationmark.triangle.fill")
                .font(.footnote)
            if isReuploading {
                ProgressView()
            } else {
                Button("Re-upload and continue") {
                    Task {
                        isReuploading = true
                        await session.reupload(api: model.api)
                        isReuploading = false
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
    }
}
