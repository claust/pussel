import SwiftUI

struct SolvingView: View {
  @Environment(AppModel.self) private var model
  let session: SolveSession
  @State private var isReuploading = false
  /// Non-nil while the zoom viewer is up; carries what it opened on.
  @State private var zoomFocus: PuzzleZoomFocus?

  var body: some View {
    ScrollView {
      VStack(spacing: 16) {
        if session.puzzleExpired {
          expiredBanner
        }
        PuzzleOverlayView(session: session) {
          zoomFocus = PuzzleZoomFocus(pieceID: nil)
        }
        PieceQueueView(session: session) { pieceID in
          zoomFocus = PuzzleZoomFocus(pieceID: pieceID)
        }
        if let error = session.errorMessage {
          Text(error)
            .font(.footnote)
            .foregroundStyle(.red)
        }
      }
      .padding()
    }
    // No `.toolbar` here: the back chevron belongs to `AppFlowView`, which
    // keeps it mounted across the phase switch so a tap right after this
    // screen appears can't land before the item does (see its `backButton`).
    .fullScreenCover(item: $zoomFocus) { focus in
      PuzzleZoomView(session: session, focus: focus)
    }
  }

  /// Shown when the backend restarted and forgot the puzzle_id — the kept
  /// trimmed image can be re-uploaded for a fresh id without re-shooting.
  private var expiredBanner: some View {
    VStack(spacing: 8) {
      Label(
        "This puzzle session expired on the server.", systemImage: "exclamationmark.triangle.fill"
      )
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
