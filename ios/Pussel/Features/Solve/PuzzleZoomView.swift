import SwiftUI

/// What the zoom viewer opens on: the whole puzzle, or one piece brought up
/// close. Identifiable so it can drive a `fullScreenCover(item:)`.
struct PuzzleZoomFocus: Identifiable, Equatable {
  /// The piece to open centred on, or nil to open at fit.
  let pieceID: UUID?

  /// Stable stand-in for the whole-puzzle case, which has no piece to
  /// identify it.
  private static let wholePuzzleID = UUID()

  var id: UUID { pieceID ?? Self.wholePuzzleID }
}

/// Full-screen, pinch-zoomable view of the puzzle with its predicted pieces
/// drawn in place — the close look at where a piece landed that the inline
/// overlay on the solve screen is too small to give.
///
/// A separate screen rather than zoom in place: the solve screen scrolls
/// vertically, so panning a zoomed puzzle inside it would be a tug of war
/// between the two for every drag.
struct PuzzleZoomView: View {
  let session: SolveSession
  /// The piece to open on. Its marker is centred and the rest are dimmed;
  /// nil opens the whole puzzle at fit with every marker at full strength.
  let focus: PuzzleZoomFocus
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    ZStack(alignment: .topTrailing) {
      Color.black.ignoresSafeArea()
      if let image = UIImage(data: session.zoomJPEG) {
        // No dimming scrim here, unlike the inline overlay: at zoom the
        // artwork under the piece is exactly what you're comparing against.
        ZoomableImageView(image: image, focusRect: focusRect) {
          PieceMarkerLayer(session: session, focusedPieceID: focus.pieceID)
        }
        .ignoresSafeArea()
      } else {
        ContentUnavailableView("Could not load the puzzle image", systemImage: "photo")
          .foregroundStyle(.white)
      }
      closeButton
    }
  }

  private var closeButton: some View {
    Button {
      dismiss()
    } label: {
      Image(systemName: "xmark")
        .font(.system(size: 15, weight: .bold))
        .foregroundStyle(.white)
        .frame(width: 36, height: 36)
        .background(.black.opacity(0.55), in: Circle())
    }
    .buttonStyle(.plain)
    .padding(16)
    .accessibilityLabel("Close")
  }

  /// The region to open on, in the puzzle's normalized coordinates, or nil to
  /// open the whole puzzle at fit. See `PuzzleFocusGeometry`.
  private var focusRect: CGRect? {
    guard let pieceID = focus.pieceID,
      let piece = session.entries.first(where: { $0.id == pieceID })?.result
    else {
      return nil
    }
    return PuzzleFocusGeometry.focusRect(
      position: CGPoint(x: piece.position.x, y: piece.position.y),
      rows: session.rows,
      cols: session.cols)
  }
}
