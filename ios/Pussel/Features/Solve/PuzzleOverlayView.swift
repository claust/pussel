import SwiftUI

/// The trimmed puzzle image with each predicted piece drawn at its normalized
/// position. Mirrors frontend/src/components/puzzle/puzzle-detail.tsx, except
/// marker size is derived from the session's estimated grid (rows/cols)
/// rather than a fixed ratio of the image, so pieces mismatched to the actual
/// grid density scale correctly.
struct PuzzleOverlayView: View {
  let session: SolveSession

  var body: some View {
    if let image = UIImage(data: session.trimmedJPEG) {
      Image(uiImage: image)
        .resizable()
        .scaledToFit()
        .overlay(Color.black.opacity(0.3))
        .overlay {
          GeometryReader { geo in
            ForEach(session.placedEntries) { entry in
              if let piece = entry.result {
                PieceMarker(entry: entry, piece: piece, canvas: geo.size, session: session)
              }
            }
          }
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
  }
}

private struct PieceMarker: View {
  let entry: CaptureEntry
  let piece: PieceResponse
  let canvas: CGSize
  let session: SolveSession

  var body: some View {
    let cellSize = CGSize(
      width: canvas.width / CGFloat(session.cols), height: canvas.height / CGFloat(session.rows))
    // The span (when present) describes the FULL displayed image frame,
    // including the backend's transparent margin, so don't trim it there.
    // In the null-span fallback, trim the margin from both display and
    // aspect math so it doesn't inflate the marker.
    let displayData = piece.pieceSpan == nil ? entry.trimmedDisplayImage : entry.displayImage
    let displayImage = UIImage(data: displayData)
    let markerSize = PieceMarkerGeometry.size(
      span: piece.pieceSpan,
      imageSize: displayImage?.size ?? cellSize,
      canvas: canvas,
      cellSize: cellSize,
      rotation: piece.rotation
    )
    Group {
      if let displayImage {
        Image(uiImage: displayImage)
          .resizable()
          .scaledToFit()
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }
    .frame(width: markerSize.width, height: markerSize.height)
    .background(.black.opacity(0.25))
    .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(.white, lineWidth: 1.5))
    .overlay(alignment: .bottom) {
      ConfidenceBar(value: piece.positionConfidence)
        .frame(height: 4)
    }
    .rotationEffect(.degrees(-Double(piece.rotation)))
    .position(x: canvas.width * piece.position.x, y: canvas.height * piece.position.y)
  }
}

struct ConfidenceBar: View {
  /// 0...1
  let value: Double

  var body: some View {
    GeometryReader { geo in
      ZStack(alignment: .leading) {
        Rectangle().fill(.red.opacity(0.8))
        Rectangle()
          .fill(.green)
          .frame(width: geo.size.width * min(max(value, 0), 1))
      }
    }
  }
}
