import SwiftUI

/// The trimmed puzzle image with each predicted piece drawn at its normalized
/// position. Mirrors frontend/src/components/puzzle/puzzle-detail.tsx, except
/// marker size is derived from the session's estimated grid (rows/cols)
/// rather than a fixed ratio of the image, so pieces mismatched to the actual
/// grid density scale correctly.
///
/// Tapping opens `PuzzleZoomView`, where the same markers are drawn over a
/// sharper copy of the puzzle at whatever magnification the user pinches to.
struct PuzzleOverlayView: View {
  let session: SolveSession
  /// Invoked when the puzzle is tapped, to open the zoom viewer.
  var onTap: () -> Void = {}

  var body: some View {
    if let image = UIImage(data: session.trimmedJPEG) {
      Image(uiImage: image)
        .resizable()
        .scaledToFit()
        .overlay(Color.black.opacity(0.3))
        .overlay {
          PieceMarkerLayer(session: session)
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .contentShape(RoundedRectangle(cornerRadius: 12))
        .onTapGesture(perform: onTap)
        .accessibilityElement()
        .accessibilityLabel("Puzzle with \(session.placedEntries.count) placed pieces")
        .accessibilityHint("Opens the puzzle in a zoomable full-screen view.")
        .accessibilityAddTraits(.isButton)
        .accessibilityAction(.default, onTap)
    }
  }
}

/// Every placed piece drawn over whatever space it's given, positioned in the
/// puzzle's normalized coordinates. Shared by the inline overlay and the zoom
/// viewer: it takes its canvas from the geometry it lands in, so the same
/// markers serve a 350pt-wide thumbnail and a pinched-in close-up.
struct PieceMarkerLayer: View {
  let session: SolveSession
  /// When set, every other marker is dimmed back so this one reads at a
  /// glance — the reason the viewer was opened.
  var focusedPieceID: UUID?

  /// A placed piece with its image already decoded.
  private struct Marker: Identifiable {
    let id: UUID
    let piece: PieceResponse
    let image: UIImage?
  }

  private var decodedMarkers: [Marker] {
    session.placedEntries.compactMap { entry in
      guard let piece = entry.result else { return nil }
      // The span (when present) describes the FULL displayed image frame,
      // including the backend's transparent margin, so don't trim it there.
      // In the null-span fallback, trim the margin from both display and
      // aspect math so it doesn't inflate the marker.
      let data = piece.pieceSpan == nil ? entry.trimmedDisplayImage : entry.displayImage
      return Marker(id: entry.id, piece: piece, image: UIImage(data: data))
    }
  }

  var body: some View {
    // Bound here, outside the GeometryReader, on purpose: the closure below
    // re-runs on every layout, and the zoom viewer relays this layer out on
    // every pan and zoom frame. Decoding out here happens once per change to
    // the pieces instead — the difference between a smooth pinch and a
    // stuttering one.
    let markers = decodedMarkers
    GeometryReader { geo in
      ForEach(markers) { marker in
        PieceMarker(
          piece: marker.piece,
          image: marker.image,
          canvas: geo.size,
          session: session,
          isDimmed: focusedPieceID != nil && marker.id != focusedPieceID
        )
      }
    }
  }
}

private struct PieceMarker: View {
  let piece: PieceResponse
  /// Already decoded by `PieceMarkerLayer`, which knows which of the piece's
  /// images to draw; nil when those bytes wouldn't decode.
  let image: UIImage?
  let canvas: CGSize
  let session: SolveSession
  var isDimmed = false

  var body: some View {
    let cellSize = CGSize(
      width: canvas.width / CGFloat(session.cols), height: canvas.height / CGFloat(session.rows))
    let markerSize = PieceMarkerGeometry.size(
      span: piece.pieceSpan,
      imageSize: image?.size ?? cellSize,
      canvas: canvas,
      cellSize: cellSize,
      rotation: piece.rotation
    )
    Group {
      if let image {
        Image(uiImage: image)
          .resizable()
          .scaledToFit()
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }
    .frame(width: markerSize.width, height: markerSize.height)
    .background(.black.opacity(0.25))
    // Chrome is sized in points, so it stays put as the canvas grows under
    // zoom rather than swelling into a band across the piece it outlines.
    .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(.white, lineWidth: 1.5))
    .overlay(alignment: .bottom) {
      ConfidenceBar(value: piece.positionConfidence)
        .frame(height: 4)
    }
    .rotationEffect(.degrees(-Double(piece.rotation)))
    .position(x: canvas.width * piece.position.x, y: canvas.height * piece.position.y)
    .opacity(isDimmed ? 0.25 : 1)
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
