import SwiftUI

/// PIECE_SIZE_RATIO in puzzle-detail.tsx.
private let pieceSizeRatio: CGFloat = 0.12

/// The trimmed puzzle image with each predicted piece drawn at its normalized
/// position. Mirrors frontend/src/components/puzzle/puzzle-detail.tsx:
/// fixed piece size (12% of image dimensions), counter-clockwise rotation,
/// confidence bar along the bottom edge.
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
                                PieceMarker(entry: entry, piece: piece, canvas: geo.size)
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

    var body: some View {
        let width = canvas.width * pieceSizeRatio
        let height = canvas.height * pieceSizeRatio
        VStack(spacing: 0) {
            if let image = UIImage(data: entry.displayImage) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            ConfidenceBar(value: piece.positionConfidence)
                .frame(height: 4)
        }
        .frame(width: width, height: height)
        .background(.black.opacity(0.25))
        .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(.white, lineWidth: 1.5))
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
