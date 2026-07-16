import PhotosUI
import SwiftUI

/// Horizontal strip of captured pieces with their prediction status, led by an
/// "add piece" tile that opens the camera full screen. Mirrors
/// frontend/src/components/puzzle/piece-queue.tsx.
struct PieceQueueView: View {
  @Environment(AppModel.self) private var model
  let session: SolveSession
  @State private var showCamera = false
  @State private var showLibrary = false
  @State private var photoItem: PhotosPickerItem?

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Pieces (\(session.entries.count))")
        .font(.headline)
      ScrollView(.horizontal, showsIndicators: false) {
        // Top-aligned so the plus square lines up with the piece thumbnails
        // rather than centering against their taller status labels.
        HStack(alignment: .top, spacing: 12) {
          AddPieceTile(action: addPiece)
          ForEach(session.entries) { entry in
            QueueTile(
              entry: entry,
              onRetry: { session.retry(id: entry.id, api: model.api) },
              onDelete: { session.remove(id: entry.id) }
            )
          }
        }
        .padding(.vertical, 2)
      }
    }
    .fullScreenCover(isPresented: $showCamera) {
      PieceCaptureView()
    }
    // Simulator path: no camera, so the tile picks from the library directly.
    .photosPicker(isPresented: $showLibrary, selection: $photoItem, matching: .images)
    .onChange(of: photoItem) { _, item in
      guard let item else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self),
          let image = UIImage(data: data)
        {
          model.addPiece(image: image)
        }
        photoItem = nil
      }
    }
  }

  private func addPiece() {
    if PieceCameraSession.isCameraAvailable {
      showCamera = true
    } else {
      showLibrary = true
    }
  }
}

/// Leading tile in the strip: a big plus that starts a new piece capture.
private struct AddPieceTile: View {
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      RoundedRectangle(cornerRadius: 10)
        .strokeBorder(.tint, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
        .frame(width: 84, height: 84)
        .overlay {
          Image(systemName: "plus")
            .font(.system(size: 34, weight: .light))
            .foregroundStyle(.tint)
        }
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Add piece")
  }
}

private struct QueueTile: View {
  let entry: CaptureEntry
  let onRetry: () -> Void
  let onDelete: () -> Void

  /// The model reports how far the piece is turned from its place in the
  /// puzzle, so undoing it shows the piece as it will sit there — the same
  /// counter-clockwise correction PuzzleOverlayView applies to its markers.
  /// Normalized to (-180, 180] so the tile animates the short way round
  /// rather than spinning three quarter turns to reach the same place.
  private var uprightRotation: Double {
    guard let piece = entry.result else { return 0 }
    let degrees = (360 - piece.rotation % 360) % 360
    return Double(degrees > 180 ? degrees - 360 : degrees)
  }

  var body: some View {
    VStack(spacing: 6) {
      ZStack {
        if let image = UIImage(data: entry.displayImage) {
          Image(uiImage: image)
            .resizable()
            .scaledToFill()
            .frame(width: 84, height: 84)
            .rotationEffect(.degrees(uprightRotation))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .animation(.easeInOut(duration: 0.25), value: uprightRotation)
        }
        if entry.status == .predicting {
          RoundedRectangle(cornerRadius: 10)
            .fill(.black.opacity(0.4))
            .frame(width: 84, height: 84)
          ProgressView()
            .tint(.white)
        }
      }
      statusLabel
      HStack(spacing: 10) {
        if entry.status.isRetryable {
          Button(action: onRetry) {
            Image(systemName: "arrow.clockwise")
          }
          .accessibilityLabel("Retry piece")
        }
        Button(role: .destructive, action: onDelete) {
          Image(systemName: "trash")
        }
        .accessibilityLabel("Delete piece")
      }
      .font(.footnote)
      .buttonStyle(.borderless)
    }
  }

  @ViewBuilder
  private var statusLabel: some View {
    switch entry.status {
    case .queued:
      Text("Queued").font(.caption2).foregroundStyle(.secondary)
    case .predicting:
      Text("Predicting…").font(.caption2).foregroundStyle(.secondary)
    case .done:
      if let piece = entry.result {
        Text("(\(Int(piece.position.x * 100))%, \(Int(piece.position.y * 100))%)")
          .font(.caption2.monospacedDigit())
          .foregroundStyle(.green)
      }
    case .expired:
      Text("Puzzle session expired")
        .font(.caption2)
        .foregroundStyle(.red)
        .lineLimit(1)
        .frame(maxWidth: 96)
    case .error(let message):
      Text(message)
        .font(.caption2)
        .foregroundStyle(.red)
        .lineLimit(1)
        .frame(maxWidth: 96)
    }
  }
}
