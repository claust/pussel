import SwiftUI

/// Horizontal strip of captured pieces with their prediction status,
/// mirroring frontend/src/components/puzzle/piece-queue.tsx.
struct PieceQueueView: View {
  @Environment(AppModel.self) private var model
  let session: SolveSession

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Pieces (\(session.entries.count))")
        .font(.headline)
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 12) {
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
  }
}

private struct QueueTile: View {
  let entry: CaptureEntry
  let onRetry: () -> Void
  let onDelete: () -> Void

  var body: some View {
    VStack(spacing: 6) {
      ZStack {
        if let image = UIImage(data: entry.displayImage) {
          Image(uiImage: image)
            .resizable()
            .scaledToFill()
            .frame(width: 84, height: 84)
            .clipShape(RoundedRectangle(cornerRadius: 10))
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
