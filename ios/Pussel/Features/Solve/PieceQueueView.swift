import SwiftUI

/// Horizontal strip of captured pieces with their prediction status,
/// mirroring frontend/src/components/puzzle/piece-queue.tsx.
struct PieceQueueView: View {
  @Environment(AppModel.self) private var model
  let session: SolveSession
  @State private var isDeleteMode = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("Pieces (\(session.entries.count))")
          .font(.headline)
        Spacer()
        if isDeleteMode {
          Button("Done") {
            withAnimation { isDeleteMode = false }
          }
          .font(.subheadline.weight(.semibold))
        }
      }
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 12) {
          ForEach(session.entries) { entry in
            QueueTile(
              entry: entry,
              isDeleteMode: isDeleteMode,
              onRetry: { session.retry(id: entry.id, api: model.api) },
              onDelete: { withAnimation { session.remove(id: entry.id) } },
              onEnterDeleteMode: { withAnimation { isDeleteMode = true } },
              onExitDeleteMode: { withAnimation { isDeleteMode = false } }
            )
          }
        }
        .padding(.vertical, 2)
        // Room for the wiggle and the delete badge to spill outside the tiles.
        .padding(.horizontal, 6)
        .padding(.top, 6)
      }
      .padding(.horizontal, -6)
      .padding(.top, -6)
    }
    // No need to leave delete mode when the last piece goes: SolvingView drops
    // this view once `entries` is empty, which resets `isDeleteMode` with it.
  }
}

private struct QueueTile: View {
  let entry: CaptureEntry
  let isDeleteMode: Bool
  let onRetry: () -> Void
  let onDelete: () -> Void
  let onEnterDeleteMode: () -> Void
  let onExitDeleteMode: () -> Void

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
      .overlay(alignment: .topTrailing) {
        if isDeleteMode {
          deleteBadge
            .offset(x: 8, y: -8)
            .transition(.scale.combined(with: .opacity))
        }
      }
      statusLabel
      if entry.status.isRetryable {
        Button(action: onRetry) {
          Image(systemName: "arrow.clockwise")
        }
        .accessibilityLabel("Retry piece")
        .font(.footnote)
        .buttonStyle(.borderless)
      }
    }
    // Seeded off the piece itself, not its position, so deleting a tile doesn't
    // reshuffle how its neighbours rock.
    .wiggle(isActive: isDeleteMode, seed: Int(entry.id.uuid.0))
    .contentShape(Rectangle())
    .onTapGesture {
      if isDeleteMode { onExitDeleteMode() }
    }
    // Simultaneous, because a plain `.onLongPressGesture` alongside
    // `.onTapGesture` loses the race and never fires.
    .simultaneousGesture(
      LongPressGesture(minimumDuration: 0.45).onEnded { _ in
        guard !isDeleteMode else { return }
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        onEnterDeleteMode()
      }
    )
  }

  private var deleteBadge: some View {
    Button(action: onDelete) {
      Image(systemName: "xmark")
        .font(.system(size: 10, weight: .bold))
        .foregroundStyle(.white)
        .frame(width: 22, height: 22)
        .background(Circle().fill(.black.opacity(0.7)))
        .overlay(Circle().stroke(.white.opacity(0.9), lineWidth: 1))
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Delete piece")
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

extension View {
  /// Home-screen style jiggle. `seed` desynchronises neighbouring tiles so the
  /// strip doesn't rock in unison.
  fileprivate func wiggle(isActive: Bool, seed: Int) -> some View {
    modifier(WiggleModifier(isActive: isActive, seed: seed))
  }
}

private struct WiggleModifier: ViewModifier {
  let isActive: Bool
  let seed: Int
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  private var enabled: Bool { isActive && !reduceMotion }

  /// Slight per-tile variation, mirroring how the home screen's icons drift
  /// out of phase with each other.
  private var angle: Double { seed.isMultiple(of: 2) ? 1.7 : 1.4 }
  private var duration: Double { 0.13 + Double(seed % 3) * 0.012 }

  // PhaseAnimator owns its own looping animation, so deleting a piece can't
  // cancel the jiggle: a plain `.repeatForever` on `rotationEffect` is
  // overridden by the ambient `withAnimation` transaction that removal runs in,
  // and nothing restarts it for the tiles that survive.
  @ViewBuilder
  func body(content: Content) -> some View {
    if enabled {
      content.phaseAnimator([false, true]) { view, isRocked in
        view.rotationEffect(.degrees(isRocked ? angle : -angle))
      } animation: { _ in
        .easeInOut(duration: duration)
      }
    } else {
      content
    }
  }
}
