import SwiftUI

/// The list of locally stored puzzles shown beneath the capture buttons on the
/// home screen. Tap a card to reopen it; long-press to delete.
struct SavedPuzzlesSection: View {
    @Environment(AppModel.self) private var model
    @State private var pendingDelete: PuzzleSummary?

    var body: some View {
        // Lazy so rows (and their thumbnail decoding) are built only as they
        // scroll into view as the library grows.
        LazyVStack(alignment: .leading, spacing: 12) {
            Text("Your puzzles")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            ForEach(model.store.puzzles) { puzzle in
                Button {
                    model.openPuzzle(puzzle.id)
                } label: {
                    SavedPuzzleRow(puzzle: puzzle)
                }
                .buttonStyle(.plain)
                .contextMenu {
                    Button(role: .destructive) {
                        pendingDelete = puzzle
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
        .confirmationDialog(
            "Delete this puzzle?",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            titleVisibility: .visible,
            presenting: pendingDelete
        ) { puzzle in
            Button("Delete", role: .destructive) {
                model.deletePuzzle(puzzle.id)
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { puzzle in
            Text("“\(puzzle.name)” and its pieces will be removed from this device.")
        }
    }
}

private struct SavedPuzzleRow: View {
    let puzzle: PuzzleSummary

    var body: some View {
        HStack(spacing: 12) {
            thumbnail
            VStack(alignment: .leading, spacing: 4) {
                Text(puzzle.name)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Image(systemName: "chevron.right")
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.tertiary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private var thumbnail: some View {
        Group {
            if let data = puzzle.thumbnail, let image = UIImage(data: data) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: 56, height: 56)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var subtitle: String {
        let pieces = puzzle.pieceCount == 1 ? "1 piece" : "\(puzzle.pieceCount) pieces"
        return "\(pieces) · \(puzzle.createdAt.formatted(date: .abbreviated, time: .shortened))"
    }
}
