import PhotosUI
import SwiftUI
import UIKit

/// Wrapping grid of captured pieces with their prediction status, led by an
/// "add piece" tile that opens the camera full screen. Mirrors
/// frontend/src/components/puzzle/piece-queue.tsx.
struct PieceQueueView: View {
  @Environment(AppModel.self) private var model
  let session: SolveSession
  /// Invoked with a predicted piece's id when its tile is tapped, to open the
  /// zoom viewer on where that piece was placed.
  var onZoomToPiece: (UUID) -> Void = { _ in }
  @State private var isDeleteMode = false
  @State private var showCamera = false
  @State private var showScan = false
  @State private var showLibrary = false
  @State private var photoItem: PhotosPickerItem?

  /// Tiles take their column's full width rather than a fixed 84pt: pinning the
  /// maximum to 84 leaves the row's slack as wide gaps between the columns, which
  /// pushes the newest piece away from the plus it was captured with.
  private static let columns = [
    GridItem(.adaptive(minimum: 84), spacing: 12, alignment: .top)
  ]

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
      // Top-aligned cells so the plus square lines up with the piece thumbnails
      // rather than centering against their taller status labels.
      LazyVGrid(columns: Self.columns, alignment: .leading, spacing: 12) {
        AddPieceTile(action: addPiece)
        // Omit the tile entirely when it can't be used, rather than rendering
        // it invisible — an .opacity(0) tile still takes taps and is announced
        // by VoiceOver. Always present in DEBUG (the Simulator scan demo opens
        // it over a black preview); in release only when there's a camera.
        if showScanTile {
          ScanPiecesTile(action: openScan)
        }
        // Newest first, so a piece appears next to the plus that captured it
        // and the grid ages away from there. The stored order stays oldest
        // first — the prediction queue works through it front to back.
        ForEach(session.entries.reversed()) { entry in
          QueueTile(
            entry: entry,
            isDeleteMode: isDeleteMode,
            onRetry: { session.retry(id: entry.id, api: model.api) },
            onDelete: { withAnimation { session.remove(id: entry.id) } },
            onEnterDeleteMode: { withAnimation { isDeleteMode = true } },
            onExitDeleteMode: { withAnimation { isDeleteMode = false } },
            onZoom: { onZoomToPiece(entry.id) }
          )
        }
      }
      .padding(.vertical, 2)
    }
    // The grid outlives the pieces — it always shows the add tile — so delete
    // mode has to be left explicitly once the last piece goes, or the Done
    // button strands itself above an empty grid.
    .onChange(of: session.entries.isEmpty) { _, isEmpty in
      if isEmpty { isDeleteMode = false }
    }
    .fullScreenCover(isPresented: cameraCoverIsPresented) {
      PieceCaptureView()
    }
    .fullScreenCover(isPresented: scanCoverIsPresented) {
      PieceScanView(session: session)
    }
    // Simulator path: no camera, so the tile picks from the library directly.
    .photosPicker(isPresented: $showLibrary, selection: $photoItem, matching: .images)
    .onChange(of: photoItem) { _, item in
      guard let item else { return }
      Task {
        await model.addPiece(from: item)
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

  /// Whether the scan-and-lock tile is shown. Always in DEBUG so the
  /// Simulator scan demo (`pusseldebug://scan`) has an entry point over its
  /// black preview; in release only when a camera exists to drive it, so it
  /// is never a dead, VoiceOver-discoverable control.
  private var showScanTile: Bool {
    #if DEBUG
      return true
    #else
      return PieceCameraSession.isCameraAvailable
    #endif
  }

  private func openScan() {
    // On a real device the camera is available; on the Simulator it's not,
    // but the DEBUG build still opens the scan view (over a black preview)
    // so `pusseldebug://scan` can demo the scan-and-lock flow there.
    #if DEBUG
      showScan = true
    #else
      if PieceCameraSession.isCameraAvailable {
        showScan = true
      }
    #endif
  }

  #if DEBUG
    /// Also presented when `pusseldebug://camera` sets
    /// `session.debugCameraOpen`, so M9's overlay is demoable on the
    /// Simulator (which has no camera, so `showCamera` alone never becomes
    /// reachable there).
    private var cameraCoverIsPresented: Binding<Bool> {
      Binding(
        get: { showCamera || session.debugCameraOpen },
        set: { newValue in
          showCamera = newValue
          session.debugCameraOpen = newValue
        }
      )
    }

    /// Also presented when `pusseldebug://scan` sets
    /// `session.debugScanOpen`, mirroring `cameraCoverIsPresented` above.
    private var scanCoverIsPresented: Binding<Bool> {
      Binding(
        get: { showScan || session.debugScanOpen },
        set: { newValue in
          showScan = newValue
          session.debugScanOpen = newValue
        }
      )
    }
  #else
    private var cameraCoverIsPresented: Binding<Bool> { $showCamera }
    private var scanCoverIsPresented: Binding<Bool> { $showScan }
  #endif
}

/// Second tile in the grid: opens the hands-free scan-and-lock flow.
private struct ScanPiecesTile: View {
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      RoundedRectangle(cornerRadius: 10)
        .strokeBorder(.tint, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
        .aspectRatio(1, contentMode: .fit)
        .overlay {
          Image(systemName: "dot.viewfinder")
            .font(.system(size: 28, weight: .light))
            .foregroundStyle(.tint)
        }
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Scan pieces")
  }
}

/// Leading tile in the grid: a big plus that starts a new piece capture.
private struct AddPieceTile: View {
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      RoundedRectangle(cornerRadius: 10)
        .strokeBorder(.tint, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
        .aspectRatio(1, contentMode: .fit)
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
  let isDeleteMode: Bool
  let onRetry: () -> Void
  let onDelete: () -> Void
  let onEnterDeleteMode: () -> Void
  let onExitDeleteMode: () -> Void
  let onZoom: () -> Void

  /// Diameter of the visible badge circle.
  private static let badgeCircleSize: CGFloat = 22
  /// Tappable box around the circle. Bigger than the circle, because 22pt is
  /// well under the 44pt guideline — but deliberately short of a full 44pt,
  /// which would turn much of the tile's corner into a delete zone for an
  /// action that has no undo.
  ///
  /// Note the *effective* target is smaller than this: hit testing is bounded
  /// by the tile's own frame, so the part of this box hanging outside the
  /// tile is drawn but never tapped. This buys the corner region inside the
  /// tile (~19pt square, up from ~14pt), and raising the number further only
  /// grows the untappable spill — a real 44pt target would need the badge
  /// re-parented outside the tile, not a bigger box here.
  private static let badgeHitSize: CGFloat = 32
  /// How far the circle's centre sits inside the tile's corner.
  private static let badgeCircleInset: CGFloat = 3
  /// Distance the badge box is pushed past the corner to land the circle there.
  /// The circle spills ~8pt outside the tile by design, which the grid's 12pt
  /// gutter absorbs without the badge landing on the neighbouring tile.
  private static let badgeCornerOffset = badgeHitSize / 2 - badgeCircleInset

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
      // A clear square sets the tile's size from the column it lands in; the
      // thumbnail then fills it and is clipped back to the square, so a piece
      // photo of any aspect ratio still tiles evenly with its neighbours.
      Color.clear
        .aspectRatio(1, contentMode: .fit)
        .overlay {
          if let image = UIImage(data: entry.displayImage) {
            Image(uiImage: image)
              .resizable()
              .scaledToFill()
              .rotationEffect(.degrees(uprightRotation))
              .animation(.easeInOut(duration: 0.25), value: uprightRotation)
          }
        }
        .overlay {
          if entry.status == .predicting {
            Color.black.opacity(0.4)
            ProgressView()
              .tint(.white)
          }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(alignment: .topTrailing) {
          if isDeleteMode {
            deleteBadge
              // The badge's box is `hitSize`, but only the circle inside it is
              // visible, so offset by half the box to park the circle's centre
              // `circleInset` in from the tile's corner.
              .offset(x: Self.badgeCornerOffset, y: -Self.badgeCornerOffset)
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
    // In delete mode a tap is the way out of it (the badges are the only
    // thing meant to act there), so zooming waits until the grid is calm.
    // A piece with no prediction yet has no place on the puzzle to zoom to.
    .onTapGesture {
      if isDeleteMode {
        onExitDeleteMode()
      } else if entry.result != nil {
        onZoom()
      }
    }
    // VoiceOver can't discover a tap gesture, so the zoom is published as an
    // action on the tile rather than left implicit. Withheld exactly where the
    // tap does something else (delete mode) or has nowhere to go (a piece with
    // no prediction yet).
    .accessibilityElement(children: .contain)
    .accessibilityActions {
      if !isDeleteMode, entry.result != nil {
        Button("See placement on puzzle", action: onZoom)
      }
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
    Button(role: .destructive, action: onDelete) {
      Image(systemName: "xmark")
        .font(.system(size: 10, weight: .bold))
        .foregroundStyle(.white)
        .frame(width: Self.badgeCircleSize, height: Self.badgeCircleSize)
        .background(Circle().fill(.black.opacity(0.7)))
        .overlay(Circle().stroke(.white.opacity(0.9), lineWidth: 1))
        .frame(width: Self.badgeHitSize, height: Self.badgeHitSize)
        .contentShape(Rectangle())
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
  /// grid doesn't rock in unison.
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
