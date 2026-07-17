import AVFoundation
import SwiftUI
import UIKit

/// Full-screen hands-free scan mode: holds the camera over a piece, waits for
/// the stability tracker to fire, and posts the geometry. No shutter — the
/// lock triggers automatically after the piece holds still for ≥1 s.
///
/// Camera lifecycle mirrors `PieceCaptureView` exactly: started in `.task`,
/// stopped in `.onDisappear`, with the same DEBUG Simulator no-camera fallback
/// so `pusseldebug://previewloop` can demo the overlay over a black background.
struct PieceScanView: View {
  @Environment(AppModel.self) private var model
  @Environment(\.dismiss) private var dismiss

  let session: SolveSession

  @State private var camera = PieceCameraSession()
  @State private var previewStreamer: PiecePreviewStreamer?
  @State private var controller: PieceScanController?

  /// Drives the brief green flash that acknowledges a successful lock.
  @State private var showLockFlash = false

  var body: some View {
    ZStack {
      Color.black.ignoresSafeArea()
      CameraPreview(
        session: camera.session,
        previewState: previewStreamer?.state ?? .none,
        updatedAt: previewStreamer?.updatedAt ?? .distantPast,
        frameSize: previewStreamer?.frameSize ?? .zero
      )
      .ignoresSafeArea()

      // Green flash on successful lock: a translucent overlay that appears
      // instantly and fades out, giving the user tactile feedback that the
      // capture fired without any button tap.
      if showLockFlash {
        Color.green.opacity(0.25)
          .ignoresSafeArea()
          .allowsHitTesting(false)
          .transition(.opacity)
      }
    }
    // Top bar: Done + hint label.
    .overlay(alignment: .top) {
      HStack {
        Button("Done") { dismiss() }
          .foregroundStyle(.white)
        Spacer()
        Text(hintText)
          .font(.subheadline.weight(.medium))
          .foregroundStyle(.white)
          .padding(.trailing, 4)
      }
      .padding(.horizontal)
      .padding(.vertical, 12)
    }
    // Verdict banner + bottom controls.
    .overlay(alignment: .bottom) {
      VStack(spacing: 12) {
        verdictBanner
        uncertainChip
        galleryStrip
      }
      .padding(.bottom, 24)
    }
    .task {
      let streamer = previewStreamer ?? PiecePreviewStreamer(api: model.api)
      previewStreamer = streamer
      camera.attachPreviewStreamer(streamer)

      let ctrl = PieceScanController(
        puzzleId: session.puzzleId,
        geometryClient: model.api,
        capture: { [weak camera] in
          guard let camera else { return nil }
          let image = await camera.capturePhoto()
          guard let image else { return nil }
          return ImageUtilities.normalizedJPEG(from: image, maxDimension: 1600, quality: 0.9)
        }
      )
      controller = ctrl
      #if DEBUG
        PieceScanController.debugActive = ctrl
      #endif
      // Pre-fill the gallery concurrently — the camera shouldn't wait behind
      // a network round-trip, and the strip appearing a beat later is fine.
      Task { await ctrl.loadEnrolled() }

      let started = await camera.start()
      guard !Task.isCancelled else { return }
      guard started else {
        #if DEBUG
          if !PieceCameraSession.isCameraAvailable {
            camera.setPreviewStreamingEnabled(true)
            return
          }
        #endif
        // Real device failure (permission denied, no device): report and close.
        model.reportPieceError("Pussel cannot use the camera. Check camera access in Settings.")
        dismiss()
        return
      }
      camera.setPreviewStreamingEnabled(true)
    }
    .onDisappear {
      camera.setPreviewStreamingEnabled(false)
      camera.stop()
      previewStreamer?.reset()
      #if DEBUG
        if PieceScanController.debugActive === controller {
          PieceScanController.debugActive = nil
        }
      #endif
    }
    .keepsScreenAwake()
    // Drive the controller from the preview stream.
    .onChange(of: previewStreamer?.updatedAt) { _, _ in
      guard let streamer = previewStreamer, let ctrl = controller else { return }
      ctrl.ingest(streamer.state, at: streamer.updatedAt)
    }
    // React to phase changes that need view-level side effects.
    .onChange(of: controller?.phase) { _, newPhase in
      guard case .verdict(let verdict) = newPhase else { return }
      if case .locked = verdict {
        triggerLockFlash()
      }
    }
  }

  // MARK: - Sub-views

  /// Hint shown in the top bar while the camera is live.
  private var hintText: String {
    switch controller?.phase {
    case .capturing: return "Locking…"
    default: return "Hold steady over a piece"
    }
  }

  /// Coloured capsule banner shown over the preview when a verdict arrives.
  @ViewBuilder
  private var verdictBanner: some View {
    if let phase = controller?.phase, case .verdict(let verdict) = phase {
      Group {
        switch verdict {
        case .locked(_, let edgeTypes):
          verdictCapsule(
            text: "Piece locked ✓ \(edgeGlyphs(edgeTypes))",
            background: .green
          )
        case .alreadyScanned:
          verdictCapsule(text: "Already scanned", background: .orange)
        case .unreadable:
          verdictCapsule(
            text: "Couldn't read the piece — adjust and hold steady", background: .gray)
        case .failure(let msg):
          verdictCapsule(text: msg, background: .red)
        case .uncertain:
          EmptyView()
        }
      }
      .transition(.scale(scale: 0.85).combined(with: .opacity))
    }
  }

  private func verdictCapsule(text: String, background: Color) -> some View {
    Text(text)
      .font(.subheadline.weight(.semibold))
      .foregroundStyle(.white)
      .padding(.horizontal, 20)
      .padding(.vertical, 10)
      .background(Capsule().fill(background))
      .shadow(radius: 4)
  }

  /// "Not sure — tap to add as new piece" chip, shown when the server returned
  /// an uncertain verdict and the user hasn't dismissed it yet.
  @ViewBuilder
  private var uncertainChip: some View {
    if controller?.pendingUncertainJPEG != nil {
      HStack(spacing: 8) {
        Button {
          Task { await controller?.confirmUncertainAsNew() }
        } label: {
          Text("Not sure — tap to add as new piece")
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
        }
        Button {
          controller?.dismissUncertainChip()
        } label: {
          Image(systemName: "xmark")
            .font(.caption.weight(.bold))
            .foregroundStyle(.white.opacity(0.8))
        }
        .accessibilityLabel("Dismiss")
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 10)
      .background(
        Capsule().fill(.ultraThinMaterial.opacity(0.9))
          .background(Capsule().fill(Color.secondary.opacity(0.5)))
      )
      .transition(.scale(scale: 0.9).combined(with: .opacity))
    }
  }

  /// Horizontal scrolling row of scanned pieces with edge-type badges.
  @ViewBuilder
  private var galleryStrip: some View {
    let pieces = controller?.gallery ?? []
    let matchedId = controller?.lastMatchedPieceId
    let isAlreadyScanned: Bool = {
      if case .verdict(.alreadyScanned) = controller?.phase { return true }
      return false
    }()

    VStack(alignment: .leading, spacing: 6) {
      if !pieces.isEmpty {
        Text("Scanned: \(pieces.count)")
          .font(.caption)
          .foregroundStyle(.white.opacity(0.7))
          .padding(.leading, 16)
        ScrollView(.horizontal, showsIndicators: false) {
          HStack(spacing: 10) {
            ForEach(pieces) { piece in
              GalleryItemView(
                piece: piece,
                isHighlighted: isAlreadyScanned && piece.pieceId == matchedId
              )
            }
          }
          .padding(.horizontal, 16)
          .padding(.vertical, 4)
        }
      }
    }
    .animation(.easeInOut(duration: 0.25), value: pieces.count)
  }

  // MARK: - Helpers

  /// Joins edge glyphs with the middle-dot separator used throughout the UI.
  private func edgeGlyphs(_ types: [GeometryEdgeType]) -> String {
    types.map(\.glyph).joined(separator: "·")
  }

  /// Shows the green lock-flash overlay for ~0.4 s.
  private func triggerLockFlash() {
    withAnimation(.easeIn(duration: 0.05)) { showLockFlash = true }
    Task {
      try? await Task.sleep(nanoseconds: 400_000_000)
      withAnimation(.easeOut(duration: 0.35)) { showLockFlash = false }
    }
  }
}

// MARK: - Gallery item view

/// 56 pt rounded square for one scanned piece: thumbnail or puzzle-piece
/// placeholder, with the edge-type glyphs underneath.
private struct GalleryItemView: View {
  let piece: ScannedPiece
  let isHighlighted: Bool

  private static let size: CGFloat = 56
  private static let cornerRadius: CGFloat = 10

  var body: some View {
    VStack(spacing: 4) {
      ZStack {
        RoundedRectangle(cornerRadius: Self.cornerRadius)
          .fill(Color.white.opacity(0.12))
          .frame(width: Self.size, height: Self.size)

        if let data = piece.thumbnailJPEG, let uiImage = UIImage(data: data) {
          Image(uiImage: uiImage)
            .resizable()
            .scaledToFill()
            .frame(width: Self.size, height: Self.size)
            .clipShape(RoundedRectangle(cornerRadius: Self.cornerRadius))
        } else {
          Image(systemName: "puzzlepiece")
            .font(.system(size: 24))
            .foregroundStyle(.white.opacity(0.5))
        }
      }
      .overlay {
        if isHighlighted {
          RoundedRectangle(cornerRadius: Self.cornerRadius)
            .strokeBorder(.orange, lineWidth: 2.5)
        }
      }
      // Pulse the orange border when highlighted: a brief scale animation
      // draws the eye to the matched piece without permanent styling.
      .scaleEffect(isHighlighted ? 1.06 : 1.0)
      .animation(.easeInOut(duration: 0.25), value: isHighlighted)

      if !piece.edgeTypes.isEmpty {
        Text(piece.edgeTypes.map(\.glyph).joined(separator: "·"))
          .font(.caption2.monospaced())
          .foregroundStyle(.white.opacity(0.7))
      }
    }
  }
}
