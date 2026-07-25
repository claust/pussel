import PhotosUI
import SwiftUI

/// The bits of the list's scroll geometry the offset restore needs.
private struct ScrollSnapshot: Equatable {
  let offset: CGFloat
  let insetTop: CGFloat

  /// How far the list has been scrolled from its resting top, which is what
  /// `ScrollPosition.scrollTo(y:)` takes — `contentOffset` is measured from
  /// under the navigation bar instead, so it sits one top inset lower.
  ///
  /// Clamped at 0: a rubber-band overscroll at the top reads as negative, and
  /// leaving the screen mid-bounce would otherwise store a position that only
  /// exists while a finger is on the glass.
  var scrolledY: CGFloat { max(0, offset + insetTop) }
}

struct CapturePuzzleView: View {
  @Environment(AppModel.self) private var model
  @State private var showCamera = false
  @State private var showLibrary = false
  @State private var photoItem: PhotosPickerItem?
  /// Drives the restore of `flow.homeScrollOffset` — see `body`.
  @State private var scrollPosition = ScrollPosition()
  /// False until `restoreScrollOffset` has run. A freshly built `ScrollView`
  /// reports its geometry (offset 0) before `onAppear`, which would overwrite
  /// the very offset we came back to restore.
  @State private var hasRestoredScroll = false

  private var hasSavedPuzzles: Bool {
    !model.store.puzzles.isEmpty
  }

  var body: some View {
    ScrollView {
      if hasSavedPuzzles {
        content
      } else {
        // Fill the viewport so the hero stays vertically centered like
        // the original single-screen capture prompt.
        content.containerRelativeFrame(.vertical, alignment: .center) { height, _ in height }
      }
    }
    .scrollBounceBehavior(.basedOnSize)
    // This screen is rebuilt from scratch every time the wizard comes back to
    // it (see `AppFlowStore.homeScrollOffset`), so the offset is remembered
    // out here and put back on appear — otherwise reopening a puzzle from far
    // down the list always returns the user to the top.
    .scrollPosition($scrollPosition)
    .onScrollGeometryChange(for: ScrollSnapshot.self) { geometry in
      ScrollSnapshot(offset: geometry.contentOffset.y, insetTop: geometry.contentInsets.top)
    } action: { _, snapshot in
      guard hasRestoredScroll else {
        restoreScrollOffset(snapshot)
        return
      }
      model.flow.homeScrollOffset = snapshot.scrolledY
    }
    .overlay(alignment: .bottom) { UndoDeleteSnackbar() }
    .animation(.snappy, value: model.store.pendingDelete)
    .fullScreenCover(isPresented: cameraCoverIsPresented) {
      GlareFreeCaptureView(
        onImage: { image in
          Task { await handle(image: image, source: .camera) }
        },
        onBarcodeJPEG: { jpeg, pieceCountEstimate in
          model.startTrimFromBarcodeLookup(jpeg: jpeg, pieceCountEstimate: pieceCountEstimate)
        }
      )
    }
    .photosPicker(isPresented: $showLibrary, selection: $photoItem, matching: .images)
    .onChange(of: photoItem) { _, item in
      guard let item else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self),
          let image = UIImage(data: data)
        {
          await handle(image: image, source: .library)
        } else {
          model.flow.errorMessage = "Could not load the selected photo."
        }
        photoItem = nil
      }
    }
    .onAppear(perform: reopenPickerIfRetaking)
  }

  /// Puts the list back where it was before the wizard left this screen.
  ///
  /// Waits for the navigation bar's inset to land: a fresh `ScrollView` reports
  /// its geometry once before the bar is accounted for, and scrolling in that
  /// pass is undone (and re-resolved a top inset off) when the inset arrives.
  private func restoreScrollOffset(_ snapshot: ScrollSnapshot) {
    guard snapshot.insetTop > 0 else { return }
    hasRestoredScroll = true
    guard let scrolledY = model.flow.homeScrollOffset else { return }
    scrollPosition.scrollTo(y: scrolledY)
  }

  private var content: some View {
    VStack(spacing: 20) {
      // Compact header once there are saved puzzles to keep below it;
      // otherwise a taller hero centered in the screen.
      Spacer(minLength: hasSavedPuzzles ? 8 : 40)
      Image(systemName: "puzzlepiece.extension.fill")
        .font(.system(size: hasSavedPuzzles ? 40 : 56))
        .foregroundStyle(.tint)
      Text("Photograph the puzzle")
        .font(.title2.bold())
      Text(
        "Point the camera at the puzzle box — a Ravensburger barcode is looked up "
          + "automatically, or tap the shutter to photograph the picture."
      )
      .multilineTextAlignment(.center)
      .foregroundStyle(.secondary)
      if model.flow.isBusy {
        ProgressView("Detecting puzzle…")
          .padding(.top, 8)
      } else {
        VStack(spacing: 12) {
          if BoxCameraSession.isCameraAvailable {
            Button {
              showCamera = true
            } label: {
              Label("Take Puzzle Photo", systemImage: "camera.fill")
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            photoLibraryButton.buttonStyle(.bordered)
          } else {
            photoLibraryButton.buttonStyle(.borderedProminent)
          }
        }
        .padding(.top, 8)
      }
      if let error = model.flow.errorMessage {
        Text(error)
          .font(.footnote)
          .foregroundStyle(.red)
          .multilineTextAlignment(.center)
      }
      if hasSavedPuzzles {
        Divider().padding(.vertical, 4)
        SavedPuzzlesSection()
      }
      Spacer(minLength: hasSavedPuzzles ? 8 : 40)
    }
    .padding(24)
    .frame(maxWidth: .infinity)
  }

  private var photoLibraryButton: some View {
    Button {
      showLibrary = true
    } label: {
      Label("Choose from Library", systemImage: "photo.on.rectangle")
        .frame(maxWidth: .infinity)
    }
    .controlSize(.large)
  }

  /// After "Retake", jump straight back into whichever picker produced the
  /// original photo instead of making the user pick a source again. A
  /// barcode-resolved image counts as a camera capture: the capture screen
  /// is where both the barcode and the shutter live.
  private func reopenPickerIfRetaking() {
    guard let source = model.flow.pendingRetake else { return }
    model.flow.pendingRetake = nil
    switch source {
    case .camera, .barcodeLookup:
      if BoxCameraSession.isCameraAvailable {
        showCamera = true
      } else {
        showLibrary = true
      }
    case .library:
      showLibrary = true
    }
  }

  private func handle(image: UIImage, source: CaptureSource) async {
    await model.startTrim(image: image, source: source)
  }

  /// Also presented when `pusseldebug://camera` (or its `boxcamera` /
  /// `glarecamera` aliases) sets `flow.debugCaptureCameraOpen`, so the
  /// capture flow is drivable on the Simulator (which has no camera, so
  /// `showCamera` alone never becomes reachable there) — mirrors
  /// `PieceQueueView.cameraCoverIsPresented`.
  private var cameraCoverIsPresented: Binding<Bool> {
    #if DEBUG
      Binding(
        get: { showCamera || model.flow.debugCaptureCameraOpen },
        set: { newValue in
          showCamera = newValue
          model.flow.debugCaptureCameraOpen = newValue
        }
      )
    #else
      $showCamera
    #endif
  }
}
