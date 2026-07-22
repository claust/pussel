import PhotosUI
import SwiftUI

struct CapturePuzzleView: View {
  @Environment(AppModel.self) private var model
  @State private var showCamera = false
  @State private var showGlareCamera = false
  @State private var showLibrary = false
  @State private var photoItem: PhotosPickerItem?

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
    .overlay(alignment: .bottom) { UndoDeleteSnackbar() }
    .animation(.snappy, value: model.store.pendingDelete)
    .fullScreenCover(isPresented: cameraCoverIsPresented) {
      BoxCameraView(
        onImage: { image in
          Task { await handle(image: image, source: .camera) }
        },
        onBarcodeJPEG: { jpeg, pieceCountEstimate in
          model.startTrimFromBarcodeLookup(jpeg: jpeg, pieceCountEstimate: pieceCountEstimate)
        }
      )
    }
    .fullScreenCover(isPresented: glareCoverIsPresented) {
      GlareFreeCaptureView(
        onImage: { image in
          Task { await handle(image: image, source: .camera) }
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
          }
          if BoxCameraSession.isCameraAvailable {
            // Experimental multi-shot capture for glossy puzzles — see
            // GlareFreeCaptureView.
            Button {
              showGlareCamera = true
            } label: {
              Label("Glare-Free Scan", systemImage: "sparkles")
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
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
  /// barcode-resolved image counts as a camera capture: the live box camera
  /// is where both the barcode and the manual shutter live.
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

  /// Also presented when `pusseldebug://boxcamera` sets
  /// `flow.debugBoxCameraOpen`, so the barcode capture flow is drivable on
  /// the Simulator (which has no camera, so `showCamera` alone never becomes
  /// reachable there) — mirrors `PieceQueueView.cameraCoverIsPresented`.
  private var cameraCoverIsPresented: Binding<Bool> {
    #if DEBUG
      Binding(
        get: { showCamera || model.flow.debugBoxCameraOpen },
        set: { newValue in
          showCamera = newValue
          model.flow.debugBoxCameraOpen = newValue
        }
      )
    #else
      $showCamera
    #endif
  }

  /// Mirrors `cameraCoverIsPresented` for the glare-free capture cover,
  /// driven on the Simulator by `pusseldebug://glarecamera`.
  private var glareCoverIsPresented: Binding<Bool> {
    #if DEBUG
      Binding(
        get: { showGlareCamera || model.flow.debugGlareCameraOpen },
        set: { newValue in
          showGlareCamera = newValue
          model.flow.debugGlareCameraOpen = newValue
        }
      )
    #else
      $showGlareCamera
    #endif
  }
}
