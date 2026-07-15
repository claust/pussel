import PhotosUI
import SwiftUI

struct CapturePuzzleView: View {
    @Environment(AppModel.self) private var model
    @State private var showCamera = false
    @State private var showLibrary = false
    @State private var photoItem: PhotosPickerItem?

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "puzzlepiece.extension.fill")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
            Text("Photograph the puzzle")
                .font(.title2.bold())
            Text("Take a straight-on photo of the finished puzzle picture — the box front works well.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Spacer()
            if model.flow.isBusy {
                ProgressView("Detecting puzzle…")
            } else {
                VStack(spacing: 12) {
                    if CameraPicker.isAvailable {
                        Button {
                            showCamera = true
                        } label: {
                            Label("Take Puzzle Photo", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                    }
                    if CameraPicker.isAvailable {
                        photoLibraryButton.buttonStyle(.bordered)
                    } else {
                        photoLibraryButton.buttonStyle(.borderedProminent)
                    }
                }
            }
            if let error = model.flow.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }
            Spacer()
        }
        .padding(24)
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { image in
                Task { await handle(image: image, source: .camera) }
            }
            .ignoresSafeArea()
        }
        .photosPicker(isPresented: $showLibrary, selection: $photoItem, matching: .images)
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    await handle(image: image, source: .library)
                } else {
                    model.flow.errorMessage = "Could not load the selected photo."
                }
                photoItem = nil
            }
        }
        .onAppear(perform: reopenPickerIfRetaking)
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
    /// original photo instead of making the user pick a source again.
    private func reopenPickerIfRetaking() {
        guard let source = model.flow.pendingRetake else { return }
        model.flow.pendingRetake = nil
        switch source {
        case .camera:
            if CameraPicker.isAvailable {
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
}
