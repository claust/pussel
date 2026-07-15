import PhotosUI
import SwiftUI

struct CapturePuzzleView: View {
    @Environment(AppModel.self) private var model
    @State private var showCamera = false
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
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { image in
                Task { await handle(image: image) }
            }
            .ignoresSafeArea()
        }
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    await handle(image: image)
                } else {
                    model.flow.errorMessage = "Could not load the selected photo."
                }
                photoItem = nil
            }
        }
    }

    private var content: some View {
        VStack(spacing: 20) {
            // Compact header once there are saved puzzles to keep below it;
            // otherwise a taller hero centered in the screen.
            Spacer(minLength: hasSavedPuzzles ? 8 : 40)
            Image(systemName: "photo.artframe")
                .font(.system(size: hasSavedPuzzles ? 40 : 56))
                .foregroundStyle(.tint)
            Text("Photograph the puzzle")
                .font(.title2.bold())
            Text("Take a straight-on photo of the finished puzzle picture — the box front works well.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            if model.flow.isBusy {
                ProgressView("Detecting puzzle…")
                    .padding(.top, 8)
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
        PhotosPicker(selection: $photoItem, matching: .images) {
            Label("Choose from Library", systemImage: "photo.on.rectangle")
                .frame(maxWidth: .infinity)
        }
        .controlSize(.large)
    }

    private func handle(image: UIImage) async {
        await model.startTrim(image: image)
    }
}
