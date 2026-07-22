#if DEBUG
  import UIKit

  /// Debug-only command channel so the wizard can be driven from the CLI
  /// while testing in the Simulator (which has no camera and no way to tap).
  /// `simctl openurl` shows a confirmation dialog for custom schemes, so the
  /// promptless path is a Darwin notification plus a command file:
  ///   echo "pusseldebug://trim?puzzle=/host/path.jpg" > /tmp/pussel-debug-command
  ///   xcrun simctl spawn booted notifyutil -p dk.delectosoft.pussel.debug
  /// Commands: trim?puzzle=, accept[?pieces=], piece?path=, reupload, reset,
  ///   open?index=, delete?index= (index into the saved-puzzles list),
  ///   camera[?open=0], previewloop?path=<host image path>[&stop=1] (M9 live
  ///   preview overlay demo — see PieceCameraSession.startDebugPreviewLoop;
  ///   also drives the box camera's barcode flow when that screen is open,
  ///   see BoxCameraSession.startDebugPreviewLoop),
  ///   scan[?open=0] (M10 scan-and-lock demo — mirrors camera),
  ///   scanconfirm (taps the M10 uncertain-confirm chip on the open scan view),
  ///   boxcamera[?open=0] (forces the capture screen's live box-camera cover
  ///   open, for the barcode auto-lookup flow — mirrors camera).
  /// Simulator apps can read host file paths directly. Compiled out of
  /// Release builds; the handler runs the same actions as the real UI.
  @MainActor
  enum DebugDriver {
    static weak var model: AppModel?
    private static let notificationName = "dk.delectosoft.pussel.debug" as CFString
    /// Override with SIMCTL_CHILD_PUSSEL_DEBUG_COMMAND_FILE at launch when
    /// the host shell cannot write to /tmp.
    private static var commandFile: String {
      ProcessInfo.processInfo.environment["PUSSEL_DEBUG_COMMAND_FILE"]
        ?? "/tmp/pussel-debug-command"
    }

    private static var isObserving = false

    static func start(_ model: AppModel) {
      Self.model = model
      // .task can re-run when the view hierarchy is recreated; register
      // the Darwin observer only once or commands would run repeatedly.
      guard !isObserving else { return }
      isObserving = true
      CFNotificationCenterAddObserver(
        CFNotificationCenterGetDarwinNotifyCenter(),
        nil,
        { _, _, _, _, _ in
          Task { @MainActor in
            DebugDriver.runPendingCommand()
          }
        },
        notificationName,
        nil,
        .deliverImmediately
      )
    }

    private static func runPendingCommand() {
      guard let command = try? String(contentsOfFile: commandFile, encoding: .utf8),
        let url = URL(string: command.trimmingCharacters(in: .whitespacesAndNewlines))
      else { return }
      _ = model?.handleDebugURL(url)
    }
  }

  extension AppModel {
    func handleDebugURL(_ url: URL) -> Bool {
      guard url.scheme == "pusseldebug" else { return false }
      let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
      let host = url.host()
      Task { @MainActor in
        await runDebugCommand(host) { name in items.first { $0.name == name }?.value }
      }
      return true
    }

    // One thin case per command keeps this dispatcher's cyclomatic complexity
    // low; each command's own guards live in its helper below.
    // swiftlint:disable:next cyclomatic_complexity
    private func runDebugCommand(_ host: String?, value: (String) -> String?) async {
      switch host {
      case "reset":
        flow.reset()
      case "trim":
        await debugTrim(path: value("puzzle"))
      case "accept":
        await debugAccept(pieces: value("pieces"))
      case "piece":
        debugAddPiece(path: value("path"))
      case "reupload":
        await debugReupload()
      case "open":
        debugOpen(index: value("index"))
      case "delete":
        debugDelete(index: value("index"))
      case "camera":
        debugCamera(open: value("open"))
      case "boxcamera":
        debugBoxCamera(open: value("open"))
      case "scan":
        debugScan(open: value("open"))
      case "scanconfirm":
        await PieceScanController.debugActive?.confirmUncertainAsNew()
      case "previewloop":
        debugPreviewLoop(path: value("path"), stop: value("stop"))
      default:
        break
      }
    }

    private func debugTrim(path: String?) async {
      if let image = Self.hostImage(path) {
        await startTrim(image: image, source: .library)
      }
    }

    /// `pieces` is an optional `?pieces=<n>` query value; the debug driver
    /// has no UI to enter a count, so it defaults to a plausible test value.
    private func debugAccept(pieces: String?) async {
      if case .confirmTrim(let candidate) = flow.phase {
        let pieceCount = pieces.flatMap(Int.init) ?? 100
        await acceptTrim(candidate, pieceCount: pieceCount)
      }
    }

    private func debugAddPiece(path: String?) {
      if let image = Self.hostImage(path) {
        addPiece(image: image)
      }
    }

    private func debugReupload() async {
      if case .solving(let session) = flow.phase {
        await session.reupload(api: api)
      }
    }

    private func debugOpen(index: String?) {
      guard let index = index.flatMap(Int.init), store.puzzles.indices.contains(index) else {
        return
      }
      openPuzzle(store.puzzles[index].id)
    }

    private func debugDelete(index: String?) {
      guard let index = index.flatMap(Int.init), store.puzzles.indices.contains(index) else {
        return
      }
      deletePuzzle(store.puzzles[index].id)
    }

    /// Forces the piece camera cover open (or closed with `?open=0`), even
    /// on the Simulator where `PieceCameraSession.isCameraAvailable` is
    /// false — see `PieceQueueView.cameraCoverIsPresented`. Must already be
    /// in the solving phase (drive there first with `trim`/`accept`).
    private func debugCamera(open: String?) {
      guard case .solving(let session) = flow.phase else { return }
      session.debugCameraOpen = (open.flatMap(Int.init) ?? 1) != 0
    }

    /// Forces the scan-and-lock cover open (or closed with `?open=0`), even
    /// on the Simulator — mirrors `debugCamera` for the M10 scan flow.
    private func debugScan(open: String?) {
      guard case .solving(let session) = flow.phase else { return }
      session.debugScanOpen = (open.flatMap(Int.init) ?? 1) != 0
    }

    /// Forces the capture screen's live box-camera cover open (or closed
    /// with `?open=0`), even on the Simulator where
    /// `BoxCameraSession.isCameraAvailable` is false — see
    /// `CapturePuzzleView.cameraCoverIsPresented`. Must be in the
    /// capture-puzzle phase (`reset` gets there).
    private func debugBoxCamera(open: String?) {
      flow.debugBoxCameraOpen = (open.flatMap(Int.init) ?? 1) != 0
    }

    /// Starts (or, with `?stop=1`, stops) a repeating fake-frame loop on the
    /// active live camera session, feeding `path` through the same
    /// downscale → analyze pipeline a real camera frame takes — the
    /// Simulator's stand-in for a live camera. Targets whichever session is
    /// open: the piece camera (M9 overlay demo) or the box camera (barcode
    /// auto-lookup flow). Requires that screen to already be open (`camera`
    /// / `boxcamera`, or the real screens on a device).
    private func debugPreviewLoop(path: String?, stop: String?) {
      if let camera = PieceCameraSession.debugActive {
        if (stop.flatMap(Int.init) ?? 0) != 0 {
          camera.stopDebugPreviewLoop()
          return
        }
        guard let image = Self.hostImage(path) else { return }
        camera.startDebugPreviewLoop(image: image)
        return
      }
      guard let camera = BoxCameraSession.debugActive else { return }
      if (stop.flatMap(Int.init) ?? 0) != 0 {
        camera.stopDebugPreviewLoop()
        return
      }
      guard let image = Self.hostImage(path) else { return }
      camera.startDebugPreviewLoop(image: image)
    }

    private static func hostImage(_ path: String?) -> UIImage? {
      guard let path else { return nil }
      return UIImage(contentsOfFile: path)
    }
  }
#endif
