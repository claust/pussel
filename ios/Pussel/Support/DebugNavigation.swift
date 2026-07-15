#if DEBUG
  import UIKit

  /// Debug-only command channel so the wizard can be driven from the CLI
  /// while testing in the Simulator (which has no camera and no way to tap).
  /// `simctl openurl` shows a confirmation dialog for custom schemes, so the
  /// promptless path is a Darwin notification plus a command file:
  ///   echo "pusseldebug://trim?puzzle=/host/path.jpg" > /tmp/pussel-debug-command
  ///   xcrun simctl spawn booted notifyutil -p dk.delectosoft.pussel.debug
  /// Commands: trim?puzzle=, accept, piece?path=, reupload, reset,
  ///   open?index=, delete?index= (index into the saved-puzzles list).
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
    private func runDebugCommand(_ host: String?, value: (String) -> String?) async {
      switch host {
      case "reset":
        flow.reset()
      case "trim":
        await debugTrim(path: value("puzzle"))
      case "accept":
        await debugAccept()
      case "piece":
        debugAddPiece(path: value("path"))
      case "reupload":
        await debugReupload()
      case "open":
        debugOpen(index: value("index"))
      case "delete":
        debugDelete(index: value("index"))
      default:
        break
      }
    }

    private func debugTrim(path: String?) async {
      if let image = Self.hostImage(path) {
        await startTrim(image: image, source: .library)
      }
    }

    private func debugAccept() async {
      if case .confirmTrim(let candidate) = flow.phase {
        await acceptTrim(candidate)
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

    private static func hostImage(_ path: String?) -> UIImage? {
      guard let path else { return nil }
      return UIImage(contentsOfFile: path)
    }
  }
#endif
