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
            ProcessInfo.processInfo.environment["PUSSEL_DEBUG_COMMAND_FILE"] ?? "/tmp/pussel-debug-command"
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
                  let url = URL(string: command.trimmingCharacters(in: .whitespacesAndNewlines)) else { return }
            _ = model?.handleDebugURL(url)
        }
    }

    extension AppModel {
        func handleDebugURL(_ url: URL) -> Bool {
            guard url.scheme == "pusseldebug" else { return false }
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
            func value(_ name: String) -> String? {
                items.first { $0.name == name }?.value
            }
            Task {
                switch url.host() {
                case "reset":
                    flow.reset()
                case "trim":
                    if let image = Self.hostImage(value("puzzle")) {
                        await startTrim(image: image, source: .library)
                    }
                case "accept":
                    if case .confirmTrim(let candidate) = flow.phase {
                        await acceptTrim(candidate)
                    }
                case "piece":
                    if let image = Self.hostImage(value("path")) {
                        addPiece(image: image)
                    }
                case "reupload":
                    if case .solving(let session) = flow.phase {
                        await session.reupload(api: api)
                    }
                case "open":
                    if let index = value("index").flatMap(Int.init), store.puzzles.indices.contains(index) {
                        openPuzzle(store.puzzles[index].id)
                    }
                case "delete":
                    if let index = value("index").flatMap(Int.init), store.puzzles.indices.contains(index) {
                        deletePuzzle(store.puzzles[index].id)
                    }
                default:
                    break
                }
            }
            return true
        }

        private static func hostImage(_ path: String?) -> UIImage? {
            guard let path else { return nil }
            return UIImage(contentsOfFile: path)
        }
    }
#endif
