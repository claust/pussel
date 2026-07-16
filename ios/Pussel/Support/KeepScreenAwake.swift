import SwiftUI
import UIKit

/// Ref-counted holder for the idle timer. Counted rather than a plain
/// set/clear because SwiftUI can bring the next view on screen before the
/// previous one disappears, and the last `onDisappear` would otherwise
/// release a lock the newly-appeared view still wants.
@MainActor
final class ScreenAwakeCounter {
  static let shared = ScreenAwakeCounter()

  /// Number of views currently asking for the screen to stay awake.
  private(set) var holders = 0

  func acquire() {
    holders += 1
    UIApplication.shared.isIdleTimerDisabled = true
  }

  func release() {
    holders = max(0, holders - 1)
    if holders == 0 {
      UIApplication.shared.isIdleTimerDisabled = false
    }
  }
}

private struct KeepScreenAwakeModifier: ViewModifier {
  func body(content: Content) -> some View {
    content
      .onAppear { ScreenAwakeCounter.shared.acquire() }
      .onDisappear { ScreenAwakeCounter.shared.release() }
  }
}

extension View {
  /// Keeps the screen from dimming while this view is on screen. Meant for the
  /// camera viewfinders: lining up a shot of the puzzle or a piece means
  /// holding still without touching the screen, which is exactly when the idle
  /// timer would fire. iOS ignores the flag while the app isn't frontmost, so
  /// backgrounding needs no extra handling.
  func keepsScreenAwake() -> some View {
    modifier(KeepScreenAwakeModifier())
  }
}
