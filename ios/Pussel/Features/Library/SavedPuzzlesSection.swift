import SwiftUI
import UIKit

/// Shared metrics for the swipe row and the card it slides over. Kept outside
/// `SwipeToDeleteRow` because generic types can't hold static stored properties.
private enum SwipeMetrics {
  static let cornerRadius: CGFloat = 12
  /// Width of the revealed Delete button.
  static let actionWidth: CGFloat = 80
  /// How far the row must have travelled at the end of a drag to stay open.
  static let openThreshold: CGFloat = 40
  /// Extra red drawn to the left of the button so its rounded leading corners
  /// stay tucked under the card instead of showing in the gap.
  static let actionUnderlap: CGFloat = cornerRadius
  /// Seconds of the release velocity to count toward the open/close decision,
  /// so a fast flick opens the row even if it fell short of `openThreshold`.
  static let flickProjection: CGFloat = 0.1
}

/// The list of locally stored puzzles shown beneath the capture buttons on the
/// home screen. Tap a card to reopen it; swipe left (or long-press) to delete.
struct SavedPuzzlesSection: View {
  @Environment(AppModel.self) private var model
  /// At most one row is swiped open at a time, mirroring `List`.
  @State private var openRowID: UUID?

  var body: some View {
    // Lazy so rows (and their thumbnail decoding) are built only as they
    // scroll into view as the library grows.
    LazyVStack(alignment: .leading, spacing: 12) {
      Text("Your puzzles")
        .font(.headline)
        .frame(maxWidth: .infinity, alignment: .leading)

      ForEach(model.store.puzzles) { puzzle in
        SwipeToDeleteRow(
          isOpen: Binding(
            get: { openRowID == puzzle.id },
            set: { isOpen in
              if isOpen {
                openRowID = puzzle.id
              } else if openRowID == puzzle.id {
                openRowID = nil
              }
            }
          ),
          onOpen: { model.openPuzzle(puzzle.id) },
          onDelete: { delete(puzzle) },
          content: { SavedPuzzleRow(puzzle: puzzle) }
        )
        .contextMenu {
          Button(role: .destructive) {
            delete(puzzle)
          } label: {
            Label("Delete", systemImage: "trash")
          }
        }
      }
    }
  }

  /// Deletes straight away — the swipe is the deliberate act, so there's no
  /// confirmation step. Animated so the surviving rows close the gap instead of
  /// jumping up.
  private func delete(_ puzzle: PuzzleSummary) {
    openRowID = nil
    withAnimation(.snappy) {
      model.deletePuzzle(puzzle.id)
    }
  }
}

/// Floating "Deleted … / Undo" bar shown while a delete's undo window is open.
/// Lives on the home screen rather than inside the list so it stays put at the
/// bottom of the screen while the list scrolls underneath it.
struct UndoDeleteSnackbar: View {
  @Environment(AppModel.self) private var model

  var body: some View {
    if let pending = model.store.pendingDelete {
      HStack(spacing: 12) {
        // The picture says which puzzle went far better than its timestamp
        // name did; "Deleted" alone then carries the meaning.
        PuzzleThumbnail(data: pending.thumbnail, size: 32, cornerRadius: 6)
        Text("Deleted")
          .font(.subheadline)
        Spacer(minLength: 0)
        Button("Undo") {
          withAnimation(.snappy) { model.undoDelete() }
        }
        .font(.subheadline.weight(.semibold))
      }
      .padding(.leading, 10)
      .padding(.trailing, 16)
      .padding(.vertical, 8)
      .background(.regularMaterial, in: Capsule())
      .shadow(radius: 8, y: 2)
      .padding(.horizontal, 24)
      .padding(.bottom, 12)
      .transition(.move(edge: .bottom).combined(with: .opacity))
      // The thumbnail is meaningless to VoiceOver, so name the puzzle here.
      .accessibilityElement(children: .contain)
      .accessibilityLabel("Deleted “\(pending.name)”")
    }
  }
}

/// The stored puzzle picture, falling back to a placeholder symbol when a
/// puzzle has no cached thumbnail.
private struct PuzzleThumbnail: View {
  let data: Data?
  let size: CGFloat
  let cornerRadius: CGFloat

  var body: some View {
    Group {
      if let data, let image = UIImage(data: data) {
        Image(uiImage: image)
          .resizable()
          .scaledToFill()
      } else {
        Image(systemName: "photo")
          .foregroundStyle(.secondary)
      }
    }
    .frame(width: size, height: size)
    .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
  }
}

/// A row that slides left under a horizontal drag to reveal a red Delete
/// button, the way `List`'s `swipeActions` does. Hand-rolled because this list
/// is a `LazyVStack` in a `ScrollView`, not a `List`.
private struct SwipeToDeleteRow<Content: View>: View {
  @Binding var isOpen: Bool
  let onOpen: () -> Void
  let onDelete: () -> Void
  /// Must be inert: the row owns the tap, because a `Button` here would win the
  /// touch outright and every swipe would register as a tap on the card.
  @ViewBuilder let content: Content

  /// Live drag distance, folded into `offset` on top of the resting position.
  @State private var translation: CGFloat = 0

  /// Resting position plus the in-flight drag, clamped so the card can't be
  /// pulled past the button or dragged right of its home position.
  private var offset: CGFloat {
    let resting: CGFloat = isOpen ? -SwipeMetrics.actionWidth : 0
    return min(0, max(-SwipeMetrics.actionWidth, resting + translation))
  }

  var body: some View {
    content
      // The card is translucent, so it needs an opaque layer under it or the
      // red button shows through the card as it slides.
      .background(
        Color(.systemBackground), in: RoundedRectangle(cornerRadius: SwipeMetrics.cornerRadius)
      )
      .contentShape(RoundedRectangle(cornerRadius: SwipeMetrics.cornerRadius))
      // `offset` carries hit testing with it, so the card stays tappable where
      // it's drawn and stops covering the button once it slides clear.
      .offset(x: offset)
      // A UIKit-backed pan, not a SwiftUI DragGesture — see HorizontalPanGesture
      // for why the latter can't be used inside a ScrollView.
      .gesture(swipe)
      // While open, the card itself only closes the row — a stray tap must not
      // open the puzzle the user is about to delete.
      .onTapGesture {
        if isOpen {
          close()
        } else {
          onOpen()
        }
      }
      .accessibilityElement(children: .combine)
      .accessibilityAddTraits(.isButton)
      // The button sits in the *card's* background rather than a ZStack sibling
      // so the card alone sizes the row and the button can stretch to its
      // height. A background is laid out in the unoffset frame, so it stays put
      // while the card slides off it.
      .background(alignment: .trailing) { deleteButton }
      // VoiceOver can't swipe; give it the same action via the rotor.
      .accessibilityAction(named: "Delete") { onDelete() }
  }

  private var deleteButton: some View {
    Button(role: .destructive, action: onDelete) {
      Label("Delete", systemImage: "trash")
        .labelStyle(.iconOnly)
        .font(.title3)
        .foregroundStyle(.white)
        .frame(maxHeight: .infinity)
        .frame(width: SwipeMetrics.actionWidth)
        // The underlap is drawn but never tapped — it stays under the card.
        .padding(.leading, SwipeMetrics.actionUnderlap)
        .background(.red, in: RoundedRectangle(cornerRadius: SwipeMetrics.cornerRadius))
        .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .accessibilityLabel("Delete puzzle")
    // Hidden behind the card when closed, so keep it out of the tab/VoiceOver
    // order until it's actually on screen.
    .disabled(!isOpen)
    .accessibilityHidden(!isOpen)
  }

  // Not `some Gesture`: a UIGestureRecognizerRepresentable isn't a Gesture, it
  // just has its own `.gesture(_:)` overload.
  private var swipe: HorizontalPanGesture {
    HorizontalPanGesture(
      onChanged: { translation = $0 },
      onEnded: { velocity in
        // Fold in the flick so a short, fast swipe opens the row rather than
        // snapping back for falling short of the distance threshold.
        let projected = offset + velocity * SwipeMetrics.flickProjection
        withAnimation(.snappy(duration: 0.25)) {
          isOpen = projected < -SwipeMetrics.openThreshold
          translation = 0
        }
      }
    )
  }

  private func close() {
    withAnimation(.snappy(duration: 0.25)) {
      isOpen = false
      translation = 0
    }
  }
}

/// A pan recognizer that gives up the moment a drag looks vertical.
///
/// It marks itself `.failed` rather than declining via a delegate: SwiftUI
/// installs its own delegate on a `UIGestureRecognizerRepresentable`, so a
/// `gestureRecognizerShouldBegin` of ours is never consulted (verified — the
/// list still refused to scroll). Failing from inside the recognizer is not
/// overridable, and hands the touch back to the enclosing scroll view.
private final class HorizontalPanRecognizer: UIPanGestureRecognizer {
  /// How far a touch must travel before its direction is called. Below UIKit's
  /// own ~10pt pan threshold, so the verdict lands before this recognizer would
  /// otherwise begin and take the touch.
  private static let decisionDistance: CGFloat = 8

  private var origin: CGPoint?

  override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
    super.touchesBegan(touches, with: event)
    origin = touches.first?.location(in: view)
  }

  override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent) {
    // Judge before `super`, which is what would flip this to `.began`.
    if state == .possible, let origin, let location = touches.first?.location(in: view) {
      let dx = abs(location.x - origin.x)
      let dy = abs(location.y - origin.y)
      if max(dx, dy) >= Self.decisionDistance, dy > dx {
        state = .failed
        return
      }
    }
    super.touchesMoved(touches, with: event)
  }

  override func reset() {
    super.reset()
    origin = nil
  }
}

/// A pan gesture that reports sideways drags and ignores vertical ones.
///
/// SwiftUI's own `DragGesture` can't be used for this: attached anywhere inside
/// a `ScrollView` it starves the scroll view's pan once it recognises, and
/// neither `simultaneousGesture` nor a larger `minimumDistance` helps. Verified
/// in the Simulator — with a `DragGesture` the list would not scroll from a
/// card at all, while the same drag over the header scrolled fine.
private struct HorizontalPanGesture: UIGestureRecognizerRepresentable {
  /// Sideways distance from the start of the pan.
  let onChanged: (CGFloat) -> Void
  /// Release velocity, in points per second.
  let onEnded: (CGFloat) -> Void

  func makeUIGestureRecognizer(context: Context) -> HorizontalPanRecognizer {
    HorizontalPanRecognizer()
  }

  func handleUIGestureRecognizerAction(_ recognizer: HorizontalPanRecognizer, context: Context) {
    switch recognizer.state {
    case .changed:
      onChanged(recognizer.translation(in: recognizer.view).x)
    case .ended, .cancelled:
      onEnded(recognizer.velocity(in: recognizer.view).x)
    default:
      break
    }
  }
}

private struct SavedPuzzleRow: View {
  let puzzle: PuzzleSummary

  var body: some View {
    HStack(spacing: 12) {
      PuzzleThumbnail(data: puzzle.thumbnail, size: 56, cornerRadius: 8)
      VStack(alignment: .leading, spacing: 4) {
        Text(puzzle.name)
          .font(.subheadline.weight(.semibold))
          .foregroundStyle(.primary)
          .lineLimit(1)
        Text(subtitle)
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Spacer(minLength: 0)
      Image(systemName: "chevron.right")
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.tertiary)
    }
    .padding(10)
    .background(
      .quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: SwipeMetrics.cornerRadius))
  }

  private var subtitle: String {
    let pieces = puzzle.pieceCount == 1 ? "1 piece" : "\(puzzle.pieceCount) pieces"
    return "\(pieces) · \(puzzle.createdAt.formatted(date: .abbreviated, time: .shortened))"
  }
}
