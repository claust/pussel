import SwiftUI
import UIKit

/// An image the user can pinch, pan and double-tap to zoom, with a SwiftUI
/// `overlay` drawn on top that tracks the image as it moves.
///
/// Backed by `UIScrollView` rather than SwiftUI gestures, which buys the
/// familiar feel — rubber-banding at the edges, momentum, zoom bounce — that a
/// hand-rolled `MagnifyGesture`/`DragGesture` pair has to reimplement badly.
///
/// The overlay is a sibling of the zoomed image view, not a child, so the
/// scroll view's zoom transform never touches it. Its frame is instead kept
/// equal to the image's, which leaves it laying out its SwiftUI content afresh
/// at every scale: content stays vector-crisp, and anything sized in points
/// (a marker's border, its confidence bar) keeps a constant size on screen
/// instead of growing into a slab over the artwork. As a side effect nothing
/// here writes SwiftUI state from a scroll callback — the frame change alone
/// drives the re-layout, so panning can't provoke a mid-update mutation.
///
/// The overlay is positioned in the image's own normalized space: it is handed
/// exactly the image's rect, so a `GeometryReader` inside it can place a
/// feature at (x, y) ∈ [0,1] as `geo.size * (x, y)`.
struct ZoomableImageView<Overlay: View>: UIViewRepresentable {
  /// The image to display. Its pixel size defines the zoom content size.
  let image: UIImage
  /// Region to zoom to, in normalized image coordinates. Applied when it
  /// changes (and once the view has been laid out); nil opens at fit.
  var focusRect: CGRect?
  /// How far past fit the image may be magnified.
  var maxMagnification: CGFloat = 6
  /// Scale, relative to fit, that a double-tap zooms to.
  var doubleTapMagnification: CGFloat = 3
  @ViewBuilder let overlay: Overlay

  func makeCoordinator() -> Coordinator {
    Coordinator(maxMagnification: maxMagnification, doubleTapMagnification: doubleTapMagnification)
  }

  func makeUIView(context: Context) -> UIScrollView {
    let coordinator = context.coordinator
    let scrollView = LayoutReportingScrollView()
    scrollView.delegate = coordinator
    scrollView.showsVerticalScrollIndicator = false
    scrollView.showsHorizontalScrollIndicator = false
    scrollView.contentInsetAdjustmentBehavior = .never
    scrollView.backgroundColor = .clear
    scrollView.onLayout = { [weak coordinator, weak scrollView] in
      guard let coordinator, let scrollView else { return }
      coordinator.refreshLayout(scrollView)
    }

    // Sized in pixels so zoomScale 1 shows one image pixel per point, and the
    // fit scale computed in `refreshLayout` is relative to the real image.
    coordinator.baseSize = CGSize(
      width: image.size.width * image.scale, height: image.size.height * image.scale)
    coordinator.imageView.image = image
    coordinator.imageView.frame = CGRect(origin: .zero, size: coordinator.baseSize)
    scrollView.addSubview(coordinator.imageView)
    scrollView.contentSize = coordinator.baseSize

    let host = UIHostingController(rootView: overlay)
    host.view.backgroundColor = .clear
    // The overlay is decoration over the image: it must never intercept the
    // pinch and pan that are the point of this view.
    host.view.isUserInteractionEnabled = false
    // Otherwise the host insets its content by the safe area, which would
    // shift the overlay off the image it is meant to track.
    host.safeAreaRegions = []
    coordinator.host = host
    scrollView.addSubview(host.view)

    let doubleTap = UITapGestureRecognizer(
      target: coordinator, action: #selector(Coordinator.handleDoubleTap(_:)))
    doubleTap.numberOfTapsRequired = 2
    scrollView.addGestureRecognizer(doubleTap)

    return scrollView
  }

  func updateUIView(_ scrollView: UIScrollView, context: Context) {
    let coordinator = context.coordinator
    coordinator.host?.rootView = overlay
    guard coordinator.requestedFocus != focusRect else { return }
    coordinator.requestedFocus = focusRect
    // Deferred when the view has no size yet (the first update lands before
    // layout): refreshLayout applies it as soon as the bounds are real.
    coordinator.pendingFocus = focusRect
    coordinator.applyPendingFocus(scrollView, animated: true)
  }

  /// Reports every layout pass, which is where the fit scale can first be
  /// computed and must be recomputed on rotation or a size change.
  final class LayoutReportingScrollView: UIScrollView {
    var onLayout: (() -> Void)?

    override func layoutSubviews() {
      super.layoutSubviews()
      onLayout?()
    }
  }

  final class Coordinator: NSObject, UIScrollViewDelegate {
    let imageView = UIImageView()
    var host: UIHostingController<Overlay>?
    var baseSize: CGSize = .zero
    /// The focus last handed to us, to detect a change (nil is a meaningful
    /// value, so this can't just be the optional itself).
    var requestedFocus: CGRect??
    var pendingFocus: CGRect?

    private let maxMagnification: CGFloat
    private let doubleTapMagnification: CGFloat
    private var hasSized = false

    init(maxMagnification: CGFloat, doubleTapMagnification: CGFloat) {
      self.maxMagnification = maxMagnification
      self.doubleTapMagnification = doubleTapMagnification
      imageView.contentMode = .scaleAspectFill
      imageView.isUserInteractionEnabled = false
    }

    func viewForZooming(in scrollView: UIScrollView) -> UIView? { imageView }

    func scrollViewDidZoom(_ scrollView: UIScrollView) {
      syncOverlay(scrollView)
    }

    /// Establishes (and re-establishes) the fit scale, then keeps the content
    /// centred and the overlay aligned.
    func refreshLayout(_ scrollView: UIScrollView) {
      guard scrollView.bounds.width > 0, scrollView.bounds.height > 0,
        baseSize.width > 0, baseSize.height > 0
      else {
        return
      }
      let fit = min(
        scrollView.bounds.width / baseSize.width, scrollView.bounds.height / baseSize.height)
      // `!hasSized` is not redundant with the scale check: an image that
      // happens to fit at exactly 1.0 matches UIScrollView's default minimum,
      // and would otherwise never be marked sized (leaving focus and
      // double-tap inert forever).
      if !hasSized || scrollView.minimumZoomScale != fit {
        // Hold the user's magnification across a bounds change (rotation)
        // rather than throwing them back to fit mid-inspection.
        let magnification = hasSized ? scrollView.zoomScale / scrollView.minimumZoomScale : 1
        scrollView.minimumZoomScale = fit
        scrollView.maximumZoomScale = fit * maxMagnification
        scrollView.zoomScale = min(fit * magnification, fit * maxMagnification)
        hasSized = true
      }
      syncOverlay(scrollView)
      applyPendingFocus(scrollView, animated: false)
    }

    /// Zooms to the pending normalized rect, once there is a fit scale to
    /// measure it against. Consumed on success so it fires only once.
    func applyPendingFocus(_ scrollView: UIScrollView, animated: Bool) {
      guard hasSized, let focus = pendingFocus, baseSize.width > 0, baseSize.height > 0 else {
        return
      }
      pendingFocus = nil
      // `zoom(to:)` takes the zoomed view's own coordinate space — the image
      // at scale 1 — and clamps to maximumZoomScale, so a rect smaller than
      // the detail we have simply stops at the sharpest scale on offer.
      let rect = CGRect(
        x: focus.minX * baseSize.width,
        y: focus.minY * baseSize.height,
        width: focus.width * baseSize.width,
        height: focus.height * baseSize.height)
      scrollView.zoom(to: rect, animated: animated)
    }

    @objc func handleDoubleTap(_ recognizer: UITapGestureRecognizer) {
      guard let scrollView = recognizer.view as? UIScrollView, hasSized else { return }
      let fit = scrollView.minimumZoomScale
      if scrollView.zoomScale > fit * 1.01 {
        scrollView.setZoomScale(fit, animated: true)
        return
      }
      // Zoom in around the tapped point, so a double-tap on a piece brings up
      // that piece rather than the middle of the puzzle.
      let target = min(fit * doubleTapMagnification, scrollView.maximumZoomScale)
      let point = recognizer.location(in: imageView)
      let size = CGSize(
        width: scrollView.bounds.width / target, height: scrollView.bounds.height / target)
      scrollView.zoom(
        to: CGRect(
          x: point.x - size.width / 2, y: point.y - size.height / 2,
          width: size.width, height: size.height),
        animated: true)
    }

    /// Centres the content while it's smaller than the viewport and pins the
    /// overlay to the image. Insets do the centring so the zoomed view's
    /// transform is left alone.
    private func syncOverlay(_ scrollView: UIScrollView) {
      let insetX = max(0, (scrollView.bounds.width - scrollView.contentSize.width) / 2)
      let insetY = max(0, (scrollView.bounds.height - scrollView.contentSize.height) / 2)
      let inset = UIEdgeInsets(top: insetY, left: insetX, bottom: insetY, right: insetX)
      if scrollView.contentInset != inset {
        scrollView.contentInset = inset
      }
      host?.view.frame = imageView.frame
    }
  }
}
