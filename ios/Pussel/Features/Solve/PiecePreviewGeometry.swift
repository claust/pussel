import Foundation

/// Pure math for mapping the backend's frame-normalized piece outline onto
/// the camera-preview overlay's bounds.
///
/// The frames `PieceCameraSession` streams are rotated to upright portrait
/// at the connection level (`videoRotationAngle = 90` on the video data
/// output — see `configureIfNeeded`), so the downscaled JPEG the backend
/// measured is visually identical to what the portrait
/// `AVCaptureVideoPreviewLayer` shows: same image, same aspect ratio, no
/// orientation difference to correct for. That makes the overlay mapping a
/// single aspect-fill: the preview layer displays the feed with
/// `.resizeAspectFill`, and `viewPoint` applies the same centered fill to
/// the polygon, reproducing the layer's geometry without going through
/// `layerPointConverted(fromCaptureDevicePoint:)` — which also lets the
/// identical path serve the Simulator, where no capture session runs and no
/// preview-layer connection exists (the DEBUG `previewloop` feeds host
/// images whose axes are likewise display axes).
enum PiecePreviewGeometry {
  /// Maps one frame-normalized point onto the overlay view's bounds with
  /// centered **aspect-fill** — the same fit the `.resizeAspectFill`
  /// preview layer applies to the (same-aspect) live feed, cropping the
  /// overflowing axis equally on both sides.
  ///
  /// - Parameters:
  ///   - point: a polygon point as returned by the backend, normalized to
  ///     the upright frame it received.
  ///   - frameSize: pixel size of that frame (the downscaled JPEG's size);
  ///     only its aspect ratio matters. A zero/degenerate size falls back
  ///     to a plain stretch onto the bounds.
  ///   - viewBounds: the overlay view's bounds size, in points.
  /// - Returns: the point in the overlay view's coordinate space.
  static func viewPoint(
    fromFramePoint point: NormalizedPoint, frameSize: CGSize, viewBounds: CGSize
  ) -> CGPoint {
    guard frameSize.width > 0, frameSize.height > 0 else {
      return CGPoint(x: point.x * viewBounds.width, y: point.y * viewBounds.height)
    }
    let scale = max(viewBounds.width / frameSize.width, viewBounds.height / frameSize.height)
    let scaledWidth = frameSize.width * scale
    let scaledHeight = frameSize.height * scale
    // Centered aspect-fill: the scaled frame overflows on one axis, so its
    // origin is pushed negative there to keep it centered.
    let originX = (viewBounds.width - scaledWidth) / 2
    let originY = (viewBounds.height - scaledHeight) / 2
    return CGPoint(x: originX + point.x * scaledWidth, y: originY + point.y * scaledHeight)
  }

  /// Maps a full polygon through `viewPoint(fromFramePoint:frameSize:viewBounds:)`,
  /// preserving point order.
  static func viewPolygon(
    fromFramePolygon polygon: [NormalizedPoint], frameSize: CGSize, viewBounds: CGSize
  ) -> [CGPoint] {
    polygon.map { viewPoint(fromFramePoint: $0, frameSize: frameSize, viewBounds: viewBounds) }
  }
}
