import CoreGraphics
import simd

/// The camera geometry of one shot in a glare-free burst: where the phone
/// was, what it saw through, and where the puzzle's surface lies.
///
/// ARKit already computes all of this per frame; the capture flow used to
/// throw it away after steering the guide dot. Kept, it turns five
/// obliquely-shot photos into five views of one known plane, which is what
/// `PlaneRectification` needs to warp them into a common top-down space.
struct PlaneCaptureGeometry: Equatable, Sendable {
  /// Camera → world, ARKit's `ARFrame.camera.transform`: +x right, +y up,
  /// −z into the scene, gravity-aligned.
  var cameraTransform: simd_float4x4
  /// Pinhole intrinsics expressed in the pixel space of the **upright**
  /// (portrait, EXIF-baked) still, not the sensor's landscape buffer — see
  /// `PlaneRectification.uprightIntrinsics`.
  var intrinsics: simd_float3x3
  /// The upright still's full pixel size, which is the space `intrinsics`
  /// describes. Warping a downscaled copy means scaling by the ratio.
  var imageSize: CGSize
  /// World y of the horizontal plane the puzzle lies on.
  var planeHeight: Float
}

/// Maps photos of a known horizontal plane into a shared, metrically
/// correct top-down space.
///
/// The five glare-free shots are deliberately taken from five different
/// spots, so each sees the puzzle under a different perspective. Registering
/// them onto the *center* shot (what the composer does on its own) fuses
/// them faithfully but inherits that shot's obliqueness: a rectangular
/// puzzle stays a trapezoid, and the backend can only crop the trapezoid,
/// never unbend it. Warping every frame — the reference included — onto the
/// plane instead makes the composite rectangular by construction.
///
/// Plane coordinates here are world (x, z) in meters, since the world is
/// gravity-aligned and the surface is horizontal; the plane's own height
/// enters only through `PlaneCaptureGeometry.planeHeight`.
enum PlaneRectification {
  /// Flips ARKit's camera axes (+y up, −z forward) into the pinhole
  /// convention the intrinsics are written for (+y down, +z forward).
  private static let axisFlip = simd_float3x3(diagonal: SIMD3<Float>(1, -1, -1))

  /// The intrinsics of the sensor's landscape buffer, restated for the
  /// upright portrait still the capture path actually hands the composer.
  ///
  /// Two corrections, in order. `ARFrame.camera.intrinsics` describe
  /// `camera.imageResolution`, which for a high-resolution capture is not
  /// the resolution of `capturedImage` — so scale first. Then the still is
  /// rotated 90° clockwise into portrait (`CIImage.oriented(.right)`,
  /// matching the camera sessions' `videoRotationAngle = 90`), which sends
  /// landscape pixel (x, y) of a W×H buffer to portrait (H − y, x).
  ///
  /// Returns nil when either size is degenerate.
  static func uprightIntrinsics(
    sensor: simd_float3x3, sensorResolution: CGSize, captureResolution: CGSize
  ) -> simd_float3x3? {
    guard sensorResolution.width > 0, sensorResolution.height > 0,
      captureResolution.width > 0, captureResolution.height > 0
    else { return nil }
    let scale = Float(captureResolution.width / sensorResolution.width)
    let scaled = simd_float3x3(diagonal: SIMD3<Float>(scale, scale, 1)) * sensor
    let height = Float(captureResolution.height)
    let rotate = simd_float3x3(rows: [
      SIMD3<Float>(0, -1, height),
      SIMD3<Float>(1, 0, 0),
      SIMD3<Float>(0, 0, 1),
    ])
    return rotate * scaled
  }

  /// The upright still's pixel size for a landscape capture of
  /// `captureResolution` — the 90° rotation swaps the axes.
  static func uprightSize(captureResolution: CGSize) -> CGSize {
    CGSize(width: captureResolution.height, height: captureResolution.width)
  }

  /// The homography taking a plane point `(u, v, 1)` — world (x, z) in
  /// meters — to a homogeneous pixel of this shot's upright still.
  ///
  /// A plane is two-dimensional, so the full 3D projection collapses to a
  /// 3×3: the plane point is `u·x̂ + v·ẑ + h·ŷ`, and rewriting the camera
  /// projection over that basis leaves one matrix per shot.
  static func planeToImage(_ geometry: PlaneCaptureGeometry) -> simd_float3x3 {
    let rotation = simd_float3x3(
      SIMD3(
        geometry.cameraTransform.columns.0.x, geometry.cameraTransform.columns.0.y,
        geometry.cameraTransform.columns.0.z),
      SIMD3(
        geometry.cameraTransform.columns.1.x, geometry.cameraTransform.columns.1.y,
        geometry.cameraTransform.columns.1.z),
      SIMD3(
        geometry.cameraTransform.columns.2.x, geometry.cameraTransform.columns.2.y,
        geometry.cameraTransform.columns.2.z))
    let translation = SIMD3(
      geometry.cameraTransform.columns.3.x, geometry.cameraTransform.columns.3.y,
      geometry.cameraTransform.columns.3.z)
    let inverseRotation = rotation.transpose
    // Columns: where the plane's u and v axes go, and where its origin
    // (lifted to the plane's height) lands.
    let basis = simd_float3x3(
      inverseRotation * SIMD3<Float>(1, 0, 0),
      inverseRotation * SIMD3<Float>(0, 0, 1),
      inverseRotation * (SIMD3<Float>(0, geometry.planeHeight, 0) - translation))
    return geometry.intrinsics * axisFlip * basis
  }

  /// Where a pixel of this shot lands on the plane, or nil when its ray
  /// runs parallel to the plane or away from it (the horizon is in frame).
  static func planePoint(
    ofPixel pixel: CGPoint, imageToPlane: simd_float3x3
  ) -> SIMD2<Float>? {
    let mapped = imageToPlane * SIMD3<Float>(Float(pixel.x), Float(pixel.y), 1)
    // `planeToImage` scales by the point's depth, so its inverse scales by
    // the reciprocal: a positive w is exactly "in front of the camera".
    guard mapped.z > .ulpOfOne else { return nil }
    return SIMD2(mapped.x / mapped.z, mapped.y / mapped.z)
  }

  /// The shared top-down space every frame of a burst is warped into.
  struct OutputSpace: Equatable {
    /// Plane `(u, v, 1)` → output pixel, top-left origin.
    var planeToOutput: simd_float3x3
    /// The output image's pixel size.
    var size: CGSize
  }

  /// Builds the output space from the reference shot: the plane region that
  /// shot sees, viewed square-on.
  ///
  /// The output's "up" is the reference photo's own up, projected onto the
  /// plane, so the rectified composite is oriented the way the user framed
  /// it rather than along an arbitrary world axis. (Deriving it from the
  /// camera's forward axis would be degenerate for exactly the shot this
  /// flow asks for — the phone held flat over the table.)
  ///
  /// Returns nil when a corner of the reference frame misses the plane,
  /// which is the caller's signal to skip rectification for this burst
  /// rather than warp against a horizon.
  static func outputSpace(
    reference: PlaneCaptureGeometry, maxDimension: CGFloat
  ) -> OutputSpace? {
    guard maxDimension > 0, reference.imageSize.width > 0, reference.imageSize.height > 0,
      let imageToPlane = invert(planeToImage(reference))
    else { return nil }

    let width = reference.imageSize.width
    let height = reference.imageSize.height
    let cornerPixels = [
      CGPoint(x: 0, y: 0), CGPoint(x: width, y: 0),
      CGPoint(x: width, y: height), CGPoint(x: 0, y: height),
    ]
    var corners: [SIMD2<Float>] = []
    for pixel in cornerPixels {
      guard let point = planePoint(ofPixel: pixel, imageToPlane: imageToPlane) else { return nil }
      corners.append(point)
    }
    guard
      let up = planeUp(imageToPlane: imageToPlane, width: width, height: height)
    else { return nil }
    // Right-handed with world up: rotating the plane's "up" a quarter turn
    // gives the direction the photo's +x runs in.
    let right = SIMD2<Float>(-up.y, up.x)

    // Local axis-aligned coordinates, y already flipped so it grows
    // downward like an image's does.
    let across = corners.map { simd_dot($0, right) }
    let down = corners.map { -simd_dot($0, up) }
    guard let minAcross = across.min(), let maxAcross = across.max(),
      let minDown = down.min(), let maxDown = down.max()
    else { return nil }
    let spanAcross = maxAcross - minAcross
    let spanDown = maxDown - minDown
    let longest = max(spanAcross, spanDown)
    guard longest > .ulpOfOne else { return nil }
    let scale = Float(maxDimension) / longest

    let planeToOutput = simd_float3x3(rows: [
      SIMD3<Float>(scale * right.x, scale * right.y, -scale * minAcross),
      SIMD3<Float>(-scale * up.x, -scale * up.y, -scale * minDown),
      SIMD3<Float>(0, 0, 1),
    ])
    let size = CGSize(
      width: CGFloat((spanAcross * scale).rounded(.down)),
      height: CGFloat((spanDown * scale).rounded(.down)))
    guard size.width >= 1, size.height >= 1 else { return nil }
    return OutputSpace(planeToOutput: planeToOutput, size: size)
  }

  /// Which way "up in the photo" points once laid on the plane: the step
  /// from the frame's bottom edge midpoint to its top edge midpoint.
  /// Measured rather than derived from a camera axis so it stays stable for
  /// a phone held flat, the pose this capture flow asks for.
  private static func planeUp(
    imageToPlane: simd_float3x3, width: CGFloat, height: CGFloat
  ) -> SIMD2<Float>? {
    guard
      let bottom = planePoint(
        ofPixel: CGPoint(x: width / 2, y: height), imageToPlane: imageToPlane),
      let top = planePoint(ofPixel: CGPoint(x: width / 2, y: 0), imageToPlane: imageToPlane)
    else { return nil }
    let step = top - bottom
    let length = simd_length(step)
    guard length > .ulpOfOne else { return nil }
    return step / length
  }

  /// The largest axis-aligned rectangle of `space` that this shot's frame
  /// actually covers — the region made of real pixels.
  ///
  /// Rectifying an oblique frame turns its rectangular sensor footprint into
  /// a keystone, so the output's bounding box has wedges along its edges
  /// that the camera never saw. Filling those (with white, or anything else)
  /// is worse than it sounds: the fill's boundary is long, straight,
  /// frame-spanning and high-contrast, which is precisely what the backend's
  /// line-based quad detector looks for — observed in the field selecting
  /// the fill boundary over the puzzle itself. Cropping instead means every
  /// pixel of the composite is real, and the only strong rectangle left in
  /// it is the puzzle's.
  ///
  /// Returns nil when the frame's projection is unusable.
  static func coveredRect(
    of geometry: PlaneCaptureGeometry, in space: OutputSpace
  ) -> CGRect? {
    guard let warp = rectification(of: geometry, into: space) else { return nil }
    let width = geometry.imageSize.width
    let height = geometry.imageSize.height
    guard width > 0, height > 0 else { return nil }
    let sourceCorners = [
      CGPoint(x: 0, y: 0), CGPoint(x: width, y: 0),
      CGPoint(x: width, y: height), CGPoint(x: 0, y: height),
    ]
    var quad: [CGPoint] = []
    for corner in sourceCorners {
      let mapped = warp * SIMD3<Float>(Float(corner.x), Float(corner.y), 1)
      guard abs(mapped.z) > .ulpOfOne else { return nil }
      quad.append(
        CGPoint(x: CGFloat(mapped.x / mapped.z), y: CGFloat(mapped.y / mapped.z)))
    }
    // The inward box: each side pulled to whichever of its two corners is
    // more inset. Exact for the mild keystones a hand-held shot produces,
    // and clipped below to the output in case the shot was square-on enough
    // that the footprint reaches past it.
    var rect = CGRect(
      x: max(quad[0].x, quad[3].x),
      y: max(quad[0].y, quad[1].y),
      width: 0, height: 0)
    rect.size = CGSize(
      width: min(quad[1].x, quad[2].x) - rect.origin.x,
      height: min(quad[2].y, quad[3].y) - rect.origin.y)
    // Round inward, then clip: `integral` grows a rect to enclose itself, so
    // clipping first would leave it a fraction of a pixel past the warped
    // extent — and that sliver renders as a black border, which is the very
    // false edge this crop exists to avoid.
    rect = CGRect(
      x: rect.minX.rounded(.up), y: rect.minY.rounded(.up),
      width: rect.width.rounded(.down), height: rect.height.rounded(.down)
    ).intersection(CGRect(origin: .zero, size: space.size))
    guard rect.width >= 1, rect.height >= 1 else { return nil }
    // A strongly rotated quad can defeat the inward box, so verify rather
    // than assume, and shrink about the centre until it holds.
    for _ in 0..<8 {
      if corners(of: rect).allSatisfy({ contains(quad: quad, point: $0) }) { return rect }
      rect = rect.insetBy(dx: rect.width * 0.05, dy: rect.height * 0.05)
      guard rect.width >= 1, rect.height >= 1 else { return nil }
    }
    return nil
  }

  private static func corners(of rect: CGRect) -> [CGPoint] {
    [
      CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: rect.maxX, y: rect.minY),
      CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: rect.minX, y: rect.maxY),
    ]
  }

  /// Whether `point` lies inside the convex `quad`: every edge must turn the
  /// same way toward it.
  private static func contains(quad: [CGPoint], point: CGPoint) -> Bool {
    var sign: CGFloat = 0
    for index in quad.indices {
      let from = quad[index]
      let to = quad[(index + 1) % quad.count]
      let cross =
        (to.x - from.x) * (point.y - from.y) - (to.y - from.y) * (point.x - from.x)
      if abs(cross) < .ulpOfOne { continue }
      if sign == 0 {
        sign = cross < 0 ? -1 : 1
      } else if (cross < 0 ? -1 : 1) != sign {
        return false
      }
    }
    return true
  }

  /// The homography warping one shot's upright still into `space`, or nil
  /// when its projection is singular.
  static func rectification(
    of geometry: PlaneCaptureGeometry, into space: OutputSpace
  ) -> simd_float3x3? {
    guard let imageToPlane = invert(planeToImage(geometry)) else { return nil }
    return space.planeToOutput * imageToPlane
  }

  // MARK: - Coordinate plumbing

  /// Restates a top-left-origin pixel homography in the lower-left-origin
  /// space Core Image and Vision warps work in, so the result can go
  /// straight to `GlareFreeComposer.warpedImage`. Source and destination
  /// flip about their own heights, which differ once a frame is rectified
  /// into a differently-shaped output.
  static func lowerLeftOrigin(
    _ warp: simd_float3x3, sourceHeight: CGFloat, destinationHeight: CGFloat
  ) -> simd_float3x3 {
    flip(height: destinationHeight) * warp * flip(height: sourceHeight)
  }

  /// Rescales a homography written for full-resolution source pixels so it
  /// can be applied to a copy downscaled by `scale`.
  static func scalingSource(_ warp: simd_float3x3, by scale: CGFloat) -> simd_float3x3 {
    let inverse = Float(1 / scale)
    return warp * simd_float3x3(diagonal: SIMD3<Float>(inverse, inverse, 1))
  }

  /// Rescales a homography so its *output* lands in a space downscaled by
  /// `scale`. The counterpart to `scalingSource`; both are needed when a
  /// warp maps between two images that are each being worked on at reduced
  /// resolution.
  static func scalingDestination(_ warp: simd_float3x3, by scale: CGFloat) -> simd_float3x3 {
    simd_float3x3(diagonal: SIMD3<Float>(Float(scale), Float(scale), 1)) * warp
  }

  /// The homography carrying one shot's pixels onto another's, by way of the
  /// plane they both look at: undo `source`'s projection onto the plane,
  /// then reapply `destination`'s.
  ///
  /// Exact only for content actually lying on the plane. A puzzle box stands
  /// a few centimetres above the table ARKit measured, so its pixels land
  /// close but not right — which makes this a *seed* for registration rather
  /// than a substitute for it.
  static func relative(
    from source: PlaneCaptureGeometry, to destination: PlaneCaptureGeometry
  ) -> simd_float3x3? {
    guard let fromImage = invert(planeToImage(source)) else { return nil }
    return planeToImage(destination) * fromImage
  }

  private static func flip(height: CGFloat) -> simd_float3x3 {
    simd_float3x3(
      rows: [
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(0, -1, Float(height)),
        SIMD3<Float>(0, 0, 1),
      ])
  }

  /// `simd_inverse`, but nil rather than a matrix of infinities when the
  /// input is singular — a degenerate pose must skip rectification, not
  /// poison every pixel downstream.
  private static func invert(_ value: simd_float3x3) -> simd_float3x3? {
    guard value.determinant.isFinite, abs(value.determinant) > .ulpOfOne else { return nil }
    let inverse = value.inverse
    let finite = (0..<3).allSatisfy {
      inverse[$0].x.isFinite && inverse[$0].y.isFinite && inverse[$0].z.isFinite
    }
    return finite ? inverse : nil
  }
}
