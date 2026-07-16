import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UIKit

enum ImageUtilities {
  /// Backend rejects uploads above 10 MB (413).
  static let maxUploadBytes = 10 * 1024 * 1024

  /// Long-side cap for the puzzle's zoom copy — the straightened crop itself,
  /// not the photo it came from (see
  /// `perspectiveCorrected(from:corners:maxDimension:)`).
  ///
  /// The uploaded image is capped at 1920 by `normalizedJPEG`, and the crop
  /// taken from it is smaller again, so on a 3× phone the server's copy is
  /// already at one image pixel per device pixel when it merely *fits* the
  /// screen: pinching it magnifies JPEG artefacts from the first gesture.
  ///
  /// True magnification costs pixels fast — the crop needs `n` × the screen's
  /// 1206px width to stay sharp at `n`×, and its area (and decoded memory)
  /// grows with the square. 3600 buys roughly 2× before softness at ~35 MB
  /// decoded; 3× would want ~76 MB for one image, which is not a trade worth
  /// making for the top of a zoom range. Above 2× the viewer is upscaling —
  /// still a far better look at a piece than the 350pt inline overlay, but
  /// not new detail.
  static let zoomMaxDimension: CGFloat = 3600

  /// Long-side cap for the *photo* kept to re-warp into the zoom copy.
  ///
  /// Sized off `zoomMaxDimension`, not chosen independently: the crop is a
  /// quadrilateral inside the photo, so the photo must be the larger of the
  /// two for the crop to reach its own cap. A puzzle shot at an angle can run
  /// close to the frame's diagonal, so the margin here is what keeps the crop
  /// from landing short of `zoomMaxDimension` on exactly the tilted shots
  /// that need straightening most.
  static let zoomSourceMaxDimension: CGFloat = 4400

  /// Downsamples JPEG data to a small thumbnail (long side ~`maxPixel`) using
  /// ImageIO, which decodes only what the thumbnail needs. Used for the
  /// home-screen puzzle list so a full-size image isn't held in memory or
  /// decoded just to draw a small card.
  static func thumbnailJPEG(from data: Data, maxPixel: Int = 240) -> Data? {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: maxPixel,
    ]
    guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    else {
      return nil
    }
    let output = NSMutableData()
    guard
      let destination = CGImageDestinationCreateWithData(output, "public.jpeg" as CFString, 1, nil)
    else {
      return nil
    }
    CGImageDestinationAddImage(
      destination, thumbnail, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else { return nil }
    return output as Data
  }

  /// Redraws the image upright (baking in EXIF orientation) and downscales it
  /// so the long side is at most `maxDimension`, then encodes JPEG. Reduces
  /// quality stepwise if the result would exceed the backend's upload limit.
  static func normalizedJPEG(
    from image: UIImage, maxDimension: CGFloat = 1920, quality: CGFloat = 0.9
  ) -> Data? {
    let pixelSize = CGSize(
      width: image.size.width * image.scale, height: image.size.height * image.scale)
    let longSide = max(pixelSize.width, pixelSize.height)
    guard longSide > 0 else { return nil }
    let ratio = min(1, maxDimension / longSide)
    let targetSize = CGSize(
      width: (pixelSize.width * ratio).rounded(.down),
      height: (pixelSize.height * ratio).rounded(.down))

    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let upright = UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: targetSize))
    }

    let minQuality: CGFloat = 0.4
    var quality = quality
    var data = upright.jpegData(compressionQuality: quality)
    while let encoded = data, encoded.count > maxUploadBytes, quality > minQuality {
      quality = max(quality - 0.15, minQuality)
      data = upright.jpegData(compressionQuality: quality)
    }
    // Fail fast rather than letting the backend reject the upload with 413.
    guard let encoded = data, encoded.count <= maxUploadBytes else { return nil }
    return encoded
  }

  // MARK: Perspective correction

  /// Perspective-corrects `image` to the quadrilateral `corners`, with the
  /// result's long side at most `maxDimension`.
  ///
  /// Returns an image rather than JPEG bytes so a caller that still has to
  /// rotate it (see `AppModel.acceptTrim`) can do so before encoding, and pay
  /// one lossy pass instead of two.
  ///
  /// This reproduces locally the crop the backend's detect-frame endpoint
  /// already returned, but from a higher-resolution source: `corners` are
  /// normalized, so they describe the same region of the photo at any
  /// resolution, and re-running the warp here avoids uploading a second,
  /// much larger copy of the photo just to get a sharper picture back.
  ///
  /// The output frames the same region as the server's `trimmed_image` and
  /// carries its aspect ratio (the target size below mirrors the dst_w/dst_h
  /// formula in backend/app/services/puzzle_detector.py's `warp`), which is
  /// what makes the two interchangeable for display: piece positions are
  /// normalized to the trimmed puzzle's frame, so they land in the same place
  /// on either copy. It is deliberately not pixel-identical — Core Image and
  /// OpenCV resample differently — and is only ever shown, never uploaded or
  /// measured against.
  ///
  /// - Parameters:
  ///   - image: the full-resolution photo the corners were detected in.
  ///   - corners: the quad to straighten, normalized to `image`'s frame with
  ///     a top-left origin, as returned by `DetectFrameResponse`.
  ///   - maxDimension: long-side cap for the result.
  /// - Returns: the straightened image, or nil when it can't be decoded, the
  ///   quad is degenerate, or rendering fails.
  static func perspectiveCorrected(
    from image: UIImage,
    corners: QuadCorners,
    maxDimension: CGFloat = zoomMaxDimension
  ) -> UIImage? {
    // Bake any EXIF orientation into the pixels first: the corners are
    // normalized to the upright photo the user saw and the backend detected
    // in (it applies exif_transpose), not to the raw sensor buffer.
    let upright =
      (image.imageOrientation == .up && image.cgImage != nil) ? image : redrawnUpright(image)
    guard let cgImage = upright.cgImage else { return nil }
    let width = CGFloat(cgImage.width)
    let height = CGFloat(cgImage.height)
    guard width > 0, height > 0 else { return nil }

    // Core Image's coordinate space has its origin at the bottom left, so the
    // top-left-origin normalized corners flip in y.
    func point(_ corner: NormalizedPoint) -> CGPoint {
      CGPoint(x: CGFloat(corner.x) * width, y: (1 - CGFloat(corner.y)) * height)
    }
    let topLeft = point(corners.topLeft)
    let topRight = point(corners.topRight)
    let bottomRight = point(corners.bottomRight)
    let bottomLeft = point(corners.bottomLeft)

    // Each output side takes the longer of the two source edges it could come
    // from, so a quad seen at an angle straightens to the larger, undistorted
    // rectangle rather than the foreshortened one.
    let targetWidth = max(distance(topLeft, topRight), distance(bottomLeft, bottomRight))
    let targetHeight = max(distance(topLeft, bottomLeft), distance(topRight, bottomRight))
    guard targetWidth >= 1, targetHeight >= 1 else { return nil }

    let filter = CIFilter.perspectiveCorrection()
    filter.inputImage = CIImage(cgImage: cgImage)
    filter.topLeft = topLeft
    filter.topRight = topRight
    filter.bottomRight = bottomRight
    filter.bottomLeft = bottomLeft
    filter.crop = true
    guard let corrected = filter.outputImage, corrected.extent.width > 0,
      corrected.extent.height > 0
    else {
      return nil
    }

    // Never upscale: a photo smaller than the cap has no more detail to give.
    let scale = min(1, maxDimension / max(targetWidth, targetHeight))
    // Scale from Core Image's own rectified extent onto the target size,
    // rather than by a single factor, so the result carries the aspect ratio
    // computed above even where the two rectifications disagree slightly.
    let transform = CGAffineTransform(
      scaleX: targetWidth * scale / corrected.extent.width,
      y: targetHeight * scale / corrected.extent.height)
    let scaled = corrected.transformed(by: transform)

    let context = CIContext()
    guard let output = context.createCGImage(scaled, from: scaled.extent) else { return nil }
    return UIImage(cgImage: output)
  }

  private static func distance(_ from: CGPoint, _ to: CGPoint) -> CGFloat {
    CGFloat(hypot(to.x - from.x, to.y - from.y))
  }

  /// Returns `image` rotated clockwise by `quarterTurns` × 90°. When
  /// `quarterTurns` normalizes to 0 the image is returned unchanged. For a
  /// non-zero turn, any existing EXIF orientation is baked into the pixels
  /// first (so the quarter-turn is correct even for a camera image that
  /// arrives rotated or mirrored), then the turn is applied by reinterpreting
  /// orientation — cheap, no resampling. `normalizedJPEG` bakes the turn into
  /// the pixels for upload (see `rotatedJPEG`).
  static func rotated(_ image: UIImage, quarterTurns: Int) -> UIImage {
    let turns = ((quarterTurns % 4) + 4) % 4
    guard turns != 0 else { return image }
    // Redraw when the orientation needs baking, or when there is no backing
    // CGImage to reinterpret (a UIImage can be CIImage-only) — the redraw
    // always yields an upright, CGImage-backed bitmap.
    let upright =
      (image.imageOrientation == .up && image.cgImage != nil) ? image : redrawnUpright(image)
    guard let cgImage = upright.cgImage else { return image }
    let orientation: UIImage.Orientation
    switch turns {
    case 1: orientation = .right
    case 2: orientation = .down
    default: orientation = .left
    }
    return UIImage(cgImage: cgImage, scale: upright.scale, orientation: orientation)
  }

  /// Redraws `image` with its EXIF orientation baked into the pixels, yielding
  /// an equivalent `.up`-oriented image.
  private static func redrawnUpright(_ image: UIImage) -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = image.scale
    return UIGraphicsImageRenderer(size: image.size, format: format).image { _ in
      image.draw(in: CGRect(origin: .zero, size: image.size))
    }
  }

  /// Rotates JPEG `data` clockwise by `quarterTurns` × 90° and re-encodes it
  /// with the rotation baked into the pixels. Returns the original data
  /// unchanged when no rotation is requested, or `nil` when a requested
  /// rotation cannot be applied — so a caller never uploads an unrotated image
  /// believing it was rotated.
  ///
  static func rotatedJPEG(from data: Data, quarterTurns: Int) -> Data? {
    guard ((quarterTurns % 4) + 4) % 4 != 0 else { return data }
    guard let image = UIImage(data: data) else { return nil }
    return normalizedJPEG(from: rotated(image, quarterTurns: quarterTurns))
  }

  /// Decodes a "data:image/...;base64,..." string (or bare base64) to bytes.
  static func decodeDataURL(_ string: String) -> Data? {
    guard let comma = string.firstIndex(of: ",") else {
      return Data(base64Encoded: string)
    }
    return Data(base64Encoded: String(string[string.index(after: comma)...]))
  }

  // MARK: Alpha trimming

  /// Returns the smallest pixel-space rect containing every pixel whose
  /// alpha exceeds `threshold` (0...255), or `nil` when `image` is empty or
  /// every pixel is at/below the threshold (fully transparent). Works in
  /// `image`'s own `cgImage` pixel space, so it assumes `.up` orientation —
  /// true for the images this app feeds it (backend-generated PNGs and
  /// already-baked captures; see `rotated(_:quarterTurns:)`).
  ///
  /// The default threshold is high because segmentation mattes (rembg) can
  /// leave a faint alpha halo of background across the frame; counting those
  /// pixels would inflate the box to well beyond the actual subject.
  static func alphaBoundingBox(of image: UIImage, threshold: UInt8 = 128) -> CGRect? {
    guard let cgImage = image.cgImage, cgImage.width > 0, cgImage.height > 0 else { return nil }
    guard let pixels = rgbaPixels(of: cgImage) else { return nil }
    return opaqueBounds(
      in: pixels, width: cgImage.width, height: cgImage.height, threshold: threshold)
  }

  /// Renders `cgImage` into a top-left-origin RGBA8 buffer the same pixel
  /// size as the image, so alpha can be read byte-by-byte.
  private static func rgbaPixels(of cgImage: CGImage) -> [UInt8]? {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerRow = 4 * width
    var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
    guard
      let context = CGContext(
        data: &pixels,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        // Explicit big-endian byte order pins the memory layout to R,G,B,A
        // regardless of host endianness, so index 3 is always alpha.
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          | CGBitmapInfo.byteOrder32Big.rawValue)
    else { return nil }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    return pixels
  }

  /// Scans an RGBA8 `pixels` buffer for the smallest rect containing every
  /// pixel whose alpha exceeds `threshold`, or `nil` when none does.
  private static func opaqueBounds(in pixels: [UInt8], width: Int, height: Int, threshold: UInt8)
    -> CGRect?
  {
    let bytesPerRow = 4 * width
    var minX = width
    var minY = height
    var maxX = -1
    var maxY = -1
    for y in 0..<height {
      let rowStart = y * bytesPerRow
      for x in 0..<width where pixels[rowStart + x * 4 + 3] > threshold {
        minX = min(minX, x)
        maxX = max(maxX, x)
        minY = min(minY, y)
        maxY = max(maxY, y)
      }
    }
    guard maxX >= minX, maxY >= minY else { return nil }
    return CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
  }

  /// Crops `image` to its alpha bounding box (see `alphaBoundingBox`),
  /// trimming a transparent margin such as the backend's cleaned piece
  /// PNGs. Returns `image` itself (same instance) when there is no alpha
  /// bbox, or it already covers the whole image, so callers can cheaply
  /// detect "nothing to trim" via reference identity.
  static func croppedToAlphaBounds(_ image: UIImage) -> UIImage {
    guard let cgImage = image.cgImage,
      let bbox = alphaBoundingBox(of: image),
      bbox.width > 0, bbox.height > 0,
      Int(bbox.width) < cgImage.width || Int(bbox.height) < cgImage.height
    else {
      return image
    }
    guard let cropped = cgImage.cropping(to: bbox) else { return image }
    return UIImage(cgImage: cropped, scale: image.scale, orientation: image.imageOrientation)
  }

  private static let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]

  /// Crops PNG `data` to its alpha bounding box and re-encodes as PNG.
  /// Returns `data` unchanged when it isn't PNG (a raw JPEG capture has no
  /// alpha channel at all, so skip the decode+scan entirely — cheap
  /// signature check), can't be decoded, or has no transparent margin to
  /// trim.
  static func alphaTrimmedPNG(from data: Data) -> Data {
    guard data.starts(with: pngSignature), let image = UIImage(data: data) else { return data }
    let cropped = croppedToAlphaBounds(image)
    guard cropped !== image, let pngData = cropped.pngData() else { return data }
    return pngData
  }
}
