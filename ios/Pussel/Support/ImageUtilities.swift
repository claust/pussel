import ImageIO
import UIKit

enum ImageUtilities {
  /// Backend rejects uploads above 10 MB (413).
  static let maxUploadBytes = 10 * 1024 * 1024

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
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
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
