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
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, "public.jpeg" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, thumbnail, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return output as Data
    }

    /// Redraws the image upright (baking in EXIF orientation) and downscales it
    /// so the long side is at most `maxDimension`, then encodes JPEG. Reduces
    /// quality stepwise if the result would exceed the backend's upload limit.
    static func normalizedJPEG(from image: UIImage, maxDimension: CGFloat = 1920, quality: CGFloat = 0.9) -> Data? {
        let pixelSize = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        let longSide = max(pixelSize.width, pixelSize.height)
        guard longSide > 0 else { return nil }
        let ratio = min(1, maxDimension / longSide)
        let targetSize = CGSize(width: (pixelSize.width * ratio).rounded(.down), height: (pixelSize.height * ratio).rounded(.down))

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
        let upright = (image.imageOrientation == .up && image.cgImage != nil) ? image : redrawnUpright(image)
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
}
