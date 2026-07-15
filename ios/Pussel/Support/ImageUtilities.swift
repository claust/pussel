import UIKit

enum ImageUtilities {
    /// Backend rejects uploads above 10 MB (413).
    static let maxUploadBytes = 10 * 1024 * 1024

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

        var quality = quality
        var data = upright.jpegData(compressionQuality: quality)
        while let encoded = data, encoded.count > maxUploadBytes, quality > 0.4 {
            quality -= 0.15
            data = upright.jpegData(compressionQuality: quality)
        }
        return data
    }

    /// Decodes a "data:image/...;base64,..." string (or bare base64) to bytes.
    static func decodeDataURL(_ string: String) -> Data? {
        guard let comma = string.firstIndex(of: ",") else {
            return Data(base64Encoded: string)
        }
        return Data(base64Encoded: String(string[string.index(after: comma)...]))
    }
}
