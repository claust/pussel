import UIKit
import XCTest

@testable import Pussel

final class ImageUtilitiesTests: XCTestCase {
    /// A solid-colour image at an exact pixel size (scale 1) for dimension checks.
    private func makeImage(width: Int, height: Int) -> UIImage {
        let size = CGSize(width: width, height: height)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            UIColor.red.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
    }

    // MARK: rotated(_:quarterTurns:)

    func testRotatedNormalizesFullTurnToIdentity() {
        let image = makeImage(width: 8, height: 4)
        // 4 quarter-turns (and 0) normalize to no rotation: same instance back.
        XCTAssertIdentical(ImageUtilities.rotated(image, quarterTurns: 0), image)
        XCTAssertIdentical(ImageUtilities.rotated(image, quarterTurns: 4), image)
        XCTAssertIdentical(ImageUtilities.rotated(image, quarterTurns: -4), image)
    }

    func testRotatedQuarterTurnSwapsReportedSize() {
        let image = makeImage(width: 8, height: 4)
        let turned = ImageUtilities.rotated(image, quarterTurns: 1)
        XCTAssertEqual(turned.size, CGSize(width: 4, height: 8))
    }

    func testRotatedHalfTurnKeepsSize() {
        let image = makeImage(width: 8, height: 4)
        let turned = ImageUtilities.rotated(image, quarterTurns: 2)
        XCTAssertEqual(turned.size, CGSize(width: 8, height: 4))
    }

    func testRotatedNegativeTurnNormalizes() {
        let image = makeImage(width: 8, height: 4)
        // -1 (≡ 3) is still a 90° turn, so the reported size swaps.
        let turned = ImageUtilities.rotated(image, quarterTurns: -1)
        XCTAssertEqual(turned.size, CGSize(width: 4, height: 8))
    }

    // MARK: rotatedJPEG(from:quarterTurns:)

    func testRotatedJPEGZeroTurnsReturnsInputUnchanged() throws {
        let data = try XCTUnwrap(makeImage(width: 8, height: 4).jpegData(compressionQuality: 0.9))
        XCTAssertEqual(ImageUtilities.rotatedJPEG(from: data, quarterTurns: 0), data)
        XCTAssertEqual(ImageUtilities.rotatedJPEG(from: data, quarterTurns: 4), data)
    }

    func testRotatedJPEGInvalidDataZeroTurnsReturnsInput() {
        let bogus = Data([0x00, 0x01, 0x02])
        XCTAssertEqual(ImageUtilities.rotatedJPEG(from: bogus, quarterTurns: 0), bogus)
    }

    func testRotatedJPEGInvalidDataWithRotationReturnsNil() {
        // A requested rotation that can't be applied must fail loudly (nil)
        // rather than silently returning the unrotated bytes.
        let bogus = Data([0x00, 0x01, 0x02])
        XCTAssertNil(ImageUtilities.rotatedJPEG(from: bogus, quarterTurns: 1))
    }

    func testRotatedJPEGSwapsDimensionsOnQuarterTurn() throws {
        let data = try XCTUnwrap(makeImage(width: 8, height: 4).jpegData(compressionQuality: 0.9))
        let rotatedData = try XCTUnwrap(ImageUtilities.rotatedJPEG(from: data, quarterTurns: 1))
        let decoded = try XCTUnwrap(UIImage(data: rotatedData))
        XCTAssertEqual(decoded.size.width * decoded.scale, 4)
        XCTAssertEqual(decoded.size.height * decoded.scale, 8)
    }
}
