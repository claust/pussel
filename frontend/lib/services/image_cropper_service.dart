import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../models/grid_size.dart';
import '../utils/platform_image.dart';

/// Service for cropping and rotating puzzle piece images.
class ImageCropperService {
  /// Crops a cell from the puzzle image based on grid position and applies rotation.
  ///
  /// [puzzleImage] - The full puzzle image to crop from.
  /// [gridSize] - The grid size (2x2 or 3x3).
  /// [cellIndex] - The cell index (0-based, row-major order).
  /// [rotationDegrees] - Rotation to apply (0, 90, 180, or 270).
  Future<PlatformImage> cropCell(
    PlatformImage puzzleImage,
    GridSize gridSize,
    int cellIndex,
    int rotationDegrees,
  ) async {
    final bytes = await puzzleImage.getBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Failed to decode puzzle image');
    }

    final dimension = gridSize.dimension;
    final row = cellIndex ~/ dimension;
    final col = cellIndex % dimension;

    final cellWidth = decoded.width ~/ dimension;
    final cellHeight = decoded.height ~/ dimension;

    // Crop the cell
    var cropped = img.copyCrop(
      decoded,
      x: col * cellWidth,
      y: row * cellHeight,
      width: cellWidth,
      height: cellHeight,
    );

    // Apply rotation if needed
    if (rotationDegrees != 0) {
      cropped = img.copyRotate(cropped, angle: rotationDegrees.toDouble());
    }

    // Encode back to JPEG bytes
    final resultBytes = Uint8List.fromList(img.encodeJpg(cropped, quality: 90));
    return PlatformImage.fromBytesOnly(resultBytes);
  }

  /// Gets a preview of the cropped cell without rotation for display.
  Future<Uint8List> getCellPreview(
    PlatformImage puzzleImage,
    GridSize gridSize,
    int cellIndex,
  ) async {
    final bytes = await puzzleImage.getBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Failed to decode puzzle image');
    }

    final dimension = gridSize.dimension;
    final row = cellIndex ~/ dimension;
    final col = cellIndex % dimension;

    final cellWidth = decoded.width ~/ dimension;
    final cellHeight = decoded.height ~/ dimension;

    final cropped = img.copyCrop(
      decoded,
      x: col * cellWidth,
      y: row * cellHeight,
      width: cellWidth,
      height: cellHeight,
    );

    return Uint8List.fromList(img.encodeJpg(cropped, quality: 85));
  }
}
