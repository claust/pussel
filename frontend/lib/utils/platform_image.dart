import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// A cross-platform image container that works on both mobile and web.
///
/// On mobile/desktop, it stores a [File] reference.
/// On web, it stores the image bytes as [Uint8List].
class PlatformImage {
  PlatformImage._({this.file, this.bytes, this.path})
    : assert(
        file != null || bytes != null,
        'Either file or bytes must be provided',
      );

  /// Creates a PlatformImage from a File (mobile/desktop).
  factory PlatformImage.fromFile(File file) =>
      PlatformImage._(file: file, path: file.path);

  /// Creates a PlatformImage from bytes (web).
  factory PlatformImage.fromBytes(Uint8List bytes, {String? path}) =>
      PlatformImage._(bytes: bytes, path: path);

  /// Creates a PlatformImage from bytes only (useful for processed images).
  factory PlatformImage.fromBytesOnly(Uint8List bytes) =>
      PlatformImage._(bytes: bytes);

  /// Creates a PlatformImage from an XFile path and bytes.
  /// This is the recommended way to create a PlatformImage after taking a photo.
  static Future<PlatformImage> fromXFile(dynamic xFile) async {
    final bytes = await xFile.readAsBytes() as Uint8List;
    if (kIsWeb) {
      return PlatformImage.fromBytes(bytes, path: xFile.path as String);
    } else {
      return PlatformImage._(
        file: File(xFile.path as String),
        bytes: bytes,
        path: xFile.path as String,
      );
    }
  }

  final File? file;
  final Uint8List? bytes;
  final String? path;

  /// Returns the image bytes. On mobile, reads from file if bytes not cached.
  Future<Uint8List> getBytes() async {
    if (bytes != null) return bytes!;
    if (file != null) return file!.readAsBytes();
    throw StateError('No image data available');
  }

  /// Returns bytes synchronously if available, null otherwise.
  Uint8List? get bytesSync => bytes;
}

/// A widget that displays a [PlatformImage] correctly on all platforms.
class PlatformImageWidget extends StatelessWidget {
  const PlatformImageWidget({
    required this.image,
    this.fit = BoxFit.cover,
    this.width,
    this.height,
    super.key,
  });

  final PlatformImage image;
  final BoxFit fit;
  final double? width;
  final double? height;

  @override
  Widget build(BuildContext context) {
    // Prefer bytes if available (works on all platforms)
    if (image.bytes != null) {
      return Image.memory(image.bytes!, fit: fit, width: width, height: height);
    }

    // Fall back to file for mobile/desktop
    if (!kIsWeb && image.file != null) {
      return Image.file(image.file!, fit: fit, width: width, height: height);
    }

    // Fallback placeholder
    return Container(
      width: width,
      height: height,
      color: Colors.grey[300],
      child: const Center(child: Icon(Icons.image_not_supported)),
    );
  }
}
