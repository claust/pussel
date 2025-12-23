import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

import '../utils/platform_image.dart';

class CameraService {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;
  CameraController? get controller => _controller;

  // Initialize the camera
  Future<void> initializeCamera() async {
    try {
      // Get available cameras
      _cameras = await availableCameras();

      if (_cameras == null || _cameras!.isEmpty) {
        throw CameraException('No cameras', 'No cameras available on device');
      }

      // Use the first camera (usually the back camera)
      final camera = _cameras!.first;

      // Initialize controller
      _controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
      );

      // Initialize the controller
      await _controller!.initialize();
      _isInitialized = true;
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      _isInitialized = false;
      rethrow;
    }
  }

  // Take a picture and return a cross-platform image
  Future<PlatformImage?> takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      debugPrint('Camera controller not initialized');
      return null;
    }

    try {
      // Take the picture
      final XFile photo = await _controller!.takePicture();

      // Read the bytes (works on all platforms)
      final bytes = await photo.readAsBytes();

      if (kIsWeb) {
        // On web, just use bytes
        return PlatformImage.fromBytes(bytes, path: photo.path);
      } else {
        // On mobile/desktop, save to temp directory and create PlatformImage
        final Directory tempDir = await getTemporaryDirectory();
        final String filePath = path.join(
          tempDir.path,
          '${DateTime.now().millisecondsSinceEpoch}.jpg',
        );

        // Write bytes to file
        final File newFile = File(filePath);
        await newFile.writeAsBytes(bytes);

        return PlatformImage.fromFile(newFile);
      }
    } catch (e) {
      debugPrint('Error taking picture: $e');
      return null;
    }
  }

  // Dispose the camera controller
  void dispose() {
    _controller?.dispose();
    _isInitialized = false;
  }
}
