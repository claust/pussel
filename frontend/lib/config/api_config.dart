import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

class ApiConfig {
  // Detect the platform and use appropriate URL
  static String get baseUrl {
    if (kIsWeb) {
      return 'http://localhost:8000'; // Use localhost for web
    } else if (Platform.isAndroid) {
      return 'http://10.0.2.2:8000'; // Use this for Android Emulator
    } else if (Platform.isIOS) {
      // For iOS simulator, use localhost
      // For physical iOS device, use your Mac's IP address on the network
      return 'http://192.168.0.133:8000'; // Your Mac's actual IP address
    } else {
      return 'http://localhost:8000'; // Default for other platforms
    }
  }

  // API endpoints
  static const String healthEndpoint = '/health';
  static const String uploadPuzzleEndpoint = '/api/v1/puzzle/upload';
  static String processPieceEndpoint(String puzzleId) =>
      '/api/v1/puzzle/$puzzleId/piece';

  // Timeout durations
  static const int connectionTimeout = 30000; // 30 seconds
  static const int receiveTimeout = 30000; // 30 seconds

  // Upload constraints
  static const int maxUploadSizeMb = 10; // 10 MB
}
