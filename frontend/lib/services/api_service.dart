import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../config/api_config.dart';
import '../models/piece.dart';
import '../models/puzzle.dart';
import '../utils/platform_image.dart';

class ApiService {
  ApiService() {
    _dio = Dio(
      BaseOptions(
        baseUrl: ApiConfig.baseUrl,
        connectTimeout: const Duration(
          milliseconds: ApiConfig.connectionTimeout,
        ),
        receiveTimeout: const Duration(milliseconds: ApiConfig.receiveTimeout),
      ),
    );

    // Add logging interceptor for debugging
    if (kDebugMode) {
      _dio.interceptors.add(
        LogInterceptor(requestBody: true, responseBody: true),
      );
    }
  }
  late final Dio _dio;

  // Check if API is available
  Future<bool> checkHealth() async {
    try {
      final response = await _dio.get(ApiConfig.healthEndpoint);
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('API health check failed: $e');
      return false;
    }
  }

  // Upload a complete puzzle image
  Future<Puzzle> uploadPuzzle(PlatformImage image) async {
    try {
      final bytes = await image.getBytes();
      final fileName = 'puzzle_${DateTime.now().millisecondsSinceEpoch}.jpg';

      final formData = FormData.fromMap({
        'file': MultipartFile.fromBytes(bytes, filename: fileName),
      });

      final response = await _dio.post(
        ApiConfig.uploadPuzzleEndpoint,
        data: formData,
      );

      if (response.statusCode == 200) {
        return Puzzle.fromJson(response.data as Map<String, dynamic>);
      } else {
        throw Exception(
          'Failed to upload puzzle image: ${response.statusCode}',
        );
      }
    } catch (e) {
      debugPrint('Error uploading puzzle: $e');
      rethrow;
    }
  }

  // Process a puzzle piece
  Future<Piece> processPiece(String puzzleId, PlatformImage pieceImage) async {
    try {
      final bytes = await pieceImage.getBytes();
      final fileName = 'piece_${DateTime.now().millisecondsSinceEpoch}.jpg';

      final formData = FormData.fromMap({
        'file': MultipartFile.fromBytes(bytes, filename: fileName),
      });

      final response = await _dio.post(
        ApiConfig.processPieceEndpoint(puzzleId),
        data: formData,
      );

      if (response.statusCode == 200) {
        final piece = Piece.fromJson(response.data as Map<String, dynamic>);
        // Attach the image to the piece
        return piece.copyWithImage(pieceImage);
      } else {
        throw Exception(
          'Failed to process puzzle piece: ${response.statusCode}',
        );
      }
    } catch (e) {
      debugPrint('Error processing piece: $e');
      rethrow;
    }
  }
}
