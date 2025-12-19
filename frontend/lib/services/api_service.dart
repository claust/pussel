import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../config/api_config.dart';
import '../models/piece.dart';
import '../models/puzzle.dart';

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
  Future<Puzzle> uploadPuzzle(File imageFile) async {
    try {
      FormData formData;

      if (kIsWeb) {
        // For web, we need to handle file uploads differently
        // The path isn't a real filesystem path in web
        final fileName = 'puzzle_${DateTime.now().millisecondsSinceEpoch}.jpg';
        formData = FormData.fromMap({
          'file': MultipartFile.fromBytes(
            // For web, we would need access to the bytes
            // This is a limitation of the current example
            // In a real app, you'd need to store the image data
            [],
            filename: fileName,
          ),
        });

        // In a real implementation, you would capture the image bytes
        // from the camera and use those directly
      } else {
        // Mobile/desktop platforms
        final String fileName = imageFile.path.split('/').last;
        formData = FormData.fromMap({
          'file': await MultipartFile.fromFile(
            imageFile.path,
            filename: fileName,
          ),
        });
      }

      final response = await _dio.post(
        ApiConfig.uploadPuzzleEndpoint,
        data: formData,
      );

      if (response.statusCode == 200) {
        final puzzle = Puzzle.fromJson(response.data as Map<String, dynamic>);
        // Save the local path for convenience
        return Puzzle(
          puzzleId: puzzle.puzzleId,
          imageUrl: puzzle.imageUrl,
          localImagePath: imageFile.path,
        );
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
  Future<Piece> processPiece(String puzzleId, File pieceImage) async {
    try {
      FormData formData;

      if (kIsWeb) {
        // For web, we need to handle file uploads differently
        final fileName = 'piece_${DateTime.now().millisecondsSinceEpoch}.jpg';
        formData = FormData.fromMap({
          'file': MultipartFile.fromBytes(
            // For web, we would need access to the bytes
            // This is a limitation of the current example
            [],
            filename: fileName,
          ),
        });

        // In a real implementation, you would capture the image bytes
        // from the camera and use those directly
      } else {
        // Mobile/desktop platforms
        final String fileName = pieceImage.path.split('/').last;
        formData = FormData.fromMap({
          'file': await MultipartFile.fromFile(
            pieceImage.path,
            filename: fileName,
          ),
        });
      }

      final response = await _dio.post(
        ApiConfig.processPieceEndpoint(puzzleId),
        data: formData,
      );

      if (response.statusCode == 200) {
        final piece = Piece.fromJson(response.data as Map<String, dynamic>);
        // Save the local path for convenience
        return Piece(
          position: piece.position,
          confidence: piece.confidence,
          rotation: piece.rotation,
          localImagePath: pieceImage.path,
        );
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
