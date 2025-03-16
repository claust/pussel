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
      final String fileName = imageFile.path.split('/').last;
      final FormData formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
        ),
      });

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
      final String fileName = pieceImage.path.split('/').last;
      final FormData formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          pieceImage.path,
          filename: fileName,
        ),
      });

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
