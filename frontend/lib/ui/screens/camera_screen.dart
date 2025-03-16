import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../../services/camera_service.dart';
import 'puzzle_screen.dart';

enum CameraMode { puzzle, piece }

class CameraScreen extends StatefulWidget {
  const CameraScreen({required this.mode, super.key, this.puzzleId});

  final CameraMode mode;
  final String? puzzleId; // Only needed for piece mode

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  final CameraService _cameraService = CameraService();
  bool _isInitializing = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // App state changed before we got the chance to initialize the camera
    if (_cameraService.controller == null ||
        !_cameraService.controller!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _cameraService.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    setState(() {
      _isInitializing = true;
      _errorMessage = null;
    });

    try {
      await _cameraService.initializeCamera();
      setState(() {
        _isInitializing = false;
      });
    } catch (e) {
      setState(() {
        _isInitializing = false;
        _errorMessage = 'Failed to initialize camera: $e';
      });
    }
  }

  Future<void> _takePicture() async {
    if (_isInitializing || !_cameraService.isInitialized) {
      return;
    }

    try {
      final File? imageFile = await _cameraService.takePicture();

      if (imageFile != null && mounted) {
        if (widget.mode == CameraMode.puzzle) {
          // Navigate to puzzle screen with image
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => PuzzleScreen(puzzleImage: imageFile),
            ),
          );
        } else if (widget.mode == CameraMode.piece && widget.puzzleId != null) {
          // Return to puzzle screen with piece image
          Navigator.pop(context, imageFile);
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error taking picture: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: Text(
        widget.mode == CameraMode.puzzle
            ? 'Take Puzzle Photo'
            : 'Take Piece Photo',
      ),
    ),
    body: _buildBody(),
    floatingActionButton: _buildFloatingActionButton(),
    floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
  );

  Widget _buildBody() {
    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(_errorMessage!, textAlign: TextAlign.center),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _initializeCamera,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (_isInitializing) {
      return const Center(child: CircularProgressIndicator());
    }

    return Column(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(8.0),
              bottomRight: Radius.circular(8.0),
            ),
            child: CameraPreview(_cameraService.controller!),
          ),
        ),
        const SizedBox(height: 80), // Space for the floating action button
      ],
    );
  }

  Widget _buildFloatingActionButton() =>
      _isInitializing || _errorMessage != null
          ? Container()
          : FloatingActionButton(
            onPressed: _takePicture,
            tooltip: 'Take Picture',
            child: const Icon(Icons.camera, size: 36),
          );
}
