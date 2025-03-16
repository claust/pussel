import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../services/camera_service.dart';
import 'puzzle_screen.dart';

enum CameraMode { puzzle, piece }

class CameraScreen extends StatefulWidget {
  const CameraScreen({
    required this.mode,
    super.key,
    this.puzzleId,
    this.puzzleImage, // Add puzzle image parameter
  });

  final CameraMode mode;
  final String? puzzleId; // Only needed for piece mode
  final File? puzzleImage; // Reference to the puzzle image for context

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

    if (kIsWeb) {
      // On web, show a note about limitations but still try to initialize
      setState(() {
        _errorMessage =
            'Camera functionality is limited on web. Some features may not work properly.';
      });
    }

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
      if (_errorMessage == null || !kIsWeb) {
        _errorMessage = null;
      }
    });

    try {
      await _cameraService.initializeCamera();
      setState(() {
        _isInitializing = false;
      });
    } catch (e) {
      setState(() {
        _isInitializing = false;
        _errorMessage =
            kIsWeb
                ? 'Camera access may be limited in web browsers. Please try a mobile device for full functionality.'
                : 'Failed to initialize camera: $e';
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
    body: Stack(
      children: [
        // When in piece mode, show the puzzle image in the background with reduced opacity
        if (widget.mode == CameraMode.piece && widget.puzzleImage != null)
          Opacity(
            opacity: 0.3,
            child: Container(
              color: Colors.black,
              child: Center(
                child: Image.file(widget.puzzleImage!, fit: BoxFit.contain),
              ),
            ),
          ),

        _buildBody(),

        // Helper overlay message when in piece mode
        if (widget.mode == CameraMode.piece)
          Positioned(
            top: 16,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(8),
              color: Colors.black54,
              child: const Text(
                'Align the piece within the frame',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white),
              ),
            ),
          ),
      ],
    ),
    floatingActionButton: _buildFloatingActionButton(),
    floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
  );

  Widget _buildBody() {
    if (_errorMessage != null && !_cameraService.isInitialized) {
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
        if (kIsWeb && _errorMessage != null)
          Container(
            padding: const EdgeInsets.all(8),
            color: Colors.amber.shade100,
            width: double.infinity,
            child: Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.brown),
            ),
          ),
        Expanded(
          child: ClipRRect(
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(8.0),
              bottomRight: Radius.circular(8.0),
            ),
            child:
                _cameraService.controller != null
                    ? CameraPreview(_cameraService.controller!)
                    : const Center(child: Text('Camera not available')),
          ),
        ),
        const SizedBox(height: 80), // Space for the floating action button
      ],
    );
  }

  Widget _buildFloatingActionButton() =>
      (_isInitializing ||
              (_errorMessage != null && !_cameraService.isInitialized))
          ? Container()
          : FloatingActionButton(
            onPressed: _takePicture,
            tooltip: 'Take Picture',
            child: const Icon(Icons.camera, size: 36),
          );
}
