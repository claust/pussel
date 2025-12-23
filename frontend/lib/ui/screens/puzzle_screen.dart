import 'package:flutter/material.dart';

import '../../models/piece.dart';
import '../../models/puzzle.dart';
import '../../services/api_service.dart';
import '../../utils/platform_image.dart';
import '../widgets/puzzle_detail.dart';
import 'camera_screen.dart';

class PuzzleScreen extends StatefulWidget {
  const PuzzleScreen({
    required this.puzzleImage,
    super.key,
    this.existingPuzzle,
  });

  final PlatformImage puzzleImage;
  final Puzzle? existingPuzzle;

  @override
  State<PuzzleScreen> createState() => _PuzzleScreenState();
}

class _PuzzleScreenState extends State<PuzzleScreen> {
  final ApiService _apiService = ApiService();
  final List<Piece> _pieces = [];

  bool _isUploading = false;
  bool _isProcessingPiece = false;
  String? _errorMessage;
  bool _isFullScreen = true;

  Puzzle? _puzzle;

  @override
  void initState() {
    super.initState();
    _puzzle = widget.existingPuzzle;

    if (_puzzle == null) {
      _uploadPuzzle();
    }
  }

  Future<void> _uploadPuzzle() async {
    setState(() {
      _isUploading = true;
      _errorMessage = null;
    });

    try {
      final puzzle = await _apiService.uploadPuzzle(widget.puzzleImage);
      setState(() {
        _puzzle = puzzle;
        _isUploading = false;
      });
    } catch (e) {
      setState(() {
        _isUploading = false;
        _errorMessage = 'Failed to upload puzzle: $e';
      });

      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error uploading puzzle: $e')));
      }
    }
  }

  Future<void> _addPuzzlePiece() async {
    if (_puzzle == null) return;

    final result = await Navigator.push<PlatformImage>(
      context,
      MaterialPageRoute(
        builder:
            (context) => CameraScreen(
              mode: CameraMode.piece,
              puzzleId: _puzzle!.puzzleId,
              puzzleImage: widget.puzzleImage,
            ),
      ),
    );

    if (result == null || !mounted) return;

    setState(() {
      _isProcessingPiece = true;
      _errorMessage = null;
    });

    try {
      final piece = await _apiService.processPiece(_puzzle!.puzzleId, result);
      setState(() {
        _pieces.add(piece);
        _isProcessingPiece = false;
        _isFullScreen = true;
      });
    } catch (e) {
      setState(() {
        _isProcessingPiece = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error processing piece: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: const Text('Puzzle Solver'),
      actions: [
        IconButton(
          icon: Icon(_isFullScreen ? Icons.grid_view : Icons.fullscreen),
          onPressed: () {
            setState(() {
              _isFullScreen = !_isFullScreen;
            });
          },
        ),
      ],
    ),
    body: AnimatedSwitcher(
      duration: const Duration(milliseconds: 300),
      transitionBuilder:
          (Widget child, Animation<double> animation) =>
              FadeTransition(opacity: animation, child: child),
      child:
          _errorMessage != null
              ? _buildErrorView()
              : _isUploading
              ? _buildLoadingView('Uploading puzzle...')
              : _isFullScreen
              ? _buildFullScreenView()
              : _buildPuzzleView(),
    ),
    floatingActionButton:
        _puzzle != null
            ? FloatingActionButton(
              onPressed: _isProcessingPiece ? null : _addPuzzlePiece,
              tooltip: 'Add Piece',
              child:
                  _isProcessingPiece
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Icon(Icons.add),
            )
            : null,
  );

  Widget _buildErrorView() => Center(
    child: Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, color: Colors.red, size: 48),
          const SizedBox(height: 16),
          Text(
            _errorMessage!,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          const SizedBox(height: 24),
          ElevatedButton(onPressed: _uploadPuzzle, child: const Text('Retry')),
        ],
      ),
    ),
  );

  Widget _buildLoadingView(String message) => Center(
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const CircularProgressIndicator(),
        const SizedBox(height: 24),
        Text(message, style: Theme.of(context).textTheme.bodyLarge),
      ],
    ),
  );

  Widget _buildFullScreenView() => Stack(
    children: [
      InteractiveViewer(
        minScale: 0.5,
        maxScale: 4.0,
        child: Container(
          width: double.infinity,
          height: double.infinity,
          color: Colors.black,
          child: PuzzleDetail(puzzleImage: widget.puzzleImage, pieces: _pieces),
        ),
      ),

      Positioned(
        bottom: 16,
        right: 16,
        child: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Text(
            'Tap grid icon to view pieces',
            style: TextStyle(color: Colors.white, fontSize: 12),
          ),
        ),
      ),
    ],
  );

  Widget _buildPuzzleView() => Column(
    children: [
      Container(
        height: 200,
        width: double.infinity,
        decoration: BoxDecoration(border: Border.all(color: Colors.grey[300]!)),
        child: PlatformImageWidget(
          image: widget.puzzleImage,
          fit: BoxFit.contain,
        ),
      ),

      Padding(
        padding: const EdgeInsets.all(8.0),
        child: Text(
          'Puzzle ID: ${_puzzle?.puzzleId ?? 'Unknown'}',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ),

      const Divider(),

      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Puzzle Pieces',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            Text(
              '${_pieces.length} pieces',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),

      Expanded(
        child:
            _pieces.isEmpty
                ? Center(
                  child: Text(
                    'Add your first puzzle piece',
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                )
                : GridView.builder(
                  padding: const EdgeInsets.all(16.0),
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 16.0,
                    crossAxisSpacing: 16.0,
                  ),
                  itemCount: _pieces.length,
                  itemBuilder:
                      (context, index) => _buildPieceItem(_pieces[index]),
                ),
      ),
    ],
  );

  Widget _buildPieceItem(Piece piece) => Card(
    elevation: 2,
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child:
              piece.image != null
                  ? PlatformImageWidget(
                    image: piece.image!,
                    width: double.infinity,
                  )
                  : Container(
                    color: Colors.grey[300],
                    child: const Center(child: Icon(Icons.image_not_supported)),
                  ),
        ),

        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Pos: (${piece.position.x.toStringAsFixed(1)}, ${piece.position.y.toStringAsFixed(1)})',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              Text(
                'Confidence: ${(piece.confidence * 100).toStringAsFixed(1)}%',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              Text(
                'Rotation: ${piece.rotation}°',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ],
    ),
  );
}
