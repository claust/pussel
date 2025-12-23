import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../../../models/grid_size.dart';
import '../../../services/image_cropper_service.dart';
import '../../../utils/platform_image.dart';
import '../../widgets/grid_overlay.dart';
import '../../widgets/rotation_selector_dialog.dart';

/// Screen for selecting a piece from the puzzle grid.
class PieceSelectionScreen extends StatefulWidget {
  const PieceSelectionScreen({
    required this.puzzleImage,
    required this.gridSize,
    super.key,
  });

  final PlatformImage puzzleImage;
  final GridSize gridSize;

  @override
  State<PieceSelectionScreen> createState() => _PieceSelectionScreenState();
}

class _PieceSelectionScreenState extends State<PieceSelectionScreen> {
  final ImageCropperService _cropperService = ImageCropperService();

  int? _selectedCell;
  bool _isProcessing = false;

  Future<void> _onCellTap(int cellIndex) async {
    setState(() => _selectedCell = cellIndex);

    // Get preview of the selected cell
    Uint8List preview;
    try {
      preview = await _cropperService.getCellPreview(
        widget.puzzleImage,
        widget.gridSize,
        cellIndex,
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading preview: $e')));
      }
      return;
    }

    if (!mounted) return;

    // Show rotation selector dialog
    final rotation = await showDialog<int>(
      context: context,
      builder: (context) => RotationSelectorDialog(piecePreview: preview),
    );

    if (rotation == null || !mounted) {
      setState(() => _selectedCell = null);
      return;
    }

    // Crop and return the piece
    await _cropAndReturn(cellIndex, rotation);
  }

  Future<void> _cropAndReturn(int cellIndex, int rotation) async {
    setState(() => _isProcessing = true);

    try {
      final croppedPiece = await _cropperService.cropCell(
        widget.puzzleImage,
        widget.gridSize,
        cellIndex,
        rotation,
      );

      if (mounted) {
        Navigator.pop(context, croppedPiece);
      }
    } catch (e) {
      setState(() => _isProcessing = false);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error cropping piece: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: Text('Select Piece (${widget.gridSize.displayName})'),
    ),
    body:
        _isProcessing
            ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Processing piece...'),
                ],
              ),
            )
            : Column(
              children: [
                // Instructions
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  color: Colors.blue.shade50,
                  child: const Text(
                    'Tap a grid cell to select it as your piece',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 16),
                  ),
                ),

                // Puzzle with grid overlay
                Expanded(
                  child: Container(
                    margin: const EdgeInsets.all(16),
                    child: AspectRatio(
                      aspectRatio: 1,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          // Puzzle image
                          PlatformImageWidget(image: widget.puzzleImage),

                          // Grid overlay
                          GridOverlay(
                            gridSize: widget.gridSize,
                            selectedCell: _selectedCell,
                            onCellTap: _onCellTap,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
  );
}
