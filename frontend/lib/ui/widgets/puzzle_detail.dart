import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../../models/grid_size.dart';
import '../../models/piece.dart';
import '../../utils/platform_image.dart';

class PuzzleDetail extends StatelessWidget {
  const PuzzleDetail({
    required this.puzzleImage,
    required this.pieces,
    this.gridSize,
    this.onTap,
    super.key,
  });

  final PlatformImage puzzleImage;
  final List<Piece> pieces;
  final GridSize? gridSize;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: LayoutBuilder(
      builder:
          (context, constraints) => Stack(
            fit: StackFit.expand,
            children: [
              // Puzzle image background
              PlatformImageWidget(image: puzzleImage, fit: BoxFit.contain),

              // Dimming overlay for better piece visibility
              Container(color: Colors.black.withAlpha(77)),

              // Pieces overlay
              ...pieces.map((piece) => _buildPieceOverlay(piece, constraints)),
            ],
          ),
    ),
  );

  Widget _buildPieceOverlay(Piece piece, BoxConstraints constraints) {
    if (piece.image == null) return Container();

    // Calculate piece size based on grid dimension
    final dimension = gridSize?.dimension ?? 3;
    final pieceWidth = constraints.maxWidth / dimension;
    final pieceHeight = constraints.maxHeight / dimension;

    return Positioned(
      left: piece.position.x * constraints.maxWidth - (pieceWidth / 2),
      top: piece.position.y * constraints.maxHeight - (pieceHeight / 2),
      child: Transform.rotate(
        angle: piece.rotation * (math.pi / 180),
        child: Container(
          width: pieceWidth,
          height: pieceHeight,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.green, width: 2),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withAlpha(77),
                blurRadius: 5,
                spreadRadius: 1,
              ),
            ],
          ),
          child: PlatformImageWidget(image: piece.image!),
        ),
      ),
    );
  }
}
