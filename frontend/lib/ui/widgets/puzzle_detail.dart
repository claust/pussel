import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../../models/piece.dart';

class PuzzleDetail extends StatelessWidget {
  const PuzzleDetail({
    required this.puzzleImage,
    required this.pieces,
    this.onTap,
    super.key,
  });

  final File puzzleImage;
  final List<Piece> pieces;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Stack(
      fit: StackFit.expand,
      children: [
        // Puzzle image background
        Image.file(puzzleImage, fit: BoxFit.contain),

        // Pieces overlay
        ...pieces.map((piece) => _buildPieceOverlay(context, piece)).toList(),
      ],
    ),
  );

  Widget _buildPieceOverlay(BuildContext context, Piece piece) {
    if (piece.localImagePath == null) return Container();

    // Get size of the parent container
    final size = MediaQuery.of(context).size;

    // Piece dimensions (adjust as needed)
    const pieceSize = 100.0;

    return Positioned(
      left: piece.position.x * size.width - (pieceSize / 2),
      top: piece.position.y * size.height - (pieceSize / 2),
      child: Transform.rotate(
        angle: piece.rotation * (math.pi / 180),
        child: Container(
          width: pieceSize,
          height: pieceSize,
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
          child: Image.file(File(piece.localImagePath!), fit: BoxFit.cover),
        ),
      ),
    );
  }
}
