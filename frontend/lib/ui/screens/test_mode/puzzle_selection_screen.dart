import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../../config/test_puzzles_config.dart';
import '../../../models/grid_size.dart';
import '../../../models/test_puzzle.dart';
import '../../../utils/platform_image.dart';
import 'test_puzzle_screen.dart';

/// Screen for selecting a test puzzle from bundled assets.
class PuzzleSelectionScreen extends StatelessWidget {
  const PuzzleSelectionScreen({required this.gridSize, super.key});

  final GridSize gridSize;

  Future<void> _selectPuzzle(BuildContext context, TestPuzzle puzzle) async {
    // Show loading indicator
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(child: CircularProgressIndicator()),
    );

    try {
      // Load asset bytes
      final byteData = await rootBundle.load(puzzle.assetPath);
      final bytes = byteData.buffer.asUint8List();
      final puzzleImage = PlatformImage.fromBytesOnly(bytes);

      if (!context.mounted) return;

      // Pop loading dialog
      Navigator.pop(context);

      // Navigate to test puzzle screen
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder:
              (context) => TestPuzzleScreen(
                puzzleImage: puzzleImage,
                gridSize: gridSize,
              ),
        ),
      );
    } catch (e) {
      if (!context.mounted) return;

      // Pop loading dialog
      Navigator.pop(context);

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error loading puzzle: $e')));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: Text('Select Puzzle (${gridSize.displayName})')),
    body: GridView.builder(
      padding: const EdgeInsets.all(16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        mainAxisSpacing: 16,
        crossAxisSpacing: 16,
      ),
      itemCount: testPuzzles.length,
      itemBuilder: (context, index) {
        final puzzle = testPuzzles[index];
        return _PuzzleTile(
          puzzle: puzzle,
          onTap: () => _selectPuzzle(context, puzzle),
        );
      },
    ),
  );
}

class _PuzzleTile extends StatelessWidget {
  const _PuzzleTile({required this.puzzle, required this.onTap});

  final TestPuzzle puzzle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Card(
    clipBehavior: Clip.antiAlias,
    child: InkWell(
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: Image.asset(
              puzzle.assetPath,
              fit: BoxFit.cover,
              errorBuilder:
                  (context, error, stackTrace) => Container(
                    color: Colors.grey[300],
                    child: const Center(
                      child: Icon(Icons.broken_image, size: 48),
                    ),
                  ),
            ),
          ),
          Container(
            padding: const EdgeInsets.all(8),
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            child: Text(
              puzzle.name,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
        ],
      ),
    ),
  );
}
