import 'package:flutter/material.dart';

import '../../../models/grid_size.dart';
import 'puzzle_selection_screen.dart';

/// Screen for selecting the grid size (2x2 or 3x3).
class GridSelectionScreen extends StatelessWidget {
  const GridSelectionScreen({super.key});

  void _selectGridSize(BuildContext context, GridSize gridSize) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PuzzleSelectionScreen(gridSize: gridSize),
      ),
    );
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Test Mode')),
    body: SafeArea(
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Icon
              Icon(
                Icons.grid_4x4,
                size: 64,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 24),

              // Title
              Text(
                'Select Grid Size',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 8),

              // Subtitle
              Text(
                'Choose how many pieces the puzzle will be divided into',
                style: Theme.of(context).textTheme.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),

              // 2x2 Button
              _GridSizeButton(
                gridSize: GridSize.twoByTwo,
                icon: Icons.grid_view,
                description: '4 pieces',
                onTap: () => _selectGridSize(context, GridSize.twoByTwo),
              ),
              const SizedBox(height: 16),

              // 3x3 Button
              _GridSizeButton(
                gridSize: GridSize.threeByThree,
                icon: Icons.grid_on,
                description: '9 pieces',
                onTap: () => _selectGridSize(context, GridSize.threeByThree),
              ),
            ],
          ),
        ),
      ),
    ),
  );
}

class _GridSizeButton extends StatelessWidget {
  const _GridSizeButton({
    required this.gridSize,
    required this.icon,
    required this.description,
    required this.onTap,
  });

  final GridSize gridSize;
  final IconData icon;
  final String description;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => SizedBox(
    width: double.infinity,
    child: OutlinedButton(
      onPressed: onTap,
      style: OutlinedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 32),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                gridSize.displayName,
                style: Theme.of(context).textTheme.titleLarge,
              ),
              Text(description, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ],
      ),
    ),
  );
}
