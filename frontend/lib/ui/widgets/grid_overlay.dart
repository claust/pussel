import 'package:flutter/material.dart';

import '../../models/grid_size.dart';

/// A widget that displays a tappable grid overlay on a puzzle image.
class GridOverlay extends StatelessWidget {
  const GridOverlay({
    required this.gridSize,
    required this.onCellTap,
    this.selectedCell,
    super.key,
  });

  /// The grid size to display.
  final GridSize gridSize;

  /// Callback when a cell is tapped.
  final void Function(int cellIndex) onCellTap;

  /// Currently selected cell index (for highlighting).
  final int? selectedCell;

  @override
  Widget build(BuildContext context) => GridView.builder(
    physics: const NeverScrollableScrollPhysics(),
    gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
      crossAxisCount: gridSize.dimension,
    ),
    itemCount: gridSize.totalCells,
    itemBuilder:
        (context, index) => GestureDetector(
          onTap: () => onCellTap(index),
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(
                color: selectedCell == index ? Colors.blue : Colors.white,
                width: selectedCell == index ? 3 : 2,
              ),
              color:
                  selectedCell == index
                      ? Colors.blue.withValues(alpha: 0.3)
                      : Colors.transparent,
            ),
            child: Center(
              child: Text(
                '${index + 1}',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  shadows: [
                    Shadow(
                      blurRadius: 4,
                      color: Colors.black.withValues(alpha: 0.8),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
  );
}
