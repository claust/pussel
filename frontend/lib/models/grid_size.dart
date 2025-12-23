/// Enum representing supported grid sizes for test mode.
enum GridSize {
  twoByTwo(2, 4),
  threeByThree(3, 9);

  const GridSize(this.dimension, this.totalCells);

  /// The dimension of the grid (2 for 2x2, 3 for 3x3).
  final int dimension;

  /// Total number of cells in the grid.
  final int totalCells;

  /// Display name for the grid size.
  String get displayName => '${dimension}x$dimension';
}
