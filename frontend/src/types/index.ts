export interface Position {
  x: number; // 0-1 normalized
  y: number; // 0-1 normalized
  normalized: boolean;
}

export interface Piece {
  position: Position;
  positionConfidence: number; // 0-1
  rotation: 0 | 90 | 180 | 270;
  rotationConfidence: number; // 0-1
  imageData?: string; // base64 or blob URL
}

export interface Puzzle {
  puzzleId: string;
  imageUrl?: string;
}

export type GridSize = '2x2' | '3x3' | '4x4' | '5x5' | '6x6';

export interface TestPuzzle {
  id: string;
  name: string;
  path: string;
}

export const GRID_DIMENSIONS: Record<GridSize, { dimension: number; totalCells: number }> = {
  '2x2': { dimension: 2, totalCells: 4 },
  '3x3': { dimension: 3, totalCells: 9 },
  '4x4': { dimension: 4, totalCells: 16 },
  '5x5': { dimension: 5, totalCells: 25 },
  '6x6': { dimension: 6, totalCells: 36 },
};

export type CameraMode = 'puzzle' | 'piece';

export type PieceSelectionMode = 'grid' | 'realistic';

export interface GeneratedPiece {
  imageData: string;
  centerX: number;
  centerY: number;
  config: Record<string, unknown>;
}

// Game mode types for drag-and-drop puzzle solving

export interface GamePiece {
  id: string;
  row: number;
  col: number;
  imageData: string;
  correctX: number; // Normalized target position (0-1)
  correctY: number;
  width: number;
  height: number;
  currentX: number; // Current dragged position (0-1)
  currentY: number;
  isPlaced: boolean; // Whether piece is snapped to correct position
  zIndex: number; // For stacking order when dragging
}

export interface CutPuzzleResponse {
  pieces: Array<{
    id: string;
    row: number;
    col: number;
    image: string;
    correct_x: number;
    correct_y: number;
    width: number;
    height: number;
  }>;
  grid: { rows: number; cols: number };
  puzzle_width: number;
  puzzle_height: number;
}
