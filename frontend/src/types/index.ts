export interface User {
  id: string;
  email: string;
  name: string;
  picture?: string | null;
  createdAt?: string;
}

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

export type GridSize = '2x2' | '3x3';

export interface TestPuzzle {
  id: string;
  name: string;
  path: string;
}

export const GRID_DIMENSIONS: Record<GridSize, { dimension: number; totalCells: number }> = {
  '2x2': { dimension: 2, totalCells: 4 },
  '3x3': { dimension: 3, totalCells: 9 },
};

export type CameraMode = 'puzzle' | 'piece';

export type PieceSelectionMode = 'grid' | 'realistic';

export interface GeneratedPiece {
  imageData: string;
  centerX: number;
  centerY: number;
  config: Record<string, unknown>;
}
