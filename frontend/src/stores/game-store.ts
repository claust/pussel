import { create } from 'zustand';
import type { GamePiece, GridSize, CutPuzzleResponse } from '@/types';

interface GameState {
  puzzleId: string | null;
  puzzleImage: string | null; // blob URL or base64
  puzzleWidth: number;
  puzzleHeight: number;
  pieces: GamePiece[];
  gridSize: GridSize;
  isLoading: boolean;
  isComplete: boolean;
  error: string | null;

  // Actions
  startGame: (puzzleId: string, puzzleImage: string, response: CutPuzzleResponse) => void;
  movePiece: (pieceId: string, x: number, y: number) => void;
  placePiece: (pieceId: string) => void;
  bringToFront: (pieceId: string) => void;
  shufflePieces: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const SNAP_THRESHOLD = 0.1; // 10% of puzzle dimension

function shufflePositions(pieces: GamePiece[]): GamePiece[] {
  // Scatter pieces randomly within the puzzle bounds
  // Keep pieces within 0.1 to 0.9 to avoid edge clipping
  // Start zIndex at 1 so unplaced pieces are always above placed pieces (which have zIndex 0)
  return pieces.map((piece, index) => ({
    ...piece,
    currentX: 0.1 + Math.random() * 0.8,
    currentY: 0.1 + Math.random() * 0.8,
    isPlaced: false,
    zIndex: index + 1,
  }));
}

function convertResponseToPieces(response: CutPuzzleResponse): GamePiece[] {
  return response.pieces.map((piece, index) => ({
    id: piece.id,
    row: piece.row,
    col: piece.col,
    imageData: piece.image,
    correctX: piece.correct_x,
    correctY: piece.correct_y,
    width: piece.width,
    height: piece.height,
    currentX: piece.correct_x, // Start at correct position, will be shuffled
    currentY: piece.correct_y,
    isPlaced: false,
    zIndex: index,
  }));
}

function checkCompletion(pieces: GamePiece[]): boolean {
  return pieces.every((piece) => piece.isPlaced);
}

export const useGameStore = create<GameState>((set, get) => ({
  puzzleId: null,
  puzzleImage: null,
  puzzleWidth: 0,
  puzzleHeight: 0,
  pieces: [],
  gridSize: '3x3',
  isLoading: false,
  isComplete: false,
  error: null,

  startGame: (puzzleId, puzzleImage, response) => {
    const pieces = convertResponseToPieces(response);
    const shuffledPieces = shufflePositions(pieces);
    const gridSize = `${response.grid.rows}x${response.grid.cols}` as GridSize;

    set({
      puzzleId,
      puzzleImage,
      puzzleWidth: response.puzzle_width,
      puzzleHeight: response.puzzle_height,
      pieces: shuffledPieces,
      gridSize,
      isComplete: false,
      error: null,
    });
  },

  movePiece: (pieceId, x, y) => {
    set((state) => ({
      pieces: state.pieces.map((piece) =>
        piece.id === pieceId && !piece.isPlaced ? { ...piece, currentX: x, currentY: y } : piece
      ),
    }));
  },

  placePiece: (pieceId) => {
    const state = get();
    const piece = state.pieces.find((p) => p.id === pieceId);
    if (!piece || piece.isPlaced) return;

    // Check if piece is close enough to correct position
    const dx = Math.abs(piece.currentX - piece.correctX);
    const dy = Math.abs(piece.currentY - piece.correctY);

    if (dx < SNAP_THRESHOLD && dy < SNAP_THRESHOLD) {
      // Snap to correct position and send to back (z-index 0) so it doesn't cover unplaced pieces
      set((state) => {
        const updatedPieces = state.pieces.map((p) =>
          p.id === pieceId
            ? { ...p, currentX: p.correctX, currentY: p.correctY, isPlaced: true, zIndex: 0 }
            : p
        );
        return {
          pieces: updatedPieces,
          isComplete: checkCompletion(updatedPieces),
        };
      });
    }
  },

  bringToFront: (pieceId) => {
    set((state) => {
      const maxZ = Math.max(...state.pieces.map((p) => p.zIndex));
      return {
        pieces: state.pieces.map((piece) =>
          piece.id === pieceId ? { ...piece, zIndex: maxZ + 1 } : piece
        ),
      };
    });
  },

  shufflePieces: () => {
    set((state) => ({
      pieces: shufflePositions(state.pieces.map((p) => ({ ...p, isPlaced: false }))),
      isComplete: false,
    }));
  },

  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),

  reset: () =>
    set({
      puzzleId: null,
      puzzleImage: null,
      puzzleWidth: 0,
      puzzleHeight: 0,
      pieces: [],
      isComplete: false,
      error: null,
    }),
}));
