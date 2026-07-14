import { create } from 'zustand';
import type { Puzzle, Piece, GridSize } from '@/types';

interface PuzzleState {
  puzzle: Puzzle | null;
  puzzleImage: string | null; // blob URL or base64
  pieces: Piece[];
  gridSize: GridSize;
  isLoading: boolean;
  error: string | null;

  setPuzzle: (puzzle: Puzzle, imageUrl: string) => void;
  addPiece: (piece: Piece) => void;
  removePiece: (id: string) => void;
  setGridSize: (gridSize: GridSize) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const usePuzzleStore = create<PuzzleState>((set) => ({
  puzzle: null,
  puzzleImage: null,
  pieces: [],
  gridSize: '3x3',
  isLoading: false,
  error: null,

  setPuzzle: (puzzle, imageUrl) => set({ puzzle, puzzleImage: imageUrl, pieces: [], error: null }),
  addPiece: (piece) => set((state) => ({ pieces: [...state.pieces, piece] })),
  // Remove by stable id (pieces from the capture queue carry the entry id),
  // rather than object identity which would silently no-op on any clone.
  removePiece: (id) => set((state) => ({ pieces: state.pieces.filter((p) => p.id !== id) })),
  setGridSize: (gridSize) => set({ gridSize }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () => set({ puzzle: null, puzzleImage: null, pieces: [], error: null }),
}));
