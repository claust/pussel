import { describe, it, expect, beforeEach } from 'vitest';
import { usePuzzleStore } from './puzzle-store';

describe('PuzzleStore', () => {
  beforeEach(() => {
    // Reset store before each test
    usePuzzleStore.getState().reset();
  });

  it('should have initial state', () => {
    const state = usePuzzleStore.getState();
    expect(state.puzzle).toBeNull();
    expect(state.puzzleImage).toBeNull();
    expect(state.pieces).toEqual([]);
    expect(state.gridSize).toBe('3x3');
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
  });

  it('should set puzzle and image', () => {
    const puzzle = { puzzleId: 'test-123' };
    const imageUrl = 'data:image/jpeg;base64,...';

    usePuzzleStore.getState().setPuzzle(puzzle, imageUrl);

    const state = usePuzzleStore.getState();
    expect(state.puzzle).toEqual(puzzle);
    expect(state.puzzleImage).toBe(imageUrl);
    expect(state.pieces).toEqual([]);
    expect(state.error).toBeNull();
  });

  it('should add pieces', () => {
    const piece1 = {
      position: { x: 0.5, y: 0.5, normalized: true },
      confidence: 0.9,
      rotation: 0 as const,
    };
    const piece2 = {
      position: { x: 0.3, y: 0.7, normalized: true },
      confidence: 0.85,
      rotation: 90 as const,
    };

    usePuzzleStore.getState().addPiece(piece1);
    expect(usePuzzleStore.getState().pieces).toHaveLength(1);

    usePuzzleStore.getState().addPiece(piece2);
    expect(usePuzzleStore.getState().pieces).toHaveLength(2);
    expect(usePuzzleStore.getState().pieces[1]).toEqual(piece2);
  });

  it('should set grid size', () => {
    usePuzzleStore.getState().setGridSize('2x2');
    expect(usePuzzleStore.getState().gridSize).toBe('2x2');

    usePuzzleStore.getState().setGridSize('3x3');
    expect(usePuzzleStore.getState().gridSize).toBe('3x3');
  });

  it('should set loading state', () => {
    usePuzzleStore.getState().setLoading(true);
    expect(usePuzzleStore.getState().isLoading).toBe(true);

    usePuzzleStore.getState().setLoading(false);
    expect(usePuzzleStore.getState().isLoading).toBe(false);
  });

  it('should set error', () => {
    usePuzzleStore.getState().setError('Something went wrong');
    expect(usePuzzleStore.getState().error).toBe('Something went wrong');

    usePuzzleStore.getState().setError(null);
    expect(usePuzzleStore.getState().error).toBeNull();
  });

  it('should reset state', () => {
    // Set some state
    usePuzzleStore.getState().setPuzzle({ puzzleId: 'test' }, 'image-url');
    usePuzzleStore.getState().addPiece({
      position: { x: 0.5, y: 0.5, normalized: true },
      confidence: 0.9,
      rotation: 0,
    });
    usePuzzleStore.getState().setError('Error');

    // Reset
    usePuzzleStore.getState().reset();

    const state = usePuzzleStore.getState();
    expect(state.puzzle).toBeNull();
    expect(state.puzzleImage).toBeNull();
    expect(state.pieces).toEqual([]);
    expect(state.error).toBeNull();
  });
});
