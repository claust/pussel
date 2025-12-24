import { describe, it, expect } from 'vitest';
import { TEST_PUZZLES, getTestPuzzleById } from './test-puzzles';

describe('TEST_PUZZLES', () => {
  it('should have 10 test puzzles', () => {
    expect(TEST_PUZZLES).toHaveLength(10);
  });

  it('should have correct structure for each puzzle', () => {
    TEST_PUZZLES.forEach((puzzle) => {
      expect(puzzle).toHaveProperty('id');
      expect(puzzle).toHaveProperty('name');
      expect(puzzle).toHaveProperty('path');
      expect(puzzle.path).toMatch(/^\/test-puzzles\/puzzle_\d{3}\.jpg$/);
    });
  });

  it('should have unique ids', () => {
    const ids = TEST_PUZZLES.map((p) => p.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });
});

describe('getTestPuzzleById', () => {
  it('should return puzzle by id', () => {
    const puzzle = getTestPuzzleById('001');
    expect(puzzle).toBeDefined();
    expect(puzzle?.id).toBe('001');
    expect(puzzle?.name).toBe('Puzzle 1');
  });

  it('should return undefined for non-existent id', () => {
    const puzzle = getTestPuzzleById('999');
    expect(puzzle).toBeUndefined();
  });
});
