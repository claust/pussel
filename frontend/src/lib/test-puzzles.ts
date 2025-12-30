import type { TestPuzzle } from '@/types';

export const TEST_PUZZLES: TestPuzzle[] = [
  { id: '001', name: 'Puzzle 1', path: '/test-puzzles/puzzle_001.jpg' },
  { id: '002', name: 'Puzzle 2', path: '/test-puzzles/puzzle_002.jpg' },
  { id: '003', name: 'Puzzle 3', path: '/test-puzzles/puzzle_003.jpg' },
  { id: '004', name: 'Puzzle 4', path: '/test-puzzles/puzzle_004.jpg' },
  { id: '005', name: 'Puzzle 5', path: '/test-puzzles/puzzle_005.jpg' },
  { id: '006', name: 'Puzzle 6', path: '/test-puzzles/puzzle_006.jpg' },
  { id: '007', name: 'Puzzle 7', path: '/test-puzzles/puzzle_007.jpg' },
  { id: '008', name: 'Puzzle 8', path: '/test-puzzles/puzzle_008.jpg' },
  { id: '009', name: 'Puzzle 9', path: '/test-puzzles/puzzle_009.jpg' },
  { id: '010', name: 'Puzzle 10', path: '/test-puzzles/puzzle_010.jpg' },
];

export function getTestPuzzleById(id: string): TestPuzzle | undefined {
  return TEST_PUZZLES.find((puzzle) => puzzle.id === id);
}
