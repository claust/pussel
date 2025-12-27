import { describe, it, expect } from 'vitest';
import { GRID_DIMENSIONS } from '@/types';

describe('GRID_DIMENSIONS', () => {
  it('should have correct values for 2x2 grid', () => {
    expect(GRID_DIMENSIONS['2x2']).toEqual({
      dimension: 2,
      totalCells: 4,
    });
  });

  it('should have correct values for 3x3 grid', () => {
    expect(GRID_DIMENSIONS['3x3']).toEqual({
      dimension: 3,
      totalCells: 9,
    });
  });
});

// Note: Canvas-based functions like cropCell require browser APIs
// and would need to be tested in an integration/e2e test environment
