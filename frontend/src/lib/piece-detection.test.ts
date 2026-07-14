import { describe, it, expect } from 'vitest';
import { isConfidentPiece, PIECE_CONFIDENCE_THRESHOLD } from './piece-detection';
import type { PieceRegion } from '@/types';

function region(overrides: Partial<PieceRegion>): PieceRegion {
  return { found: true, polygon: [], confidence: 1, ...overrides };
}

describe('isConfidentPiece', () => {
  it('is false for null or undefined', () => {
    expect(isConfidentPiece(null)).toBe(false);
    expect(isConfidentPiece(undefined)).toBe(false);
  });

  it('is false when nothing was found, even at high confidence', () => {
    expect(isConfidentPiece(region({ found: false, confidence: 1 }))).toBe(false);
  });

  it('is false when confidence is below the threshold', () => {
    expect(isConfidentPiece(region({ confidence: PIECE_CONFIDENCE_THRESHOLD - 0.01 }))).toBe(false);
  });

  it('is true at or above the threshold', () => {
    expect(isConfidentPiece(region({ confidence: PIECE_CONFIDENCE_THRESHOLD }))).toBe(true);
    expect(isConfidentPiece(region({ confidence: 1 }))).toBe(true);
  });
});
