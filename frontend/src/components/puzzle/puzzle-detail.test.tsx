import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PuzzleDetail } from './puzzle-detail';
import type { Piece } from '@/types';

function makePiece(overrides: Partial<Piece> = {}): Piece {
  return {
    position: { x: 0.42, y: 0.13, normalized: true },
    positionConfidence: 0.9,
    rotation: 0,
    rotationConfidence: 0.8,
    imageData: 'data:image/png;base64,test',
    ...overrides,
  };
}

describe('PuzzleDetail', () => {
  it('places a piece at its snapped position when the grid is known', () => {
    render(
      <PuzzleDetail
        puzzleImage="data:image/jpeg;base64,puzzle"
        pieces={[
          makePiece({
            gridRow: 0,
            gridCol: 1,
            snappedPosition: { x: 0.5, y: 0.25, normalized: true },
          }),
        ]}
      />
    );

    const piece = screen.getByAltText('Piece 1').parentElement!;
    expect(piece).toHaveStyle({ left: '50%', top: '25%' });
  });

  it('falls back to the raw predicted position when the grid is unknown', () => {
    render(
      <PuzzleDetail
        puzzleImage="data:image/jpeg;base64,puzzle"
        pieces={[makePiece({ gridRow: null, gridCol: null, snappedPosition: null })]}
      />
    );

    const piece = screen.getByAltText('Piece 1').parentElement!;
    expect(piece).toHaveStyle({ left: '42%', top: '13%' });
  });
});
