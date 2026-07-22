import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PieceCard } from './piece-card';
import type { Piece } from '@/types';

function makePiece(overrides: Partial<Piece> = {}): Piece {
  return {
    position: { x: 0.42, y: 0.13, normalized: true },
    positionConfidence: 0.9,
    rotation: 90,
    rotationConfidence: 0.8,
    ...overrides,
  };
}

describe('PieceCard', () => {
  it('shows the snapped position when the grid is known', () => {
    render(
      <PieceCard
        piece={makePiece({ snappedPosition: { x: 0.5, y: 0.25, normalized: true } })}
        index={0}
      />
    );

    expect(screen.getByText('(50%, 25%)')).toBeInTheDocument();
  });

  it('shows the raw predicted position when the grid is unknown', () => {
    render(<PieceCard piece={makePiece({ snappedPosition: null })} index={0} />);

    expect(screen.getByText('(42%, 13%)')).toBeInTheDocument();
  });
});
