import { describe, it, expect, vi, afterEach } from 'vitest';
import { API_BASE, processPiece } from './api';

describe('API Client', () => {
  it('should have correct API base URL', () => {
    expect(API_BASE).toBe('http://localhost:8000');
  });
});

describe('processPiece', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mockPieceResponse(body: Record<string, unknown>) {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(body),
      })
    );
  }

  it('maps grid snap fields from the API response', async () => {
    mockPieceResponse({
      position: { x: 0.42, y: 0.13 },
      position_confidence: 0.9,
      rotation: 90,
      rotation_confidence: 0.8,
      grid_row: 0,
      grid_col: 1,
      snapped_position: { x: 0.5, y: 0.16667 },
    });

    const piece = await processPiece('puzzle-1', new Blob());

    expect(piece.position).toEqual({ x: 0.42, y: 0.13, normalized: true });
    expect(piece.gridRow).toBe(0);
    expect(piece.gridCol).toBe(1);
    expect(piece.snappedPosition).toEqual({ x: 0.5, y: 0.16667, normalized: true });
  });

  it('returns null grid snap fields when the puzzle grid is unknown', async () => {
    mockPieceResponse({
      position: { x: 0.42, y: 0.13 },
      position_confidence: 0.9,
      rotation: 0,
      rotation_confidence: 0.8,
      grid_row: null,
      grid_col: null,
      snapped_position: null,
    });

    const piece = await processPiece('puzzle-1', new Blob());

    expect(piece.gridRow).toBeNull();
    expect(piece.gridCol).toBeNull();
    expect(piece.snappedPosition).toBeNull();
  });

  it('tolerates responses without the grid snap fields', async () => {
    mockPieceResponse({
      position: { x: 0.42, y: 0.13 },
      position_confidence: 0.9,
      rotation: 0,
      rotation_confidence: 0.8,
    });

    const piece = await processPiece('puzzle-1', new Blob());

    expect(piece.gridRow).toBeNull();
    expect(piece.gridCol).toBeNull();
    expect(piece.snappedPosition).toBeNull();
  });
});
