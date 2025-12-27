import type { Puzzle, Piece } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      method: 'GET',
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function uploadPuzzle(imageBlob: Blob): Promise<Puzzle> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'puzzle.jpg');

  const res = await fetch(`${API_BASE}/api/v1/puzzle/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new ApiError('Failed to upload puzzle', res.status);
  }

  return res.json();
}

export async function processPiece(puzzleId: string, pieceBlob: Blob): Promise<Piece> {
  const formData = new FormData();
  formData.append('file', pieceBlob, 'piece.jpg');

  const res = await fetch(`${API_BASE}/api/v1/puzzle/${puzzleId}/piece`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new ApiError('Failed to process piece', res.status);
  }

  return res.json();
}

export { API_BASE };
