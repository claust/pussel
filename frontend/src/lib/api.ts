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

interface PuzzleApiResponse {
  puzzle_id: string;
  image_url?: string;
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

  const data: PuzzleApiResponse = await res.json();
  return {
    puzzleId: data.puzzle_id,
    imageUrl: data.image_url,
  };
}

interface PieceApiResponse {
  position: { x: number; y: number };
  position_confidence: number;
  rotation: 0 | 90 | 180 | 270;
  rotation_confidence: number;
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

  const data: PieceApiResponse = await res.json();
  return {
    position: { ...data.position, normalized: true },
    positionConfidence: data.position_confidence,
    rotation: data.rotation,
    rotationConfidence: data.rotation_confidence,
  };
}

export { API_BASE };
