import type { Puzzle, Piece, GeneratedPiece } from '@/types';

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

interface GeneratePieceApiResponse {
  piece_image: string;
  piece_config: Record<string, unknown>;
}

export async function generateRealisticPiece(
  puzzleId: string,
  centerX: number,
  centerY: number,
  pieceSizeRatio: number = 0.25
): Promise<GeneratedPiece> {
  const res = await fetch(`${API_BASE}/api/v1/puzzle/${puzzleId}/generate-piece`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      center_x: centerX,
      center_y: centerY,
      piece_size_ratio: pieceSizeRatio,
    }),
  });

  if (!res.ok) {
    throw new ApiError('Failed to generate piece', res.status);
  }

  const data: GeneratePieceApiResponse = await res.json();
  return {
    imageData: data.piece_image,
    centerX,
    centerY,
    config: data.piece_config,
  };
}

export { API_BASE };
