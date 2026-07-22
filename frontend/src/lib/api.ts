import type {
  Puzzle,
  Piece,
  GeneratedPiece,
  CutPuzzleResponse,
  QuadCorners,
  DetectFrameResult,
  PieceRegion,
} from '@/types';
import { useAuthStore } from '@/stores/auth-store';

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

function getAuthHeaders(): Record<string, string> {
  const token = useAuthStore.getState().backendToken;
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
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
    headers: getAuthHeaders(),
    body: formData,
  });

  if (res.status === 401) {
    throw new ApiError('Authentication required. Please sign in.', res.status);
  }

  if (!res.ok) {
    throw new ApiError('Failed to upload puzzle', res.status);
  }

  const data: PuzzleApiResponse = await res.json();
  return {
    puzzleId: data.puzzle_id,
    imageUrl: data.image_url,
  };
}

interface CornerApi {
  x: number;
  y: number;
}

interface DetectFrameApiResponse {
  trimmed_image: string;
  corners: {
    top_left: CornerApi;
    top_right: CornerApi;
    bottom_right: CornerApi;
    bottom_left: CornerApi;
  };
  confidence: number;
}

export async function detectFrame(
  photoBlob: Blob,
  corners?: QuadCorners
): Promise<DetectFrameResult> {
  const formData = new FormData();
  formData.append('file', photoBlob, 'puzzle-photo.jpg');
  if (corners) {
    formData.append(
      'corners',
      JSON.stringify({
        top_left: corners.topLeft,
        top_right: corners.topRight,
        bottom_right: corners.bottomRight,
        bottom_left: corners.bottomLeft,
      })
    );
  }

  const res = await fetch(`${API_BASE}/api/v1/puzzle/detect-frame`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });

  if (res.status === 401) {
    throw new ApiError('Authentication required. Please sign in.', res.status);
  }

  if (!res.ok) {
    throw new ApiError('Failed to detect puzzle frame', res.status);
  }

  const data: DetectFrameApiResponse = await res.json();
  return {
    trimmedImageUrl: data.trimmed_image,
    corners: {
      topLeft: data.corners.top_left,
      topRight: data.corners.top_right,
      bottomRight: data.corners.bottom_right,
      bottomLeft: data.corners.bottom_left,
    },
    confidence: data.confidence,
  };
}

export async function detectPieceRegion(
  frameBlob: Blob,
  signal?: AbortSignal
): Promise<PieceRegion> {
  const formData = new FormData();
  formData.append('file', frameBlob, 'frame.jpg');

  const res = await fetch(`${API_BASE}/api/v1/piece/preview`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
    signal,
  });

  if (res.status === 401) {
    throw new ApiError('Authentication required. Please sign in.', res.status);
  }

  if (!res.ok) {
    throw new ApiError('Failed to detect piece region', res.status);
  }

  return res.json() as Promise<PieceRegion>;
}

interface PieceApiResponse {
  position: { x: number; y: number };
  position_confidence: number;
  rotation: 0 | 90 | 180 | 270;
  rotation_confidence: number;
  cleaned_image?: string; // Base64 PNG with background removed
  grid_row?: number | null; // 0-based nearest grid cell row; null when the grid is unknown
  grid_col?: number | null; // 0-based nearest grid cell column; null when the grid is unknown
  snapped_position?: { x: number; y: number } | null; // nearest cell center; null when the grid is unknown
}

export async function processPiece(
  puzzleId: string,
  pieceBlob: Blob,
  signal?: AbortSignal
): Promise<Piece> {
  const formData = new FormData();
  formData.append('file', pieceBlob, 'piece.jpg');

  const res = await fetch(`${API_BASE}/api/v1/puzzle/${puzzleId}/piece`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
    signal,
  });

  if (res.status === 401) {
    throw new ApiError('Authentication required. Please sign in.', res.status);
  }

  if (!res.ok) {
    throw new ApiError('Failed to process piece', res.status);
  }

  const data: PieceApiResponse = await res.json();
  return {
    position: { ...data.position, normalized: true },
    positionConfidence: data.position_confidence,
    rotation: data.rotation,
    rotationConfidence: data.rotation_confidence,
    imageData: data.cleaned_image, // Use cleaned image with background removed
    gridRow: data.grid_row ?? null,
    gridCol: data.grid_col ?? null,
    snappedPosition: data.snapped_position ? { ...data.snapped_position, normalized: true } : null,
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
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      center_x: centerX,
      center_y: centerY,
      piece_size_ratio: pieceSizeRatio,
    }),
  });

  if (res.status === 401) {
    throw new ApiError('Authentication required. Please sign in.', res.status);
  }

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

export async function cutPuzzle(
  puzzleId: string,
  rows: number,
  cols: number,
  seed?: number
): Promise<CutPuzzleResponse> {
  const res = await fetch(`${API_BASE}/api/v1/puzzle/${puzzleId}/cut-all`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      rows,
      cols,
      seed,
    }),
  });

  if (res.status === 401) {
    throw new ApiError('Authentication required. Please sign in.', res.status);
  }

  if (!res.ok) {
    throw new ApiError('Failed to cut puzzle', res.status);
  }

  return res.json();
}

export { API_BASE };
