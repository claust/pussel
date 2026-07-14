import type { PieceRegion } from '@/types';

// Regions below this backend confidence are treated as "no piece" (e.g. a face
// or a table object that slipped past the detector's hard gates). Shared by
// every consumer of the piece-preview endpoint so the live overlay and the
// auto-capture pipeline agree on what counts as a detection.
export const PIECE_CONFIDENCE_THRESHOLD = 0.5;

/**
 * Whether a detected region should be treated as an actual puzzle piece.
 *
 * @param region - The piece-preview result, or null when none is available.
 * @returns True when a region was found with confidence at or above the
 *   shared threshold.
 */
export function isConfidentPiece(region: PieceRegion | null | undefined): boolean {
  return Boolean(region?.found) && (region?.confidence ?? 0) >= PIECE_CONFIDENCE_THRESHOLD;
}
