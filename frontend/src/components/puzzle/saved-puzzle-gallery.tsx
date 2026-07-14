'use client';

import { Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { SavedPuzzleMeta } from '@/lib/puzzle-library';

interface SavedPuzzleGalleryProps {
  puzzles: SavedPuzzleMeta[];
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  disabled?: boolean;
}

/**
 * Grid of previously captured real puzzles. Selecting a card reuses that
 * puzzle's stored image instead of photographing a new one.
 */
export function SavedPuzzleGallery({
  puzzles,
  onSelect,
  onDelete,
  disabled = false,
}: SavedPuzzleGalleryProps) {
  if (puzzles.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {puzzles.map((puzzle) => (
        <div key={puzzle.id} className="group relative">
          <button
            type="button"
            onClick={() => onSelect(puzzle.id)}
            disabled={disabled}
            className="hover:border-primary focus-visible:ring-ring block w-full overflow-hidden rounded-lg border transition-shadow hover:shadow-md focus-visible:ring-2 focus-visible:outline-none disabled:opacity-50"
          >
            <img
              src={puzzle.thumbnail}
              alt={puzzle.name}
              className="aspect-square w-full object-cover"
            />
            <span className="block truncate px-2 py-1.5 text-left text-sm font-medium">
              {puzzle.name}
            </span>
          </button>
          <Button
            variant="secondary"
            size="icon"
            onClick={() => onDelete(puzzle.id)}
            disabled={disabled}
            aria-label={`Delete ${puzzle.name}`}
            // Visible by default so touch devices (no hover) can delete;
            // fades to hover-reveal only on larger, pointer-capable screens.
            className="absolute top-1.5 right-1.5 h-7 w-7 opacity-100 transition-opacity focus-visible:opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
    </div>
  );
}
