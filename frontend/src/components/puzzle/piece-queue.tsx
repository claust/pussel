'use client';

import { AlertCircle, Check, Loader2, RotateCcw, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { CaptureEntry } from '@/stores/capture-queue-store';
import { getDisplayPosition } from '@/types';
import { cn } from '@/lib/utils';

interface PieceQueueProps {
  entries: CaptureEntry[];
  onRetry: (id: string) => void;
  onDelete: (id: string) => void;
  className?: string;
}

/**
 * Horizontal strip of captured pieces flowing through the prediction
 * pipeline: queued → predicting → done (or error with retry). New captures
 * pop in so an auto-commit is clearly visible.
 */
export function PieceQueue({ entries, onRetry, onDelete, className }: PieceQueueProps) {
  if (entries.length === 0) {
    return (
      <div
        className={cn(
          'text-muted-foreground rounded-lg border border-dashed p-4 text-center text-sm',
          className
        )}
        data-testid="piece-queue-empty"
      >
        Captured pieces appear here — hold a piece up to the camera.
      </div>
    );
  }

  return (
    <div className={cn(className)} data-testid="piece-queue">
      <div className="flex flex-wrap gap-3 pb-2">
        {entries.map((entry, index) => {
          const imageSrc = entry.piece?.imageData ?? entry.imageUrl;
          return (
            <div
              key={entry.id}
              className={cn(
                'animate-queue-pop group focus-visible:ring-ring relative w-24 shrink-0 rounded-lg border-2 focus:outline-none focus-visible:ring-2',
                entry.status === 'done' && 'border-green-500',
                entry.status === 'error' && 'border-destructive',
                (entry.status === 'queued' || entry.status === 'predicting') && 'border-border'
              )}
              // Focusable (but not an activatable control — the preview is revealed purely via
              // CSS :focus-within) so the enlarged view is reachable by keyboard (tab to the
              // tile) and by touch (tap the tile) without triggering the retry/delete buttons.
              tabIndex={0}
              aria-label={`Captured piece ${index + 1} — focus to enlarge`}
              data-testid="queue-entry"
              data-status={entry.status}
            >
              {/* Enlarged preview on hover, keyboard focus, or touch tap — floats above the
                  tile and ignores pointer events so it never blocks the delete/retry controls.
                  group-focus-within covers the focusable tile and the buttons inside it. */}
              <div
                className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 hidden -translate-x-1/2 group-focus-within:block group-hover:block"
                data-testid="queue-entry-preview"
              >
                <div className="bg-background border-border overflow-hidden rounded-lg border-2 shadow-xl">
                  <img
                    src={imageSrc}
                    alt={`Captured piece ${index + 1} enlarged`}
                    className="bg-muted h-64 w-64 max-w-[75vw] object-contain"
                  />
                </div>
              </div>

              <div className="overflow-hidden rounded-[inherit]">
                <img
                  src={imageSrc}
                  alt={`Captured piece ${index + 1}`}
                  className="bg-muted aspect-square w-full object-cover"
                />
              </div>

              {/* Status overlay — bottom radius matches the tile's rounded-lg so its corners
                  don't protrude past the rounded border (the outer tile has no overflow clip). */}
              <div className="bg-background/85 absolute inset-x-0 bottom-0 flex h-6 items-center justify-center gap-1 rounded-b-lg text-[10px] font-medium">
                {entry.status === 'queued' && <span className="text-muted-foreground">Queued</span>}
                {entry.status === 'predicting' && (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    <span>Predicting…</span>
                  </>
                )}
                {entry.status === 'done' && entry.piece && (
                  <>
                    <Check className="h-3 w-3 text-green-600" />
                    <span>
                      ({(getDisplayPosition(entry.piece).x * 100).toFixed(0)}%,{' '}
                      {(getDisplayPosition(entry.piece).y * 100).toFixed(0)}%)
                    </span>
                  </>
                )}
                {entry.status === 'error' && (
                  <>
                    <AlertCircle className="text-destructive h-3 w-3" />
                    <span className="text-destructive">Failed</span>
                  </>
                )}
              </div>

              {/* Retry failed predictions + delete, laid out side by side so they don't overlap */}
              <div className="absolute top-1 right-1 flex gap-1">
                {entry.status === 'error' && (
                  <Button
                    size="icon"
                    variant="secondary"
                    className="h-6 w-6"
                    onClick={() => onRetry(entry.id)}
                    title="Retry prediction"
                  >
                    <RotateCcw className="h-3 w-3" />
                    <span className="sr-only">Retry prediction</span>
                  </Button>
                )}
                <Button
                  size="icon"
                  variant="secondary"
                  className="h-6 w-6"
                  onClick={() => onDelete(entry.id)}
                  title="Remove piece"
                  data-testid={`queue-entry-delete-${index}`}
                >
                  <X className="h-3 w-3" />
                  <span className="sr-only">Remove piece</span>
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
