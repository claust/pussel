'use client';

import { AlertCircle, Check, Loader2, RotateCcw, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { CaptureEntry } from '@/stores/capture-queue-store';
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
    <div className={cn('overflow-x-auto', className)} data-testid="piece-queue">
      <div className="flex gap-3 pb-2">
        {entries.map((entry, index) => (
          <div
            key={entry.id}
            className={cn(
              'animate-queue-pop relative w-24 shrink-0 overflow-hidden rounded-lg border-2',
              entry.status === 'done' && 'border-green-500',
              entry.status === 'error' && 'border-destructive',
              (entry.status === 'queued' || entry.status === 'predicting') && 'border-border'
            )}
            data-testid="queue-entry"
            data-status={entry.status}
          >
            <img
              src={entry.piece?.imageData ?? entry.imageUrl}
              alt={`Captured piece ${index + 1}`}
              className="bg-muted aspect-square w-full object-cover"
            />

            {/* Status overlay */}
            <div className="bg-background/85 absolute inset-x-0 bottom-0 flex h-6 items-center justify-center gap-1 text-[10px] font-medium">
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
                    ({(entry.piece.position.x * 100).toFixed(0)}%,{' '}
                    {(entry.piece.position.y * 100).toFixed(0)}%)
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
        ))}
      </div>
    </div>
  );
}
