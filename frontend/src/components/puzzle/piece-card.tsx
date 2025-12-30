'use client';

import type { Piece } from '@/types';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface PieceCardProps {
  piece: Piece;
  index: number;
  className?: string;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-600';
  if (confidence >= 0.5) return 'text-yellow-600';
  return 'text-red-600';
}

export function PieceCard({ piece, index, className }: PieceCardProps) {
  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardContent className="p-2">
        {/* Piece image */}
        <div className="bg-muted relative aspect-square overflow-hidden rounded">
          {piece.imageData ? (
            <img
              src={piece.imageData}
              alt={`Piece ${index + 1}`}
              className="h-full w-full object-cover"
              style={{ transform: `rotate(${piece.rotation}deg)` }}
            />
          ) : (
            <div className="text-muted-foreground flex h-full w-full items-center justify-center">
              No image
            </div>
          )}
        </div>

        {/* Piece info */}
        <div className="mt-2 space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Position:</span>
            <span>
              ({(piece.position.x * 100).toFixed(0)}%, {(piece.position.y * 100).toFixed(0)}%)
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Pos. conf:</span>
            <span className={getConfidenceColor(piece.positionConfidence)}>
              {(piece.positionConfidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Rotation:</span>
            <span>{piece.rotation}°</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Rot. conf:</span>
            <span className={getConfidenceColor(piece.rotationConfidence)}>
              {(piece.rotationConfidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
