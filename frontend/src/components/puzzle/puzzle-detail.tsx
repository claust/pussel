'use client';

import { useMemo } from 'react';
import type { Piece, GridSize } from '@/types';
import { GRID_DIMENSIONS } from '@/types';
import { cn } from '@/lib/utils';

interface PuzzleDetailProps {
  puzzleImage: string;
  pieces: Piece[];
  gridSize?: GridSize;
  onClick?: () => void;
  className?: string;
}

export function PuzzleDetail({
  puzzleImage,
  pieces,
  gridSize = '3x3',
  onClick,
  className,
}: PuzzleDetailProps) {
  const { dimension } = GRID_DIMENSIONS[gridSize];
  const pieceSize = useMemo(() => 100 / dimension, [dimension]);

  return (
    <div
      className={cn(onClick && 'cursor-pointer', className)}
      onClick={onClick}
      {...(onClick && {
        role: 'button' as const,
        tabIndex: 0,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            onClick();
          }
        },
      })}
    >
      {/* Puzzle background with pieces overlay */}
      <div className="flex justify-center">
        <div className="relative overflow-hidden rounded-lg">
          <img
            src={puzzleImage}
            alt="Puzzle"
            className="block h-auto max-h-[80vh] w-auto max-w-full"
          />
          <div className="absolute inset-0 bg-black/30" />

          {/* Pieces overlay - positioned relative to the image container */}
          {pieces.map((piece, index) => (
            <div
              key={index}
              className="absolute border-2 border-green-500 shadow-lg"
              style={{
                left: `${piece.position.x * 100}%`,
                top: `${piece.position.y * 100}%`,
                width: `${pieceSize}%`,
                height: `${pieceSize}%`,
                transform: `translate(-50%, -50%) rotate(${-piece.rotation}deg)`,
              }}
            >
              {piece.imageData && (
                <img
                  src={piece.imageData}
                  alt={`Piece ${index + 1}`}
                  className="h-full w-full object-cover"
                />
              )}
              {/* Confidence indicator */}
              <div
                className="absolute right-0 bottom-0 left-0 h-1"
                style={{
                  background: `linear-gradient(to right, #22c55e ${piece.positionConfidence * 100}%, #ef4444 ${piece.positionConfidence * 100}%)`,
                }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
