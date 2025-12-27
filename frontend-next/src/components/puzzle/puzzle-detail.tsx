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
      className={cn('relative cursor-pointer overflow-hidden rounded-lg', className)}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                onClick();
              }
            }
          : undefined
      }
    >
      {/* Puzzle background with dark overlay */}
      <div className="relative">
        <img src={puzzleImage} alt="Puzzle" className="h-full w-full object-contain" />
        <div className="absolute inset-0 bg-black/30" />
      </div>

      {/* Pieces overlay */}
      {pieces.map((piece, index) => (
        <div
          key={index}
          className="absolute border-2 border-green-500 shadow-lg"
          style={{
            left: `${piece.position.x * 100}%`,
            top: `${piece.position.y * 100}%`,
            width: `${pieceSize}%`,
            height: `${pieceSize}%`,
            transform: `translate(-50%, -50%) rotate(${piece.rotation}deg)`,
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
              background: `linear-gradient(to right, #22c55e ${piece.confidence * 100}%, #ef4444 ${piece.confidence * 100}%)`,
            }}
          />
        </div>
      ))}
    </div>
  );
}
