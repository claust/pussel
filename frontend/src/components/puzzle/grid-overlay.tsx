'use client';

import type { GridSize } from '@/types';
import { GRID_DIMENSIONS } from '@/types';
import { cn } from '@/lib/utils';

interface GridOverlayProps {
  gridSize: GridSize;
  onCellClick: (cellIndex: number) => void;
  selectedCell?: number | null;
  className?: string;
}

export function GridOverlay({ gridSize, onCellClick, selectedCell, className }: GridOverlayProps) {
  const { dimension, totalCells } = GRID_DIMENSIONS[gridSize];

  return (
    <div
      className={cn('grid h-full w-full', className)}
      style={{
        gridTemplateColumns: `repeat(${dimension}, 1fr)`,
        gridTemplateRows: `repeat(${dimension}, 1fr)`,
      }}
    >
      {Array.from({ length: totalCells }).map((_, index) => {
        const isSelected = selectedCell === index;

        return (
          <button
            key={index}
            onClick={() => onCellClick(index)}
            className={cn(
              'flex items-center justify-center border-2 transition-colors',
              isSelected
                ? 'border-blue-500 bg-blue-500/30'
                : 'border-white/50 hover:border-white hover:bg-white/10'
            )}
          >
            <span
              className={cn(
                'text-2xl font-bold',
                isSelected ? 'text-blue-500' : 'text-white drop-shadow-lg'
              )}
            >
              {index + 1}
            </span>
          </button>
        );
      })}
    </div>
  );
}
