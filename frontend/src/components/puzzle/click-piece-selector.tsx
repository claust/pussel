'use client';

import type { MouseEvent } from 'react';
import { useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface ClickPieceSelectorProps {
  onPositionClick: (x: number, y: number) => void;
  isLoading?: boolean;
  className?: string;
}

export function ClickPieceSelector({
  onPositionClick,
  isLoading = false,
  className,
}: ClickPieceSelectorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [clickPosition, setClickPosition] = useState<{ x: number; y: number } | null>(null);

  const handleClick = (e: MouseEvent<HTMLDivElement>) => {
    if (isLoading || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width; // Normalized 0-1
    const y = (e.clientY - rect.top) / rect.height; // Normalized 0-1

    // Clamp values to valid range
    const clampedX = Math.max(0, Math.min(1, x));
    const clampedY = Math.max(0, Math.min(1, y));

    setClickPosition({ x: clampedX, y: clampedY });
    onPositionClick(clampedX, clampedY);
  };

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      className={cn(
        'relative h-full w-full cursor-crosshair',
        isLoading && 'cursor-wait opacity-50',
        className
      )}
    >
      {/* Click indicator */}
      {clickPosition && (
        <div
          className="pointer-events-none absolute h-8 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full border-4 border-blue-500 bg-blue-500/30 shadow-lg"
          style={{
            left: `${clickPosition.x * 100}%`,
            top: `${clickPosition.y * 100}%`,
          }}
        >
          {/* Inner dot */}
          <div className="absolute top-1/2 left-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-blue-500" />
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/30">
          <div className="flex flex-col items-center gap-2">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
            <span className="text-sm font-medium text-white drop-shadow-lg">
              Generating piece...
            </span>
          </div>
        </div>
      )}

      {/* Instruction hint */}
      {!isLoading && !clickPosition && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-lg bg-black/50 px-4 py-2 text-sm font-medium text-white">
            Click to select piece position
          </span>
        </div>
      )}
    </div>
  );
}
