'use client';

import type { PointerEvent } from 'react';
import { useRef, useState } from 'react';
import { Check, Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { Corner, QuadCorners } from '@/types';

type CornerName = keyof QuadCorners;

const CORNER_NAMES: CornerName[] = ['topLeft', 'topRight', 'bottomRight', 'bottomLeft'];

const CORNER_LABELS: Record<CornerName, string> = {
  topLeft: 'Top left corner',
  topRight: 'Top right corner',
  bottomRight: 'Bottom right corner',
  bottomLeft: 'Bottom left corner',
};

interface CornerAdjustProps {
  imageUrl: string; // the raw (untrimmed) photo
  initialCorners: QuadCorners;
  onApply: (corners: QuadCorners) => void;
  onCancel: () => void;
  isLoading?: boolean;
  className?: string;
}

export function CornerAdjust({
  imageUrl,
  initialCorners,
  onApply,
  onCancel,
  isLoading = false,
  className,
}: CornerAdjustProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [corners, setCorners] = useState<QuadCorners>(initialCorners);
  const [dragging, setDragging] = useState<CornerName | null>(null);

  const updateCorner = (name: CornerName, e: PointerEvent<HTMLButtonElement>) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const corner: Corner = {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
    setCorners((prev) => ({ ...prev, [name]: corner }));
  };

  const handlePointerDown = (name: CornerName) => (e: PointerEvent<HTMLButtonElement>) => {
    if (isLoading) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(name);
    updateCorner(name, e);
  };

  const handlePointerMove = (name: CornerName) => (e: PointerEvent<HTMLButtonElement>) => {
    if (dragging !== name) return;
    updateCorner(name, e);
  };

  // Reset on up, cancel, or lost capture so `dragging` can never get stuck
  // (e.g. an OS gesture cancels the pointer or the window loses focus).
  const handlePointerEnd = () => setDragging(null);

  const polygonPoints = CORNER_NAMES.map(
    (name) => `${corners[name].x * 100},${corners[name].y * 100}`
  ).join(' ');

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      <div className="flex justify-center">
        <div ref={containerRef} className="relative touch-none overflow-hidden rounded-lg">
          <img
            src={imageUrl}
            alt="Puzzle photo"
            className="block h-auto max-h-[70vh] w-auto max-w-full select-none"
            draggable={false}
          />

          {/* Quadrilateral outline */}
          <svg
            className="pointer-events-none absolute inset-0 h-full w-full"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            <polygon
              points={polygonPoints}
              fill="rgb(59 130 246 / 0.15)"
              stroke="rgb(59 130 246)"
              strokeWidth="0.5"
              vectorEffect="non-scaling-stroke"
            />
          </svg>

          {/* Draggable corner handles */}
          {CORNER_NAMES.map((name) => (
            <button
              key={name}
              type="button"
              disabled={isLoading}
              data-testid={`corner-handle-${name}`}
              aria-label={`${CORNER_LABELS[name]} at ${Math.round(corners[name].x * 100)}%, ${Math.round(corners[name].y * 100)}%`}
              className={cn(
                'absolute h-8 w-8 -translate-x-1/2 -translate-y-1/2 cursor-grab rounded-full border-4 border-blue-500 bg-white/80 shadow-lg',
                dragging === name && 'cursor-grabbing border-blue-600 bg-blue-100',
                isLoading && 'cursor-wait opacity-50'
              )}
              style={{
                left: `${corners[name].x * 100}%`,
                top: `${corners[name].y * 100}%`,
              }}
              onPointerDown={handlePointerDown(name)}
              onPointerMove={handlePointerMove(name)}
              onPointerUp={handlePointerEnd}
              onPointerCancel={handlePointerEnd}
              onLostPointerCapture={handlePointerEnd}
            />
          ))}
        </div>
      </div>

      <p className="text-muted-foreground text-center text-sm">
        Drag the handles to the corners of the puzzle picture.
      </p>

      <div className="flex gap-3">
        <Button variant="outline" className="flex-1 gap-2" onClick={onCancel} disabled={isLoading}>
          <X className="h-4 w-4" />
          Cancel
        </Button>
        <Button className="flex-1 gap-2" onClick={() => onApply(corners)} disabled={isLoading}>
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
          Apply
        </Button>
      </div>
    </div>
  );
}
