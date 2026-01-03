'use client';

import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import type { GamePiece } from '@/types';

interface DraggablePieceProps {
  piece: GamePiece;
  containerWidth: number;
  containerHeight: number;
  puzzleWidth: number;
  puzzleHeight: number;
}

export function DraggablePiece({
  piece,
  containerWidth,
  containerHeight,
  puzzleWidth,
  puzzleHeight,
}: DraggablePieceProps) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: piece.id,
    disabled: piece.isPlaced,
  });

  // Calculate pixel position from normalized coordinates
  const pixelX = piece.currentX * containerWidth;
  const pixelY = piece.currentY * containerHeight;

  // Calculate scale factor to match object-cover behavior of background image
  // Use max() so pieces scale the same way as the background (cover, not contain)
  const scaleX = containerWidth / puzzleWidth;
  const scaleY = containerHeight / puzzleHeight;
  const scale = Math.max(scaleX, scaleY);
  const displayWidth = piece.width * scale;
  const displayHeight = piece.height * scale;

  const style: React.CSSProperties = {
    position: 'absolute',
    left: pixelX,
    top: pixelY,
    transform: CSS.Translate.toString(transform),
    zIndex: isDragging ? 1000 : piece.zIndex,
    cursor: piece.isPlaced ? 'default' : isDragging ? 'grabbing' : 'grab',
    touchAction: 'none',
    transition: isDragging ? undefined : 'box-shadow 0.2s ease',
  };

  return (
    <div ref={setNodeRef} style={style} {...listeners} {...attributes} className="select-none">
      <div
        className="relative"
        style={{
          transform: 'translate(-50%, -50%)',
          filter: isDragging
            ? 'drop-shadow(0 8px 16px rgba(0,0,0,0.3))'
            : 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))',
          transition: 'filter 0.2s ease, transform 0.1s ease',
          scale: isDragging ? '1.05' : '1',
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={piece.imageData}
          alt={piece.id}
          width={displayWidth}
          height={displayHeight}
          className="pointer-events-none"
          style={{
            maxWidth: 'none',
            borderRadius: piece.isPlaced ? '2px' : undefined,
            outline: piece.isPlaced ? '3px solid #22c55e' : undefined,
            outlineOffset: '-1px',
          }}
          draggable={false}
        />
      </div>
    </div>
  );
}
