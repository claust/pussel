'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import {
  DndContext,
  DragEndEvent,
  DragStartEvent,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import { DraggablePiece } from './draggable-piece';
import { useGameStore } from '@/stores/game-store';

interface GameBoardProps {
  puzzleImage: string;
  className?: string;
  style?: React.CSSProperties;
}

export function GameBoard({ puzzleImage, className = '', style }: GameBoardProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const { pieces, puzzleWidth, puzzleHeight, movePiece, placePiece, bringToFront } = useGameStore();

  // Configure sensors for both mouse and touch
  const mouseSensor = useSensor(MouseSensor, {
    activationConstraint: {
      distance: 5, // 5px movement before drag starts
    },
  });

  const touchSensor = useSensor(TouchSensor, {
    activationConstraint: {
      delay: 100, // 100ms delay before drag starts
      tolerance: 5, // 5px tolerance
    },
  });

  const sensors = useSensors(mouseSensor, touchSensor);

  // Track container size for coordinate calculations
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setContainerSize({ width: rect.width, height: rect.height });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  const handleDragStart = useCallback(
    (event: DragStartEvent) => {
      const pieceId = event.active.id as string;
      bringToFront(pieceId);
    },
    [bringToFront]
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const pieceId = event.active.id as string;
      const { delta } = event;

      if (containerSize.width === 0 || containerSize.height === 0) return;

      // Find the piece
      const piece = pieces.find((p) => p.id === pieceId);
      if (!piece || piece.isPlaced) return;

      // Calculate new normalized position
      const deltaX = delta.x / containerSize.width;
      const deltaY = delta.y / containerSize.height;

      const newX = Math.max(0, Math.min(1, piece.currentX + deltaX));
      const newY = Math.max(0, Math.min(1, piece.currentY + deltaY));

      // Update piece position
      movePiece(pieceId, newX, newY);

      // Check if piece should snap to correct position
      placePiece(pieceId);
    },
    [containerSize, pieces, movePiece, placePiece]
  );

  return (
    <DndContext sensors={sensors} onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
      <div
        ref={containerRef}
        className={`relative overflow-hidden rounded-lg ${className}`}
        style={{ touchAction: 'none', ...style }}
      >
        {/* Dimmed puzzle background */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={puzzleImage}
          alt="Puzzle background"
          className="h-full w-full object-cover opacity-30"
          draggable={false}
        />

        {/* Draggable pieces */}
        {containerSize.width > 0 &&
          pieces.map((piece) => (
            <DraggablePiece
              key={piece.id}
              piece={piece}
              containerWidth={containerSize.width}
              containerHeight={containerSize.height}
              puzzleWidth={puzzleWidth}
              puzzleHeight={puzzleHeight}
            />
          ))}
      </div>
    </DndContext>
  );
}
