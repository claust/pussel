'use client';

import { useState, useCallback } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { CameraView } from './camera-view';
import { cn } from '@/lib/utils';
import type { CameraMode } from '@/types';

interface CameraModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: CameraMode;
  onCapture: (blob: Blob) => void;
  puzzleImageUrl?: string | null;
}

export function CameraModal({
  open,
  onOpenChange,
  mode,
  onCapture,
  puzzleImageUrl,
}: CameraModalProps) {
  const title = mode === 'puzzle' ? 'Capture Puzzle' : 'Capture Piece';
  const [isLandscape, setIsLandscape] = useState(false);

  const handleCapture = (blob: Blob) => {
    onCapture(blob);
    onOpenChange(false);
  };

  const handleOrientationChange = useCallback((landscape: boolean) => {
    setIsLandscape(landscape);
  }, []);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          'flex flex-col p-0',
          isLandscape
            ? 'h-[85vh] w-[95vw] max-w-[1200px] sm:max-w-[1200px]'
            : 'h-[90vh] w-[90vw] max-w-[800px] sm:max-w-[800px]'
        )}
      >
        <DialogHeader className="p-4 pb-0">
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <CameraView
          onCapture={handleCapture}
          onCancel={() => onOpenChange(false)}
          onOrientationChange={handleOrientationChange}
          overlayImage={mode === 'piece' ? puzzleImageUrl : null}
          className="flex-1"
        />
      </DialogContent>
    </Dialog>
  );
}
