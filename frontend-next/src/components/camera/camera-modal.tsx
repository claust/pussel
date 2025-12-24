'use client';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { CameraView } from './camera-view';
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

  const handleCapture = (blob: Blob) => {
    onCapture(blob);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[90vh] max-w-2xl flex-col p-0">
        <DialogHeader className="p-4 pb-0">
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <CameraView
          onCapture={handleCapture}
          onCancel={() => onOpenChange(false)}
          overlayImage={mode === 'piece' ? puzzleImageUrl : null}
          className="flex-1"
        />
      </DialogContent>
    </Dialog>
  );
}
