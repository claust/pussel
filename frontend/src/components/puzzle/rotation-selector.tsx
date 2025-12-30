'use client';

import { useState } from 'react';
import { RotateCw } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface RotationSelectorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  previewImage: string;
  onConfirm: (rotation: 0 | 90 | 180 | 270) => void;
}

const ROTATIONS = [0, 90, 180, 270] as const;

export function RotationSelector({
  open,
  onOpenChange,
  previewImage,
  onConfirm,
}: RotationSelectorProps) {
  const [selectedRotation, setSelectedRotation] = useState<0 | 90 | 180 | 270>(0);

  const handleConfirm = () => {
    onConfirm(selectedRotation);
    onOpenChange(false);
    setSelectedRotation(0);
  };

  const handleCancel = () => {
    onOpenChange(false);
    setSelectedRotation(0);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Select Rotation</DialogTitle>
        </DialogHeader>

        {/* Preview */}
        <div className="flex justify-center py-4">
          <div className="bg-muted relative h-48 w-48 overflow-hidden rounded-lg">
            <img
              src={previewImage}
              alt="Piece preview"
              className="h-full w-full object-cover transition-transform duration-200"
              style={{ transform: `rotate(${selectedRotation}deg)` }}
            />
          </div>
        </div>

        {/* Rotation options */}
        <div className="flex justify-center gap-2">
          {ROTATIONS.map((rotation) => (
            <Button
              key={rotation}
              variant={selectedRotation === rotation ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedRotation(rotation)}
              className={cn('gap-1', selectedRotation === rotation && 'ring-primary ring-2')}
            >
              <RotateCw className="h-4 w-4" style={{ transform: `rotate(${rotation}deg)` }} />
              {rotation}°
            </Button>
          ))}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleConfirm}>Confirm</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
