'use client';

import { Grid2X2, Puzzle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { PieceSelectionMode } from '@/types';
import { cn } from '@/lib/utils';

interface PieceModeToggleProps {
  mode: PieceSelectionMode;
  onModeChange: (mode: PieceSelectionMode) => void;
  className?: string;
}

export function PieceModeToggle({ mode, onModeChange, className }: PieceModeToggleProps) {
  return (
    <div className={cn('bg-muted flex gap-1 rounded-lg p-1', className)}>
      <Button
        variant={mode === 'grid' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onModeChange('grid')}
        className="gap-1.5"
      >
        <Grid2X2 className="h-4 w-4" />
        Grid
      </Button>
      <Button
        variant={mode === 'realistic' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onModeChange('realistic')}
        className="gap-1.5"
      >
        <Puzzle className="h-4 w-4" />
        Realistic
      </Button>
    </div>
  );
}
