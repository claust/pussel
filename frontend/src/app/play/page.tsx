'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Camera, Loader2, Puzzle, Shuffle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { CameraModal } from '@/components/camera';
import { GameBoard } from '@/components/game';
import { useGameStore } from '@/stores/game-store';
import { uploadPuzzle, cutPuzzle } from '@/lib/api';
import { blobToDataUrl } from '@/lib/image-utils';
import type { GridSize } from '@/types';
import { GRID_DIMENSIONS } from '@/types';

type GamePhase = 'capture' | 'select-grid' | 'loading' | 'playing' | 'complete';

const GRID_OPTIONS: GridSize[] = ['2x2', '3x3', '4x4', '5x5', '6x6'];

export default function PlayPage() {
  const router = useRouter();
  const {
    puzzleImage,
    puzzleWidth,
    puzzleHeight,
    pieces,
    isComplete,
    startGame,
    shufflePieces,
    reset,
  } = useGameStore();

  const [phase, setPhase] = useState<GamePhase>('capture');
  const [cameraOpen, setCameraOpen] = useState(false);
  const [tempPuzzleId, setTempPuzzleId] = useState<string | null>(null);
  const [tempPuzzleImage, setTempPuzzleImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCapture = async (blob: Blob) => {
    setError(null);
    setPhase('loading');

    try {
      const result = await uploadPuzzle(blob);
      const imageUrl = await blobToDataUrl(blob);
      setTempPuzzleId(result.puzzleId);
      setTempPuzzleImage(imageUrl);
      setPhase('select-grid');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload puzzle');
      setPhase('capture');
    }
  };

  const handleSelectGrid = async (gridSize: GridSize) => {
    if (!tempPuzzleId || !tempPuzzleImage) return;

    setError(null);
    setPhase('loading');

    try {
      const { dimension } = GRID_DIMENSIONS[gridSize];
      const response = await cutPuzzle(tempPuzzleId, dimension, dimension);
      startGame(tempPuzzleId, tempPuzzleImage, response);
      setPhase('playing');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cut puzzle');
      setPhase('select-grid');
    }
  };

  const handleBack = () => {
    reset();
    setTempPuzzleId(null);
    setTempPuzzleImage(null);
    setPhase('capture');
    router.push('/');
  };

  const handleRestart = () => {
    reset();
    setTempPuzzleId(null);
    setTempPuzzleImage(null);
    setPhase('capture');
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b p-4">
        <Button variant="ghost" size="icon" onClick={handleBack}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-semibold">
          {phase === 'capture' && 'Capture Puzzle'}
          {phase === 'select-grid' && 'Select Grid Size'}
          {phase === 'loading' && 'Loading...'}
          {phase === 'playing' && 'Play Mode'}
          {phase === 'complete' && 'Completed!'}
        </h1>
        <div className="w-10" />
      </header>

      {/* Content */}
      <main className="flex-1 p-4">
        {error && (
          <div className="bg-destructive/10 text-destructive mb-4 rounded-lg p-3 text-sm">
            {error}
          </div>
        )}

        {/* Phase: Capture */}
        {phase === 'capture' && (
          <Card className="flex h-64 flex-col items-center justify-center">
            <CardContent className="flex flex-col items-center gap-4 pt-6">
              <Puzzle className="text-muted-foreground h-12 w-12" />
              <p className="text-muted-foreground text-center">
                Take a photo of the puzzle you want to solve
              </p>
              <Button onClick={() => setCameraOpen(true)}>
                <Camera className="mr-2 h-4 w-4" />
                Capture Puzzle
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Phase: Select Grid */}
        {phase === 'select-grid' && tempPuzzleImage && (
          <div className="space-y-6">
            {/* Preview */}
            <div className="relative mx-auto aspect-square max-w-sm overflow-hidden rounded-lg">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={tempPuzzleImage}
                alt="Puzzle preview"
                className="h-full w-full object-cover"
              />
            </div>

            {/* Grid size buttons */}
            <div className="space-y-3">
              <p className="text-muted-foreground text-center text-sm">
                How many pieces should the puzzle have?
              </p>
              <div className="grid grid-cols-5 gap-2">
                {GRID_OPTIONS.map((size) => (
                  <Button
                    key={size}
                    variant="outline"
                    className="flex flex-col py-4"
                    onClick={() => void handleSelectGrid(size)}
                  >
                    <span className="text-lg font-semibold">{size}</span>
                    <span className="text-muted-foreground text-xs">
                      {GRID_DIMENSIONS[size].totalCells} pcs
                    </span>
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Phase: Loading */}
        {phase === 'loading' && (
          <div className="flex h-64 flex-col items-center justify-center gap-4">
            <Loader2 className="text-muted-foreground h-12 w-12 animate-spin" />
            <p className="text-muted-foreground">Cutting puzzle into pieces...</p>
          </div>
        )}

        {/* Phase: Playing */}
        {phase === 'playing' && puzzleImage && (
          <div className="space-y-4">
            {/* Game board with drag-and-drop */}
            <GameBoard
              puzzleImage={puzzleImage}
              className="bg-muted/50 mx-auto max-w-lg"
              style={{
                aspectRatio:
                  puzzleWidth && puzzleHeight ? `${puzzleWidth} / ${puzzleHeight}` : '1 / 1',
              }}
            />

            {/* Stats */}
            <div className="text-muted-foreground text-center text-sm">
              {pieces.filter((p) => p.isPlaced).length} / {pieces.length} pieces placed
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <Button variant="outline" className="flex-1" onClick={shufflePieces}>
                <Shuffle className="mr-2 h-4 w-4" />
                Shuffle
              </Button>
              <Button variant="outline" className="flex-1" onClick={handleRestart}>
                New Puzzle
              </Button>
            </div>
          </div>
        )}

        {/* Phase: Complete */}
        {isComplete && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <Card className="mx-4 max-w-sm">
              <CardContent className="flex flex-col items-center gap-4 pt-6">
                <div className="text-4xl">🎉</div>
                <h2 className="text-xl font-bold">Puzzle Complete!</h2>
                <p className="text-muted-foreground text-center">
                  Congratulations! You solved the puzzle.
                </p>
                <Button onClick={handleRestart} className="w-full">
                  Play Again
                </Button>
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      {/* Camera Modal */}
      <CameraModal
        open={cameraOpen}
        onOpenChange={setCameraOpen}
        mode="puzzle"
        onCapture={(blob) => void handleCapture(blob)}
      />
    </div>
  );
}
