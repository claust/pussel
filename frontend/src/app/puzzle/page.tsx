'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Camera, Grid, Maximize2, Plus, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { CameraModal } from '@/components/camera';
import { PuzzleDetail, PieceCard } from '@/components/puzzle';
import { usePuzzleStore } from '@/stores/puzzle-store';
import { uploadPuzzle, processPiece } from '@/lib/api';
import { blobToDataUrl } from '@/lib/image-utils';
import type { CameraMode } from '@/types';

export default function PuzzlePage() {
  const router = useRouter();
  const {
    puzzle,
    puzzleImage,
    pieces,
    isLoading,
    error,
    setPuzzle,
    addPiece,
    setLoading,
    setError,
    reset,
  } = usePuzzleStore();

  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>('puzzle');
  const [viewMode, setViewMode] = useState<'grid' | 'fullscreen'>('grid');

  const handleOpenCamera = (mode: CameraMode) => {
    setCameraMode(mode);
    setCameraOpen(true);
  };

  const handleCapture = async (blob: Blob) => {
    setLoading(true);
    setError(null);

    try {
      if (cameraMode === 'puzzle') {
        // Upload puzzle image
        const result = await uploadPuzzle(blob);
        const imageUrl = await blobToDataUrl(blob);
        setPuzzle(result, imageUrl);
      } else if (puzzle) {
        // Process piece
        const result = await processPiece(puzzle.puzzleId, blob);
        // Prefer cleaned image (background removed) from API, fallback to original
        const imageData = result.imageData || (await blobToDataUrl(blob));
        addPiece({ ...result, imageData });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    reset();
    router.push('/');
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b p-4">
        <Button variant="ghost" size="icon" onClick={handleBack}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-semibold">{puzzle ? 'Add Pieces' : 'Capture Puzzle'}</h1>
        {puzzle && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setViewMode(viewMode === 'grid' ? 'fullscreen' : 'grid')}
          >
            {viewMode === 'grid' ? <Maximize2 className="h-5 w-5" /> : <Grid className="h-5 w-5" />}
          </Button>
        )}
        {!puzzle && <div className="w-10" />}
      </header>

      {/* Content */}
      <main className="flex-1 p-4">
        {error && (
          <div className="bg-destructive/10 text-destructive mb-4 rounded-lg p-3 text-sm">
            {error}
          </div>
        )}

        {!puzzle ? (
          // No puzzle yet - show upload prompt
          <Card className="flex h-64 flex-col items-center justify-center">
            <CardContent className="flex flex-col items-center gap-4 pt-6">
              <Camera className="text-muted-foreground h-12 w-12" />
              <p className="text-muted-foreground text-center">
                Take a photo of the complete puzzle to get started
              </p>
              <Button onClick={() => handleOpenCamera('puzzle')} disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Camera className="mr-2 h-4 w-4" />
                )}
                Capture Puzzle
              </Button>
            </CardContent>
          </Card>
        ) : viewMode === 'fullscreen' ? (
          // Fullscreen view
          <PuzzleDetail
            puzzleImage={puzzleImage!}
            pieces={pieces}
            onClick={() => setViewMode('grid')}
            className="h-[calc(100vh-200px)]"
          />
        ) : (
          // Grid view
          <div className="space-y-4">
            {/* Puzzle preview */}
            <PuzzleDetail
              puzzleImage={puzzleImage!}
              pieces={pieces}
              onClick={() => setViewMode('fullscreen')}
              className="h-48"
            />

            {/* Pieces grid */}
            <div className="grid grid-cols-2 gap-3">
              {pieces.map((piece, index) => (
                <PieceCard key={index} piece={piece} index={index} />
              ))}
            </div>

            {/* Add piece button */}
            <Button
              className="w-full"
              onClick={() => handleOpenCamera('piece')}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Plus className="mr-2 h-4 w-4" />
              )}
              Add Piece
            </Button>
          </div>
        )}
      </main>

      {/* Camera Modal */}
      <CameraModal
        open={cameraOpen}
        onOpenChange={setCameraOpen}
        mode={cameraMode}
        onCapture={(blob) => void handleCapture(blob)}
        puzzleImageUrl={puzzleImage}
      />
    </div>
  );
}
