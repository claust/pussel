'use client';

import { useState, useEffect, use } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Grid, Maximize2, Plus, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  PuzzleDetail,
  PieceCard,
  GridOverlay,
  RotationSelector,
  ClickPieceSelector,
  PieceModeToggle,
} from '@/components/puzzle';
import { usePuzzleStore } from '@/stores/puzzle-store';
import { uploadPuzzle, processPiece, generateRealisticPiece } from '@/lib/api';
import { getTestPuzzleById } from '@/lib/test-puzzles';
import { cropCell, blobToDataUrl, dataUrlToBlob } from '@/lib/image-utils';
import type { PieceSelectionMode } from '@/types';

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function TestPuzzlePage({ params }: PageProps) {
  const { id } = use(params);
  const router = useRouter();
  const testPuzzle = getTestPuzzleById(id);

  const {
    puzzle,
    puzzleImage,
    pieces,
    gridSize,
    isLoading,
    error,
    setPuzzle,
    addPiece,
    setLoading,
    setError,
    reset,
  } = usePuzzleStore();

  const [viewMode, setViewMode] = useState<'grid' | 'fullscreen'>('grid');
  const [showGridOverlay, setShowGridOverlay] = useState(false);
  const [selectedCell, setSelectedCell] = useState<number | null>(null);
  const [cellPreview, setCellPreview] = useState<string | null>(null);
  const [showRotationSelector, setShowRotationSelector] = useState(false);
  const [puzzleBlob, setPuzzleBlob] = useState<Blob | null>(null);
  const [pieceSelectionMode, setPieceSelectionMode] = useState<PieceSelectionMode>('grid');

  // Load and upload the test puzzle on mount
  useEffect(() => {
    if (!testPuzzle) {
      router.push('/test-mode/select');
      return;
    }

    const loadPuzzle = async () => {
      setLoading(true);
      try {
        // Fetch the test puzzle image
        const response = await fetch(testPuzzle.path);
        if (!response.ok) {
          throw new Error('Failed to load test puzzle');
        }
        const blob = await response.blob();
        setPuzzleBlob(blob);

        // Upload to backend
        const result = await uploadPuzzle(blob);
        const imageUrl = await blobToDataUrl(blob);
        setPuzzle(result, imageUrl);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load puzzle');
      } finally {
        setLoading(false);
      }
    };

    void loadPuzzle();

    return () => {
      reset();
    };
  }, [testPuzzle, router, setPuzzle, setLoading, setError, reset]);

  const handleCellClick = async (cellIndex: number) => {
    if (!puzzleBlob) return;

    setSelectedCell(cellIndex);

    // Generate preview for rotation selector
    try {
      const croppedBlob = await cropCell(puzzleBlob, gridSize, cellIndex, 0);
      const previewUrl = await blobToDataUrl(croppedBlob);
      setCellPreview(previewUrl);
      setShowRotationSelector(true);
    } catch {
      setError('Failed to crop cell');
    }
  };

  const handleRotationConfirm = async (rotation: 0 | 90 | 180 | 270) => {
    if (!puzzleBlob || selectedCell === null || !puzzle) return;

    setLoading(true);
    setShowGridOverlay(false);

    try {
      // Crop the cell with rotation
      const croppedBlob = await cropCell(puzzleBlob, gridSize, selectedCell, rotation);
      const imageUrl = await blobToDataUrl(croppedBlob);

      // Process the piece
      const result = await processPiece(puzzle.puzzleId, croppedBlob);
      addPiece({ ...result, imageData: imageUrl });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process piece');
    } finally {
      setLoading(false);
      setSelectedCell(null);
      setCellPreview(null);
    }
  };

  const handleRealisticPieceClick = async (centerX: number, centerY: number) => {
    if (!puzzle) return;

    setLoading(true);
    setShowGridOverlay(false);

    try {
      // Generate realistic piece from backend
      const generatedPiece = await generateRealisticPiece(puzzle.puzzleId, centerX, centerY);

      // Convert base64 to blob for ML processing
      const pieceBlob = await dataUrlToBlob(generatedPiece.imageData);

      // Process with ML model to get position prediction
      const result = await processPiece(puzzle.puzzleId, pieceBlob);
      addPiece({ ...result, imageData: generatedPiece.imageData });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate piece');
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    reset();
    router.push('/test-mode/select');
  };

  if (!testPuzzle) {
    return null;
  }

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b p-4">
        <Button variant="ghost" size="icon" onClick={handleBack}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div className="text-center">
          <h1 className="text-lg font-semibold">{testPuzzle.name}</h1>
          <p className="text-muted-foreground text-sm">{gridSize} grid</p>
        </div>
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
      <main className="mx-auto w-full max-w-2xl flex-1 p-4">
        {error && (
          <div className="bg-destructive/10 text-destructive mb-4 rounded-lg p-3 text-sm">
            {error}
          </div>
        )}

        {isLoading && !puzzle ? (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="text-primary h-8 w-8 animate-spin" />
          </div>
        ) : puzzle ? (
          <>
            {viewMode === 'fullscreen' ? (
              // Fullscreen view
              <PuzzleDetail
                puzzleImage={puzzleImage!}
                pieces={pieces}
                gridSize={gridSize}
                onClick={() => setViewMode('grid')}
                className="h-[calc(100vh-200px)]"
              />
            ) : (
              // Grid view
              <div className="space-y-4">
                {/* Puzzle preview with optional grid overlay */}
                <div className="relative">
                  {showGridOverlay ? (
                    <div className="flex justify-center">
                      <div className="relative overflow-hidden rounded-lg">
                        <img
                          src={puzzleImage!}
                          alt="Puzzle"
                          className="block h-auto max-h-[80vh] w-auto max-w-full"
                        />
                        <div className="absolute inset-0">
                          {pieceSelectionMode === 'grid' ? (
                            <GridOverlay
                              gridSize={gridSize}
                              onCellClick={(index) => void handleCellClick(index)}
                              selectedCell={selectedCell}
                            />
                          ) : (
                            <ClickPieceSelector
                              onPositionClick={(x, y) => void handleRealisticPieceClick(x, y)}
                              isLoading={isLoading}
                            />
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <PuzzleDetail
                      puzzleImage={puzzleImage!}
                      pieces={pieces}
                      gridSize={gridSize}
                      onClick={() => setViewMode('fullscreen')}
                    />
                  )}
                </div>

                {/* Pieces grid */}
                {pieces.length > 0 && (
                  <div className="grid grid-cols-2 gap-3">
                    {pieces.map((piece, index) => (
                      <PieceCard key={index} piece={piece} index={index} />
                    ))}
                  </div>
                )}

                {/* Mode toggle and add piece button */}
                <div className="flex flex-col gap-3">
                  {showGridOverlay && (
                    <div className="flex justify-center">
                      <PieceModeToggle
                        mode={pieceSelectionMode}
                        onModeChange={setPieceSelectionMode}
                      />
                    </div>
                  )}
                  <Button
                    className="w-full"
                    onClick={() => setShowGridOverlay(!showGridOverlay)}
                    disabled={isLoading}
                    variant={showGridOverlay ? 'secondary' : 'default'}
                  >
                    {isLoading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Plus className="mr-2 h-4 w-4" />
                    )}
                    {showGridOverlay
                      ? 'Cancel Selection'
                      : pieceSelectionMode === 'grid'
                        ? 'Add Piece from Grid'
                        : 'Add Realistic Piece'}
                  </Button>
                </div>
              </div>
            )}
          </>
        ) : null}
      </main>

      {/* Rotation Selector */}
      {cellPreview && (
        <RotationSelector
          open={showRotationSelector}
          onOpenChange={setShowRotationSelector}
          previewImage={cellPreview}
          onConfirm={(rotation) => void handleRotationConfirm(rotation)}
        />
      )}
    </div>
  );
}
