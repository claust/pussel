'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Camera, Loader2, Plus, RotateCcw, ScanLine } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { CameraModal } from '@/components/camera';
import { CornerAdjust, PieceCard, PuzzleDetail } from '@/components/puzzle';
import { usePuzzleStore } from '@/stores/puzzle-store';
import { detectFrame, uploadPuzzle, processPiece } from '@/lib/api';
import { blobToDataUrl, dataUrlToBlob } from '@/lib/image-utils';
import type { DetectFrameResult } from '@/types';

type RealModePhase = 'capture-puzzle' | 'confirm-trim' | 'adjust-corners' | 'solving';

// Real puzzles have many small pieces; the model predicts a continuous position,
// so the overlay uses a fixed piece size rather than a grid-derived one.
const PIECE_SIZE_RATIO = 0.12;

const LOW_CONFIDENCE_THRESHOLD = 0.4;

export default function RealModePage() {
  const [phase, setPhase] = useState<RealModePhase>('capture-puzzle');
  const [puzzleCameraOpen, setPuzzleCameraOpen] = useState(false);
  const [pieceCameraOpen, setPieceCameraOpen] = useState(false);
  const [rawPhotoBlob, setRawPhotoBlob] = useState<Blob | null>(null);
  const [rawPhotoUrl, setRawPhotoUrl] = useState<string | null>(null);
  const [detection, setDetection] = useState<DetectFrameResult | null>(null);

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

  // Clean up shared store state when leaving the page
  useEffect(() => {
    return () => {
      reset();
    };
  }, [reset]);

  const handlePuzzlePhoto = async (blob: Blob) => {
    setLoading(true);
    setError(null);
    try {
      const photoUrl = await blobToDataUrl(blob);
      setRawPhotoBlob(blob);
      setRawPhotoUrl(photoUrl);

      const result = await detectFrame(blob);
      setDetection(result);
      setPhase('confirm-trim');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to detect puzzle');
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptTrim = async () => {
    if (!detection) return;

    setLoading(true);
    setError(null);
    try {
      const trimmedBlob = await dataUrlToBlob(detection.trimmedImageUrl);
      const result = await uploadPuzzle(trimmedBlob);
      setPuzzle(result, detection.trimmedImageUrl);
      setPhase('solving');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload puzzle');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyCorners = async (corners: DetectFrameResult['corners']) => {
    if (!rawPhotoBlob) return;

    setLoading(true);
    setError(null);
    try {
      const result = await detectFrame(rawPhotoBlob, corners);
      setDetection(result);
      setPhase('confirm-trim');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trim puzzle');
    } finally {
      setLoading(false);
    }
  };

  const handlePieceCapture = async (blob: Blob) => {
    if (!puzzle) return;

    setLoading(true);
    setError(null);
    try {
      const result = await processPiece(puzzle.puzzleId, blob);
      addPiece(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process piece');
    } finally {
      setLoading(false);
    }
  };

  const handleRetake = () => {
    setDetection(null);
    setRawPhotoBlob(null);
    setRawPhotoUrl(null);
    setError(null);
    setPhase('capture-puzzle');
    setPuzzleCameraOpen(true);
  };

  const handleNewPuzzle = () => {
    reset();
    setDetection(null);
    setRawPhotoBlob(null);
    setRawPhotoUrl(null);
    setPhase('capture-puzzle');
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b p-4">
        <Button asChild variant="ghost" size="icon">
          <Link href="/">
            <ArrowLeft className="h-5 w-5" />
          </Link>
        </Button>
        <div className="text-center">
          <h1 className="text-lg font-semibold">Solve Real Puzzle</h1>
          <p className="text-muted-foreground text-sm">
            {phase === 'solving'
              ? `${pieces.length} ${pieces.length === 1 ? 'piece' : 'pieces'} captured`
              : 'Photograph your puzzle to get started'}
          </p>
        </div>
        <div className="w-10" />
      </header>

      {/* Content */}
      <main className="mx-auto w-full max-w-2xl flex-1 p-4">
        {error && (
          <div className="bg-destructive/10 text-destructive mb-4 rounded-lg p-3 text-sm">
            {error}
          </div>
        )}

        {phase === 'capture-puzzle' && (
          <Card>
            <CardHeader className="text-center">
              <CardTitle>Capture Your Puzzle</CardTitle>
              <CardDescription>
                Take a photo of the complete puzzle picture — the box lid or the finished puzzle. It
                will be automatically trimmed to just the picture.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                className="w-full gap-2"
                size="lg"
                onClick={() => setPuzzleCameraOpen(true)}
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Camera className="h-5 w-5" />
                )}
                {isLoading ? 'Detecting puzzle...' : 'Take Puzzle Photo'}
              </Button>
            </CardContent>
          </Card>
        )}

        {phase === 'confirm-trim' && detection && (
          <div className="space-y-4">
            <div className="text-center">
              <h2 className="text-lg font-semibold">Is this your puzzle?</h2>
              <p className="text-muted-foreground text-sm">
                The photo was trimmed to the detected puzzle picture.
              </p>
            </div>

            {detection.confidence < LOW_CONFIDENCE_THRESHOLD && (
              <div className="rounded-lg bg-yellow-500/10 p-3 text-sm text-yellow-600 dark:text-yellow-400">
                Detection looks uncertain — consider adjusting the corners manually.
              </div>
            )}

            <div className="flex justify-center">
              <img
                src={detection.trimmedImageUrl}
                alt="Trimmed puzzle"
                className="block h-auto max-h-[60vh] w-auto max-w-full rounded-lg"
              />
            </div>

            <div className="flex flex-col gap-3">
              <Button
                className="w-full gap-2"
                size="lg"
                onClick={() => void handleAcceptTrim()}
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ScanLine className="h-4 w-4" />
                )}
                Use This
              </Button>
              <div className="flex gap-3">
                <Button
                  variant="secondary"
                  className="flex-1"
                  onClick={() => setPhase('adjust-corners')}
                  disabled={isLoading}
                >
                  Adjust Corners
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 gap-2"
                  onClick={handleRetake}
                  disabled={isLoading}
                >
                  <RotateCcw className="h-4 w-4" />
                  Retake
                </Button>
              </div>
            </div>
          </div>
        )}

        {phase === 'adjust-corners' && detection && rawPhotoUrl && (
          <CornerAdjust
            imageUrl={rawPhotoUrl}
            initialCorners={detection.corners}
            onApply={(corners) => void handleApplyCorners(corners)}
            onCancel={() => setPhase('confirm-trim')}
            isLoading={isLoading}
          />
        )}

        {phase === 'solving' && puzzle && puzzleImage && (
          <div className="space-y-4">
            <PuzzleDetail
              puzzleImage={puzzleImage}
              pieces={pieces}
              pieceSizeRatio={PIECE_SIZE_RATIO}
            />

            <Button
              className="w-full gap-2"
              size="lg"
              onClick={() => setPieceCameraOpen(true)}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              {isLoading ? 'Predicting position...' : 'Capture Piece'}
            </Button>

            {pieces.length > 0 && (
              <div className="grid grid-cols-2 gap-3">
                {pieces.map((piece, index) => (
                  <PieceCard key={index} piece={piece} index={index} />
                ))}
              </div>
            )}

            <Button
              variant="outline"
              className="w-full"
              onClick={handleNewPuzzle}
              disabled={isLoading}
            >
              New Puzzle
            </Button>
          </div>
        )}
      </main>

      {/* Puzzle photo camera */}
      <CameraModal
        open={puzzleCameraOpen}
        onOpenChange={setPuzzleCameraOpen}
        mode="puzzle"
        onCapture={(blob) => void handlePuzzlePhoto(blob)}
      />

      {/* Piece capture camera with live piece-detection overlay */}
      <CameraModal
        open={pieceCameraOpen}
        onOpenChange={setPieceCameraOpen}
        mode="piece"
        onCapture={(blob) => void handlePieceCapture(blob)}
        livePieceDetection
      />
    </div>
  );
}
