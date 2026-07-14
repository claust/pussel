'use client';

import { useEffect, useState } from 'react';
import { Camera, Loader2, AlertCircle, Upload } from 'lucide-react';
import { useCamera } from '@/hooks/use-camera';
import { Button } from '@/components/ui/button';
import { detectPieceRegion } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { PieceRegion } from '@/types';
import { FileUpload } from './file-upload';

// Live piece detection: frames are downscaled to this size before upload
const DETECT_FRAME_MAX_DIM = 320;
// Minimum time between detection requests (the loop is also serialized on the response)
const DETECT_INTERVAL_MS = 400;

interface CameraViewProps {
  onCapture: (blob: Blob) => void;
  onCancel: () => void;
  onOrientationChange?: (isLandscape: boolean) => void;
  overlayImage?: string | null;
  livePieceDetection?: boolean;
  className?: string;
}

export function CameraView({
  onCapture,
  onCancel,
  onOrientationChange,
  overlayImage,
  livePieceDetection = false,
  className,
}: CameraViewProps) {
  const { videoRef, isReady, isLoading, error, dimensions, start, stop, capture } = useCamera();
  const [pieceRegion, setPieceRegion] = useState<PieceRegion | null>(null);

  useEffect(() => {
    void start();
    return () => {
      stop();
    };
  }, [start, stop]);

  // Stream downscaled frames to the backend and overlay the detected piece outline
  useEffect(() => {
    if (!livePieceDetection || !isReady) return;

    let cancelled = false;
    const canvas = document.createElement('canvas');

    const detectLoop = async () => {
      while (!cancelled) {
        const startedAt = Date.now();
        const video = videoRef.current;
        if (video && video.videoWidth > 0) {
          try {
            const scale = DETECT_FRAME_MAX_DIM / Math.max(video.videoWidth, video.videoHeight);
            canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
            canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
            canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise<Blob | null>((resolve) =>
              canvas.toBlob(resolve, 'image/jpeg', 0.7)
            );
            if (blob) {
              const region = await detectPieceRegion(blob);
              if (!cancelled) setPieceRegion(region);
            }
          } catch {
            // Keep the camera usable even if detection fails; try again next tick
            if (!cancelled) setPieceRegion(null);
          }
        }
        const elapsed = Date.now() - startedAt;
        await new Promise((resolve) =>
          setTimeout(resolve, Math.max(DETECT_INTERVAL_MS - elapsed, 100))
        );
      }
    };

    void detectLoop();
    return () => {
      cancelled = true;
      setPieceRegion(null);
    };
  }, [livePieceDetection, isReady, videoRef]);

  // Notify parent of orientation changes
  useEffect(() => {
    if (dimensions && onOrientationChange) {
      onOrientationChange(dimensions.isLandscape);
    }
  }, [dimensions, onOrientationChange]);

  const handleCapture = async () => {
    const blob = await capture();
    if (blob) {
      onCapture(blob);
    }
  };

  if (error) {
    return (
      <div className={cn('flex flex-col items-center justify-center gap-4 p-8', className)}>
        <AlertCircle className="text-destructive h-12 w-12" />
        <p className="text-destructive text-center">{error}</p>
        <p className="text-muted-foreground text-center text-sm">You can upload an image instead</p>
        <div className="flex flex-col gap-2">
          <FileUpload onFileSelect={onCapture}>
            <Button className="gap-2">
              <Upload className="h-4 w-4" />
              Upload Image
            </Button>
          </FileUpload>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button variant="secondary" onClick={() => void start()}>
              Retry Camera
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('relative flex flex-col', className)}>
      {/* Video container */}
      <div className="relative flex-1 overflow-hidden rounded-lg bg-black">
        <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" />

        {/* Overlay image (for piece capture mode) */}
        {overlayImage && isReady && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <img
              src={overlayImage}
              alt="Puzzle reference"
              className="h-full w-full object-contain opacity-30"
            />
          </div>
        )}

        {/* Live piece detection outline; slice mirrors the video's object-cover mapping */}
        {livePieceDetection && isReady && dimensions && (
          <>
            {pieceRegion?.found && pieceRegion.polygon.length >= 3 && (
              <svg
                className="pointer-events-none absolute inset-0 h-full w-full"
                viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                preserveAspectRatio="xMidYMid slice"
              >
                <polygon
                  points={pieceRegion.polygon
                    .map((p) => `${p.x * dimensions.width},${p.y * dimensions.height}`)
                    .join(' ')}
                  fill="rgb(34 197 94 / 0.15)"
                  stroke="rgb(34 197 94)"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            )}
            <div className="pointer-events-none absolute top-3 left-1/2 -translate-x-1/2">
              <span
                className={cn(
                  'rounded-full px-3 py-1 text-xs font-medium text-white',
                  pieceRegion?.found ? 'bg-green-600/80' : 'bg-black/50'
                )}
              >
                {pieceRegion?.found ? 'Piece detected' : 'Looking for piece…'}
              </span>
            </div>
          </>
        )}

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <Loader2 className="h-8 w-8 animate-spin text-white" />
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4 p-4">
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          size="lg"
          className="h-16 w-16 rounded-full"
          onClick={() => void handleCapture()}
          disabled={!isReady || isLoading}
        >
          <Camera className="h-6 w-6" />
        </Button>
        <FileUpload onFileSelect={onCapture}>
          <Button variant="outline" size="icon" className="h-10 w-10">
            <Upload className="h-4 w-4" />
          </Button>
        </FileUpload>
      </div>
    </div>
  );
}
