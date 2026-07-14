'use client';

import { useEffect, useRef, useState } from 'react';
import { AlertCircle, Camera, Loader2, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useCamera } from '@/hooks/use-camera';
import { ApiError, detectPieceRegion } from '@/lib/api';
import { playCommitSound } from '@/lib/capture-sound';
import { computeSharpness, computeSignature } from '@/lib/frame-quality';
import { PieceTracker, type NormalizedBBox } from '@/lib/piece-tracker';
import { useCaptureQueueStore } from '@/stores/capture-queue-store';
import { cn } from '@/lib/utils';
import type { PieceRegion } from '@/types';
import { FileUpload } from './file-upload';

// Live piece detection: frames are downscaled to this size before upload
const DETECT_FRAME_MAX_DIM = 320;
// Minimum time between detection requests (the loop is also serialized on the response)
const DETECT_INTERVAL_MS = 400;
// Analysis crop size for sharpness/signature computation
const ANALYSIS_MAX_DIM = 96;
// Long-side cap for the stored best-frame crop sent to prediction
const SNAPSHOT_MAX_DIM = 800;
// Extra context kept around the detected bbox, as a fraction of its size
const BBOX_MARGIN = 0.12;
// How long the "piece captured" confirmation stays visible
const FLASH_MS = 1200;

interface LivePieceCaptureProps {
  className?: string;
}

interface CropRect {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

function bboxToCrop(bbox: NormalizedBBox, videoWidth: number, videoHeight: number): CropRect {
  const mx = bbox.width * BBOX_MARGIN;
  const my = bbox.height * BBOX_MARGIN;
  const x1 = Math.max(0, bbox.x - mx);
  const y1 = Math.max(0, bbox.y - my);
  const x2 = Math.min(1, bbox.x + bbox.width + mx);
  const y2 = Math.min(1, bbox.y + bbox.height + my);
  return {
    sx: Math.round(x1 * videoWidth),
    sy: Math.round(y1 * videoHeight),
    sw: Math.max(1, Math.round((x2 - x1) * videoWidth)),
    sh: Math.max(1, Math.round((y2 - y1) * videoHeight)),
  };
}

/**
 * Embedded camera with the auto-capture pipeline: continuously detects the
 * piece held in front of the camera, keeps the sharpest view of it, and when
 * the piece leaves the frame (or is swapped for another) commits that best
 * frame to the capture queue with a sound + visual confirmation.
 */
export function LivePieceCapture({ className }: LivePieceCaptureProps) {
  const { videoRef, isReady, isLoading, error, dimensions, start, stop, capture } = useCamera();
  const [pieceRegion, setPieceRegion] = useState<PieceRegion | null>(null);
  const [tracking, setTracking] = useState(false);
  const [flash, setFlash] = useState(false);
  const enqueue = useCaptureQueueStore((s) => s.enqueue);

  const flashTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    void start();
    return () => {
      stop();
    };
  }, [start, stop]);

  useEffect(() => {
    return () => {
      if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
    };
  }, []);

  const commitBlob = (blob: Blob) => {
    enqueue({
      id: typeof crypto !== 'undefined' ? crypto.randomUUID() : `capture-${Date.now()}`,
      blob,
      imageUrl: URL.createObjectURL(blob),
      capturedAt: Date.now(),
    });
    playCommitSound();
    setFlash(true);
    if (flashTimeoutRef.current) clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = setTimeout(() => setFlash(false), FLASH_MS);
  };
  const commitBlobRef = useRef(commitBlob);
  commitBlobRef.current = commitBlob;

  // Set by the capture loop; drops the active track and its pending snapshot
  const abandonTrackRef = useRef<() => void>(() => {});

  // The auto-capture loop: stream downscaled frames to the piece detector,
  // track the piece across responses, and keep its best full-res crop.
  useEffect(() => {
    if (!isReady) return;

    let cancelled = false;
    const controller = new AbortController();
    const tracker = new PieceTracker();
    // Full-res copy of the frame that was sent for detection, so the returned
    // bbox is cropped from the exact pixels it was computed on
    const frameCanvas = document.createElement('canvas');
    const detectCanvas = document.createElement('canvas');
    const analysisCanvas = document.createElement('canvas');
    // Best-frame snapshot per track id (only the active track accumulates)
    const snapshots = new Map<string, HTMLCanvasElement>();

    abandonTrackRef.current = () => {
      tracker.abandon();
      snapshots.clear();
    };

    const commitSnapshot = (trackId: string) => {
      const canvas = snapshots.get(trackId);
      snapshots.delete(trackId);
      if (!canvas) return;
      canvas.toBlob(
        (blob) => {
          if (blob && !cancelled) commitBlobRef.current(blob);
        },
        'image/jpeg',
        0.92
      );
    };

    const detectLoop = async () => {
      while (!cancelled) {
        const startedAt = Date.now();
        const video = videoRef.current;
        if (video && video.videoWidth > 0) {
          try {
            frameCanvas.width = video.videoWidth;
            frameCanvas.height = video.videoHeight;
            frameCanvas.getContext('2d')?.drawImage(video, 0, 0);

            // Only ever downscale; never enlarge a stream smaller than the target
            const scale = Math.min(
              1,
              DETECT_FRAME_MAX_DIM / Math.max(video.videoWidth, video.videoHeight)
            );
            detectCanvas.width = Math.max(1, Math.round(video.videoWidth * scale));
            detectCanvas.height = Math.max(1, Math.round(video.videoHeight * scale));
            detectCanvas
              .getContext('2d')
              ?.drawImage(video, 0, 0, detectCanvas.width, detectCanvas.height);
            const blob = await new Promise<Blob | null>((resolve) =>
              detectCanvas.toBlob(resolve, 'image/jpeg', 0.7)
            );
            if (blob) {
              const region = await detectPieceRegion(blob, controller.signal);
              if (cancelled) return;
              setPieceRegion(region);

              // Measure the detected crop for quality + appearance
              let bbox: NormalizedBBox | undefined;
              let sharpness: number | undefined;
              let signature: number[] | undefined;
              if (region.found && region.bbox) {
                bbox = region.bbox;
                const crop = bboxToCrop(bbox, frameCanvas.width, frameCanvas.height);
                const aScale = Math.min(1, ANALYSIS_MAX_DIM / Math.max(crop.sw, crop.sh));
                analysisCanvas.width = Math.max(1, Math.round(crop.sw * aScale));
                analysisCanvas.height = Math.max(1, Math.round(crop.sh * aScale));
                const actx = analysisCanvas.getContext('2d', { willReadFrequently: true });
                actx?.drawImage(
                  frameCanvas,
                  crop.sx,
                  crop.sy,
                  crop.sw,
                  crop.sh,
                  0,
                  0,
                  analysisCanvas.width,
                  analysisCanvas.height
                );
                const pixels = actx?.getImageData(
                  0,
                  0,
                  analysisCanvas.width,
                  analysisCanvas.height
                );
                if (pixels) {
                  sharpness = computeSharpness(pixels);
                  signature = computeSignature(pixels);
                }
              }

              const result = tracker.update({
                timestamp: Date.now(),
                found: Boolean(bbox),
                bbox,
                sharpness,
                signature,
              });

              for (const event of result.events) {
                if (event.type === 'committed') commitSnapshot(event.trackId);
                if (event.type === 'discarded') snapshots.delete(event.trackId);
              }

              if (result.snapshotRequested && result.activeTrackId && bbox) {
                const crop = bboxToCrop(bbox, frameCanvas.width, frameCanvas.height);
                let snapshot = snapshots.get(result.activeTrackId);
                if (!snapshot) {
                  snapshot = document.createElement('canvas');
                  snapshots.set(result.activeTrackId, snapshot);
                }
                const sScale = Math.min(1, SNAPSHOT_MAX_DIM / Math.max(crop.sw, crop.sh));
                snapshot.width = Math.max(1, Math.round(crop.sw * sScale));
                snapshot.height = Math.max(1, Math.round(crop.sh * sScale));
                snapshot
                  .getContext('2d')
                  ?.drawImage(
                    frameCanvas,
                    crop.sx,
                    crop.sy,
                    crop.sw,
                    crop.sh,
                    0,
                    0,
                    snapshot.width,
                    snapshot.height
                  );
              }

              setTracking(result.activeTrackId !== null);
            }
          } catch (err) {
            // Cleanup aborted the in-flight request; stop silently
            if (controller.signal.aborted) return;
            // Auth expired: stop polling instead of hammering the endpoint
            if (err instanceof ApiError && err.status === 401) return;
            // Otherwise keep the camera usable and try again next tick
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
      // Unmounting mid-track: the best frame is dropped rather than committed
      // (committing from a torn-down camera view surprises more than it helps)
      cancelled = true;
      controller.abort();
      setPieceRegion(null);
      setTracking(false);
    };
  }, [isReady, videoRef]);

  // Manual shutter: enqueue the current frame directly and reset the track so
  // the same piece doesn't get auto-committed a second time
  const handleManualCapture = async () => {
    const blob = await capture();
    if (blob) {
      abandonTrackRef.current();
      commitBlob(blob);
    }
  };

  if (error) {
    return (
      <div
        className={cn('flex flex-col items-center justify-center gap-4 p-8', className)}
        data-testid="live-capture-fallback"
      >
        <AlertCircle className="text-destructive h-12 w-12" />
        <p className="text-destructive text-center">{error}</p>
        <p className="text-muted-foreground text-center text-sm">
          You can upload piece images instead — each upload is added to the queue.
        </p>
        <div className="flex flex-col gap-2">
          <FileUpload onFileSelect={commitBlob}>
            <Button className="gap-2">
              <Upload className="h-4 w-4" />
              Upload Piece Image
            </Button>
          </FileUpload>
          <Button variant="secondary" onClick={() => void start()}>
            Retry Camera
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('relative flex flex-col', className)} data-testid="live-capture">
      <div className="relative flex-1 overflow-hidden rounded-lg bg-black">
        <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" />

        {/* Live piece detection outline; slice mirrors the video's object-cover mapping */}
        {isReady && dimensions && (
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
                  'rounded-full px-3 py-1 text-xs font-medium whitespace-nowrap text-white',
                  flash ? 'bg-primary/90' : tracking ? 'bg-green-600/80' : 'bg-black/50'
                )}
              >
                {flash
                  ? 'Piece captured!'
                  : tracking
                    ? 'Tracking piece — set it aside when done'
                    : 'Hold a piece up to the camera'}
              </span>
            </div>
          </>
        )}

        {/* Commit confirmation flash */}
        {flash && (
          <div className="animate-capture-flash pointer-events-none absolute inset-0 bg-green-400/40" />
        )}

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <Loader2 className="h-8 w-8 animate-spin text-white" />
          </div>
        )}
      </div>

      {/* Controls: manual shutter as fallback, plus file upload */}
      <div className="flex items-center justify-center gap-4 p-3">
        <Button
          size="lg"
          variant="secondary"
          className="h-12 w-12 rounded-full"
          onClick={() => void handleManualCapture()}
          disabled={!isReady || isLoading}
          title="Capture manually"
        >
          <Camera className="h-5 w-5" />
        </Button>
        <FileUpload onFileSelect={commitBlob}>
          <Button variant="outline" size="icon" className="h-10 w-10" title="Upload piece image">
            <Upload className="h-4 w-4" />
          </Button>
        </FileUpload>
      </div>
    </div>
  );
}
