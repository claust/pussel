'use client';

import { useEffect } from 'react';
import { Camera, Loader2, AlertCircle, Upload } from 'lucide-react';
import { useCamera } from '@/hooks/use-camera';
import { Button } from '@/components/ui/button';
import { FileUpload } from './file-upload';
import { cn } from '@/lib/utils';

interface CameraViewProps {
  onCapture: (blob: Blob) => void;
  onCancel: () => void;
  onOrientationChange?: (isLandscape: boolean) => void;
  overlayImage?: string | null;
  className?: string;
}

export function CameraView({
  onCapture,
  onCancel,
  onOrientationChange,
  overlayImage,
  className,
}: CameraViewProps) {
  const { videoRef, isReady, isLoading, error, dimensions, start, stop, capture } = useCamera();

  useEffect(() => {
    void start();
    return () => {
      stop();
    };
  }, [start, stop]);

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
