'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

export interface UseCameraOptions {
  facingMode?: 'user' | 'environment';
  width?: number;
  height?: number;
}

export interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isReady: boolean;
  isLoading: boolean;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  capture: () => Promise<Blob | null>;
}

export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const { facingMode = 'environment', width = 1920, height = 1080 } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Check if mediaDevices is available
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera not supported in this browser');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: width },
          height: { ideal: height },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsReady(true);
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Camera access denied. Please allow camera permissions.';
      setError(message);
      setIsReady(false);
    } finally {
      setIsLoading(false);
    }
  }, [facingMode, width, height]);

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsReady(false);
  }, []);

  const capture = useCallback(async (): Promise<Blob | null> => {
    if (!videoRef.current || !isReady) {
      return null;
    }

    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    ctx.drawImage(video, 0, 0);

    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        'image/jpeg',
        0.9
      );
    });
  }, [isReady]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    videoRef,
    isReady,
    isLoading,
    error,
    start,
    stop,
    capture,
  };
}
