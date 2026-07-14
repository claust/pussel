'use client';

import { useEffect, useRef } from 'react';
import { processPiece } from '@/lib/api';
import { useCaptureQueueStore } from '@/stores/capture-queue-store';
import { usePuzzleStore } from '@/stores/puzzle-store';

/**
 * Drains the capture queue serially against the backend.
 *
 * Whenever an entry is 'queued' and nothing is in flight, sends it to
 * `processPiece` (background removal + CNN prediction). Successful results
 * are stored on the entry and appended to the puzzle store so they appear on
 * the puzzle overlay. One request at a time: rembg is the slow step and the
 * backend shouldn't be hammered with concurrent segmentations.
 */
export function usePredictionWorker(puzzleId: string | undefined): void {
  // Subscriptions only re-trigger the drain; all decisions read live state
  // below so re-renders and strict-mode double-invocation can't double-send.
  const entries = useCaptureQueueStore((s) => s.entries);
  const isProcessing = useCaptureQueueStore((s) => s.isProcessing);
  const controllerRef = useRef<AbortController | null>(null);

  // Cancel any in-flight prediction when the puzzle changes or the component
  // unmounts. Keyed on puzzleId only, so an ordinary queue change (a new
  // capture arriving) never aborts a request that is already mid-flight.
  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, [puzzleId]);

  useEffect(() => {
    if (!puzzleId) return;
    const store = useCaptureQueueStore.getState();
    if (store.isProcessing) return;
    const next = store.entries.find((e) => e.status === 'queued');
    // A queued entry always carries its blob; the guard also narrows the
    // optional type for processPiece below.
    if (!next || !next.blob) return;
    const blob = next.blob;

    const controller = new AbortController();
    controllerRef.current = controller;
    store.setProcessing(true);
    store.setStatus(next.id, 'predicting');

    void processPiece(puzzleId, blob, controller.signal)
      .then((piece) => {
        // The queue may have been cleared (new puzzle / page left) meanwhile;
        // in that case the result belongs to nothing and is dropped.
        const live = useCaptureQueueStore.getState();
        if (!live.entries.some((e) => e.id === next.id)) return;
        // Stamp the entry id so the overlay piece can be removed by id later.
        const stored = { ...piece, id: next.id };
        live.setResult(next.id, stored);
        usePuzzleStore.getState().addPiece(stored);
      })
      .catch((err: unknown) => {
        // Aborted because the puzzle changed / the page was left: the entry is
        // being discarded, so don't surface an error for it.
        if (controller.signal.aborted) return;
        const live = useCaptureQueueStore.getState();
        if (!live.entries.some((e) => e.id === next.id)) return;
        live.setError(next.id, err instanceof Error ? err.message : 'Prediction failed');
      })
      .finally(() => {
        if (controllerRef.current === controller) controllerRef.current = null;
        // Always release the lock so the queue keeps draining (or retries)
        useCaptureQueueStore.getState().setProcessing(false);
      });
  }, [puzzleId, entries, isProcessing]);
}
