import { create } from 'zustand';
import type { Piece } from '@/types';

export type CaptureStatus = 'queued' | 'predicting' | 'done' | 'error';

export interface CaptureEntry {
  id: string;
  /** Object URL of the best raw camera crop; swapped for the cleaned image once done */
  imageUrl: string;
  /** Raw capture blob, needed until prediction runs; released once done */
  blob?: Blob;
  status: CaptureStatus;
  capturedAt: number;
  /** Prediction result (includes the background-removed image) once done */
  piece?: Piece;
  error?: string;
}

interface CaptureQueueState {
  entries: CaptureEntry[];
  /** True while a prediction request is in flight (queue drains serially) */
  isProcessing: boolean;

  enqueue: (entry: { id: string; blob: Blob; imageUrl: string; capturedAt: number }) => void;
  setStatus: (id: string, status: CaptureStatus) => void;
  setResult: (id: string, piece: Piece) => void;
  setError: (id: string, error: string) => void;
  retry: (id: string) => void;
  remove: (id: string) => void;
  setProcessing: (isProcessing: boolean) => void;
  clear: () => void;
}

export const useCaptureQueueStore = create<CaptureQueueState>((set, get) => ({
  entries: [],
  isProcessing: false,

  enqueue: ({ id, blob, imageUrl, capturedAt }) =>
    set((state) => ({
      entries: [...state.entries, { id, blob, imageUrl, capturedAt, status: 'queued' as const }],
    })),

  setStatus: (id, status) =>
    set((state) => ({
      entries: state.entries.map((e) => (e.id === id ? { ...e, status } : e)),
    })),

  setResult: (id, piece) =>
    set((state) => ({
      entries: state.entries.map((e) => {
        if (e.id !== id) return e;
        // Once the cleaned image is available the UI renders that instead of
        // the raw capture, so release the raw object URL and blob — otherwise
        // hundreds of captures would pin their blobs in memory for the session.
        if (piece.imageData) {
          URL.revokeObjectURL(e.imageUrl);
          return {
            ...e,
            status: 'done' as const,
            piece,
            error: undefined,
            imageUrl: piece.imageData,
            blob: undefined,
          };
        }
        // No cleaned image came back; keep the raw capture so the UI can show it.
        return { ...e, status: 'done' as const, piece, error: undefined };
      }),
    })),

  setError: (id, error) =>
    set((state) => ({
      entries: state.entries.map((e) =>
        e.id === id ? { ...e, status: 'error' as const, error } : e
      ),
    })),

  retry: (id) =>
    set((state) => ({
      entries: state.entries.map((e) =>
        e.id === id && e.status === 'error'
          ? { ...e, status: 'queued' as const, error: undefined }
          : e
      ),
    })),

  remove: (id) => {
    // Release the removed entry's object URL before dropping it
    const entry = get().entries.find((e) => e.id === id);
    if (entry) {
      URL.revokeObjectURL(entry.imageUrl);
    }
    set((state) => ({ entries: state.entries.filter((e) => e.id !== id) }));
  },

  setProcessing: (isProcessing) => set({ isProcessing }),

  clear: () => {
    // Release the object URLs owned by the queue before dropping the entries
    for (const entry of get().entries) {
      URL.revokeObjectURL(entry.imageUrl);
    }
    set({ entries: [], isProcessing: false });
  },
}));
