import { create } from 'zustand';
import type { Piece } from '@/types';

export type CaptureStatus = 'queued' | 'predicting' | 'done' | 'error';

export interface CaptureEntry {
  id: string;
  /** Object URL of the best raw camera crop for this piece */
  imageUrl: string;
  blob: Blob;
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
      entries: state.entries.map((e) =>
        e.id === id ? { ...e, status: 'done' as const, piece, error: undefined } : e
      ),
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
