'use client';

import { useCallback, useEffect, useState } from 'react';
import {
  deletePuzzle,
  listPuzzles,
  renamePuzzle,
  savePuzzle,
  type SavedPuzzleMeta,
} from '@/lib/puzzle-library';

interface UseSavedPuzzles {
  puzzles: SavedPuzzleMeta[];
  isReady: boolean; // initial load has completed (IndexedDB is async)
  save: (blob: Blob, name: string) => Promise<SavedPuzzleMeta>;
  remove: (id: string) => Promise<void>;
  rename: (id: string, name: string) => Promise<void>;
}

/**
 * React access to the on-device library of saved real puzzles.
 *
 * Loads the gallery on mount and keeps local state in sync as puzzles are
 * saved, renamed, or removed. IndexedDB is unavailable during SSR, so the list
 * stays empty until the effect runs on the client.
 */
export function useSavedPuzzles(): UseSavedPuzzles {
  const [puzzles, setPuzzles] = useState<SavedPuzzleMeta[]>([]);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listPuzzles()
      .then((list) => {
        if (!cancelled) setPuzzles(list);
      })
      .catch(() => {
        // A blocked/unavailable IndexedDB simply yields an empty gallery.
      })
      .finally(() => {
        if (!cancelled) setIsReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const save = useCallback(async (blob: Blob, name: string) => {
    const meta = await savePuzzle(blob, name);
    setPuzzles((prev) => [meta, ...prev]);
    return meta;
  }, []);

  const remove = useCallback(async (id: string) => {
    await deletePuzzle(id);
    setPuzzles((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const rename = useCallback(async (id: string, name: string) => {
    await renamePuzzle(id, name);
    setPuzzles((prev) => prev.map((p) => (p.id === id ? { ...p, name } : p)));
  }, []);

  return { puzzles, isReady, save, remove, rename };
}
