import 'fake-indexeddb/auto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { IDBFactory } from 'fake-indexeddb';

// savePuzzle generates a thumbnail via canvas/createImageBitmap, which jsdom
// does not implement. Stub the image helpers so the tests exercise the real
// IndexedDB logic (the point of this module) rather than canvas rendering.
vi.mock('./image-utils', () => ({
  compressImage: (blob: Blob) => Promise.resolve(blob),
  blobToDataUrl: () => Promise.resolve('data:image/jpeg;base64,dGh1bWI='),
}));

import {
  deletePuzzle,
  getPuzzleBlob,
  listPuzzles,
  renamePuzzle,
  savePuzzle,
} from './puzzle-library';

const blob = (marker: string) => new Blob([marker], { type: 'image/jpeg' });

describe('puzzle-library', () => {
  beforeEach(() => {
    // Fresh, isolated IndexedDB per test.
    indexedDB = new IDBFactory();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('saves a puzzle and returns metadata with a thumbnail', async () => {
    const meta = await savePuzzle(blob('a'), 'Puzzle 1');
    expect(meta.name).toBe('Puzzle 1');
    expect(meta.id).toBeTruthy();
    expect(meta.thumbnail).toContain('data:image/jpeg');
    expect(typeof meta.createdAt).toBe('number');
  });

  it('lists saved puzzles newest first', async () => {
    const now = vi.spyOn(Date, 'now');
    now.mockReturnValue(1000);
    const first = await savePuzzle(blob('a'), 'First');
    now.mockReturnValue(2000);
    const second = await savePuzzle(blob('b'), 'Second');

    const list = await listPuzzles();
    expect(list.map((p) => p.id)).toEqual([second.id, first.id]);
    // Metadata list carries no image blob.
    expect(list[0]).not.toHaveProperty('blob');
  });

  it('retrieves the stored image for a known id', async () => {
    const meta = await savePuzzle(blob('hello'), 'P');
    // The image lives in a separate store keyed by the same id; a known id
    // round-trips a value, an unknown one (below) yields null.
    expect(await getPuzzleBlob(meta.id)).not.toBeNull();
  });

  it('returns null for an unknown blob id', async () => {
    expect(await getPuzzleBlob('does-not-exist')).toBeNull();
  });

  it('renames a saved puzzle', async () => {
    const meta = await savePuzzle(blob('a'), 'Old');
    await renamePuzzle(meta.id, 'New');
    const list = await listPuzzles();
    expect(list.find((p) => p.id === meta.id)?.name).toBe('New');
  });

  it('no-ops when renaming an unknown id', async () => {
    await savePuzzle(blob('a'), 'Keep');
    await expect(renamePuzzle('missing', 'X')).resolves.toBeUndefined();
    const list = await listPuzzles();
    expect(list.map((p) => p.name)).toEqual(['Keep']);
  });

  it('deletes both metadata and image', async () => {
    const meta = await savePuzzle(blob('a'), 'P');
    await deletePuzzle(meta.id);
    expect(await listPuzzles()).toEqual([]);
    expect(await getPuzzleBlob(meta.id)).toBeNull();
  });
});
