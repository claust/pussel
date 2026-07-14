/**
 * Client-side library of previously captured real puzzles.
 *
 * Persists the trimmed puzzle image on the device (IndexedDB) so that, in real
 * mode, a user can re-select a puzzle they already photographed instead of
 * re-taking the photo. Blobs live in a separate object store from the light
 * metadata so listing the gallery never has to pull full-resolution images
 * into memory.
 *
 * Storage is per-device/per-browser. The saved image is re-uploaded to the
 * backend on reuse to obtain a fresh puzzle_id, so a persisted entry keeps
 * working across backend restarts and redeploys.
 */
import { blobToDataUrl, compressImage } from '@/lib/image-utils';

const DB_NAME = 'pussel';
const DB_VERSION = 1;
const META_STORE = 'puzzle-meta';
const IMAGE_STORE = 'puzzle-images';

const THUMBNAIL_MAX_WIDTH = 320;
const THUMBNAIL_QUALITY = 0.6;

/** Lightweight record used to render the gallery (no full image). */
export interface SavedPuzzleMeta {
  id: string;
  name: string;
  thumbnail: string; // small data URL for the gallery card
  createdAt: number; // epoch ms
}

interface ImageRecord {
  id: string;
  blob: Blob;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(IMAGE_STORE)) {
        db.createObjectStore(IMAGE_STORE, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function promisifyTx(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

function promisifyRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Persist a trimmed puzzle image and return its gallery metadata.
 *
 * @param blob - The trimmed puzzle image to store (as uploaded to the backend).
 * @param name - Display name for the gallery card.
 * @returns The saved metadata, including a generated thumbnail.
 */
export async function savePuzzle(blob: Blob, name: string): Promise<SavedPuzzleMeta> {
  const thumbnailBlob = await compressImage(blob, THUMBNAIL_MAX_WIDTH, THUMBNAIL_QUALITY);
  const thumbnail = await blobToDataUrl(thumbnailBlob);
  const meta: SavedPuzzleMeta = {
    id: crypto.randomUUID(),
    name,
    thumbnail,
    createdAt: Date.now(),
  };

  const db = await openDb();
  try {
    const tx = db.transaction([META_STORE, IMAGE_STORE], 'readwrite');
    tx.objectStore(META_STORE).put(meta);
    tx.objectStore(IMAGE_STORE).put({ id: meta.id, blob } satisfies ImageRecord);
    await promisifyTx(tx);
  } finally {
    db.close();
  }

  return meta;
}

/** List saved puzzles, newest first. Reads metadata only (no image blobs). */
export async function listPuzzles(): Promise<SavedPuzzleMeta[]> {
  const db = await openDb();
  try {
    const tx = db.transaction(META_STORE, 'readonly');
    const all = await promisifyRequest(
      tx.objectStore(META_STORE).getAll() as IDBRequest<SavedPuzzleMeta[]>
    );
    return all.sort((a, b) => b.createdAt - a.createdAt);
  } finally {
    db.close();
  }
}

/** Fetch the full-resolution image blob for a saved puzzle, or null if missing. */
export async function getPuzzleBlob(id: string): Promise<Blob | null> {
  const db = await openDb();
  try {
    const tx = db.transaction(IMAGE_STORE, 'readonly');
    const record = await promisifyRequest(
      tx.objectStore(IMAGE_STORE).get(id) as IDBRequest<ImageRecord | undefined>
    );
    return record?.blob ?? null;
  } finally {
    db.close();
  }
}

/** Delete a saved puzzle (both metadata and image). */
export async function deletePuzzle(id: string): Promise<void> {
  const db = await openDb();
  try {
    const tx = db.transaction([META_STORE, IMAGE_STORE], 'readwrite');
    tx.objectStore(META_STORE).delete(id);
    tx.objectStore(IMAGE_STORE).delete(id);
    await promisifyTx(tx);
  } finally {
    db.close();
  }
}

/** Rename a saved puzzle. No-op if the id is unknown. */
export async function renamePuzzle(id: string, name: string): Promise<void> {
  const db = await openDb();
  try {
    const tx = db.transaction(META_STORE, 'readwrite');
    const store = tx.objectStore(META_STORE);
    const existing = await promisifyRequest(
      store.get(id) as IDBRequest<SavedPuzzleMeta | undefined>
    );
    if (existing) {
      store.put({ ...existing, name });
    }
    await promisifyTx(tx);
  } finally {
    db.close();
  }
}
