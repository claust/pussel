import type { GridSize } from '@/types';
import { GRID_DIMENSIONS } from '@/types';

/**
 * Crop a cell from a puzzle image based on grid position
 */
export async function cropCell(
  imageBlob: Blob,
  gridSize: GridSize,
  cellIndex: number,
  rotationDegrees: number = 0
): Promise<Blob> {
  const { dimension } = GRID_DIMENSIONS[gridSize];
  const img = await createImageBitmap(imageBlob);

  const cellWidth = img.width / dimension;
  const cellHeight = img.height / dimension;
  const row = Math.floor(cellIndex / dimension);
  const col = cellIndex % dimension;

  const canvas = document.createElement('canvas');
  canvas.width = cellWidth;
  canvas.height = cellHeight;
  const ctx = canvas.getContext('2d')!;

  // Apply rotation around center
  if (rotationDegrees !== 0) {
    ctx.translate(cellWidth / 2, cellHeight / 2);
    ctx.rotate((rotationDegrees * Math.PI) / 180);
    ctx.translate(-cellWidth / 2, -cellHeight / 2);
  }

  ctx.drawImage(
    img,
    col * cellWidth,
    row * cellHeight,
    cellWidth,
    cellHeight,
    0,
    0,
    cellWidth,
    cellHeight
  );

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create blob from canvas'));
        }
      },
      'image/jpeg',
      0.9
    );
  });
}

/**
 * Get a preview of a cell without rotation (for display in selector)
 */
export async function getCellPreview(
  imageBlob: Blob,
  gridSize: GridSize,
  cellIndex: number
): Promise<string> {
  const blob = await cropCell(imageBlob, gridSize, cellIndex, 0);
  return URL.createObjectURL(blob);
}

/**
 * Convert a File or Blob to a data URL
 */
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Convert a data URL to a Blob
 */
export async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl);
  return res.blob();
}

/**
 * Compress an image blob to a target size
 */
export async function compressImage(
  imageBlob: Blob,
  maxWidth: number = 1920,
  quality: number = 0.8
): Promise<Blob> {
  const img = await createImageBitmap(imageBlob);

  let width = img.width;
  let height = img.height;

  if (width > maxWidth) {
    height = (height * maxWidth) / width;
    width = maxWidth;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, width, height);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to compress image'));
        }
      },
      'image/jpeg',
      quality
    );
  });
}
