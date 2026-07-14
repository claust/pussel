/**
 * Frame quality and appearance measures for the live piece-capture pipeline.
 *
 * All functions operate on a structural `PixelData` (compatible with
 * `ImageData`) so they stay pure and unit-testable without a DOM canvas.
 */

export interface PixelData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

// Signature layout: 8 hue bins x 3 saturation bins + 4 value bins for
// low-saturation (gray-ish) pixels. Coarse on purpose — it only needs to
// tell "different piece" apart from "same piece, moved/rotated".
const HUE_BINS = 8;
const SAT_BINS = 3;
const GRAY_VALUE_BINS = 4;
// Pixels below this saturation carry no reliable hue; bin them by value instead
const GRAY_SAT_THRESHOLD = 0.15;

export const SIGNATURE_SIZE = HUE_BINS * SAT_BINS + GRAY_VALUE_BINS;

/**
 * Sharpness as the variance of a 4-neighbour Laplacian on the grayscale image.
 * Higher is sharper. Only comparable between frames of similar content/scale,
 * which is exactly how the tracker uses it (within one piece track).
 */
export function computeSharpness(image: PixelData): number {
  const { data, width, height } = image;
  if (width < 3 || height < 3) return 0;

  // Grayscale plane
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }

  let sum = 0;
  let sumSq = 0;
  const count = (width - 2) * (height - 2);
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      const lap = gray[i - width] + gray[i + width] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      sum += lap;
      sumSq += lap * lap;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

/**
 * Rotation-invariant appearance signature: an L1-normalized coarse HSV
 * histogram. Two crops of the same piece (moved, rotated, slightly blurred)
 * land close together; a different piece usually does not.
 */
export function computeSignature(image: PixelData): number[] {
  const { data } = image;
  const hist = new Array<number>(SIGNATURE_SIZE).fill(0);
  const pixelCount = data.length / 4;
  if (pixelCount === 0) return hist;

  for (let p = 0; p < data.length; p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const sat = max === 0 ? 0 : delta / max;

    if (sat < GRAY_SAT_THRESHOLD) {
      const vBin = Math.min(GRAY_VALUE_BINS - 1, Math.floor(max * GRAY_VALUE_BINS));
      hist[HUE_BINS * SAT_BINS + vBin] += 1;
      continue;
    }

    let hue: number;
    if (max === r) {
      hue = ((g - b) / delta) % 6;
    } else if (max === g) {
      hue = (b - r) / delta + 2;
    } else {
      hue = (r - g) / delta + 4;
    }
    hue = (hue + 6) % 6; // 0..6
    const hBin = Math.min(HUE_BINS - 1, Math.floor((hue / 6) * HUE_BINS));
    // Remaining saturation range mapped onto SAT_BINS
    const satNorm = (sat - GRAY_SAT_THRESHOLD) / (1 - GRAY_SAT_THRESHOLD);
    const sBin = Math.min(SAT_BINS - 1, Math.floor(satNorm * SAT_BINS));
    hist[hBin * SAT_BINS + sBin] += 1;
  }

  for (let i = 0; i < hist.length; i++) {
    hist[i] /= pixelCount;
  }
  return hist;
}

/**
 * Total-variation distance between two signatures, in [0, 1].
 * 0 = identical distributions, 1 = disjoint.
 */
export function signatureDistance(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / 2;
}

/**
 * Blend a new signature into a running (exponential moving average) signature.
 * Keeps the track's identity reference stable against per-frame noise.
 */
export function blendSignature(running: number[], next: number[], alpha: number): number[] {
  return running.map((v, i) => v * (1 - alpha) + (next[i] ?? 0) * alpha);
}
