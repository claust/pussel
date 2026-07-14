import { describe, it, expect } from 'vitest';
import {
  computeSharpness,
  computeSignature,
  signatureDistance,
  blendSignature,
  SIGNATURE_SIZE,
  type PixelData,
} from './frame-quality';

function makeImage(
  width: number,
  height: number,
  fill: (x: number, y: number) => [number, number, number]
): PixelData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = fill(x, y);
      const p = (y * width + x) * 4;
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = 255;
    }
  }
  return { data, width, height };
}

const solidGray = (w: number, h: number) => makeImage(w, h, () => [128, 128, 128]);
const checkerboard = (w: number, h: number) =>
  makeImage(w, h, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0]));
const solidRed = (w: number, h: number) => makeImage(w, h, () => [220, 30, 30]);
const solidBlue = (w: number, h: number) => makeImage(w, h, () => [30, 30, 220]);

describe('computeSharpness', () => {
  it('is zero for a flat image', () => {
    expect(computeSharpness(solidGray(16, 16))).toBe(0);
  });

  it('is higher for high-frequency content than for flat content', () => {
    const sharp = computeSharpness(checkerboard(16, 16));
    const flat = computeSharpness(solidGray(16, 16));
    expect(sharp).toBeGreaterThan(flat);
    expect(sharp).toBeGreaterThan(1000);
  });

  it('returns 0 for degenerate tiny images', () => {
    expect(computeSharpness(solidGray(2, 2))).toBe(0);
  });
});

describe('computeSignature', () => {
  it('has the documented size and sums to ~1', () => {
    const sig = computeSignature(solidRed(8, 8));
    expect(sig).toHaveLength(SIGNATURE_SIZE);
    const sum = sig.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it('is identical for identical images', () => {
    const a = computeSignature(solidRed(8, 8));
    const b = computeSignature(solidRed(8, 8));
    expect(signatureDistance(a, b)).toBeCloseTo(0, 10);
  });

  it('separates differently colored pieces', () => {
    const red = computeSignature(solidRed(8, 8));
    const blue = computeSignature(solidBlue(8, 8));
    expect(signatureDistance(red, blue)).toBeGreaterThan(0.8);
  });

  it('bins gray pixels separately from colored ones', () => {
    const gray = computeSignature(solidGray(8, 8));
    const red = computeSignature(solidRed(8, 8));
    expect(signatureDistance(gray, red)).toBeGreaterThan(0.8);
  });

  it('is rotation invariant', () => {
    // Half red / half blue image, horizontal vs vertical split (a 90° rotation)
    const horizontal = makeImage(8, 8, (_x, y) => (y < 4 ? [220, 30, 30] : [30, 30, 220]));
    const vertical = makeImage(8, 8, (x) => (x < 4 ? [220, 30, 30] : [30, 30, 220]));
    expect(signatureDistance(computeSignature(horizontal), computeSignature(vertical))).toBeCloseTo(
      0,
      10
    );
  });
});

describe('signatureDistance', () => {
  it('is 1 for disjoint distributions', () => {
    const a = new Array(SIGNATURE_SIZE).fill(0);
    const b = new Array(SIGNATURE_SIZE).fill(0);
    a[0] = 1;
    b[1] = 1;
    expect(signatureDistance(a, b)).toBe(1);
  });
});

describe('blendSignature', () => {
  it('moves the running signature toward the new one by alpha', () => {
    const running = [1, 0];
    const next = [0, 1];
    expect(blendSignature(running, next, 0.25)).toEqual([0.75, 0.25]);
  });
});
