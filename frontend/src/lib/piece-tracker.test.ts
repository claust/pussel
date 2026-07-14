import { describe, it, expect } from 'vitest';
import {
  PieceTracker,
  bboxIoU,
  type TrackerObservation,
  type NormalizedBBox,
} from './piece-tracker';

const BOX: NormalizedBBox = { x: 0.4, y: 0.4, width: 0.2, height: 0.2 };
// Signatures: piece A concentrated in bin 0, piece B in bin 1 (max distance)
const SIG_A = [1, 0, 0, 0];
const SIG_B = [0, 1, 0, 0];

function obs(overrides: Partial<TrackerObservation> & { timestamp: number }): TrackerObservation {
  return { found: true, bbox: BOX, signature: SIG_A, sharpness: 100, ...overrides };
}

function makeTracker() {
  return new PieceTracker({
    gapCommitMs: 1000,
    minFrames: 3,
    iouBreakThreshold: 0.05,
    signatureBreakThreshold: 0.4,
    breakFrames: 2,
    signatureAlpha: 0.3,
  });
}

describe('bboxIoU', () => {
  it('is 1 for identical boxes and 0 for disjoint boxes', () => {
    expect(bboxIoU(BOX, BOX)).toBeCloseTo(1);
    expect(bboxIoU(BOX, { x: 0.8, y: 0.8, width: 0.1, height: 0.1 })).toBe(0);
  });
});

describe('PieceTracker', () => {
  it('starts a track and requests a snapshot on first detection', () => {
    const tracker = makeTracker();
    const result = tracker.update(obs({ timestamp: 0 }));
    expect(result.events).toEqual([{ type: 'started', trackId: 'track-1' }]);
    expect(result.snapshotRequested).toBe(true);
    expect(result.activeTrackId).toBe('track-1');
  });

  it('requests a snapshot only when frame quality improves', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0, sharpness: 100 }));
    const worse = tracker.update(obs({ timestamp: 400, sharpness: 50 }));
    expect(worse.snapshotRequested).toBe(false);
    const better = tracker.update(obs({ timestamp: 800, sharpness: 200 }));
    expect(better.snapshotRequested).toBe(true);
  });

  it('commits the track after a sustained detection gap', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    tracker.update(obs({ timestamp: 400 }));
    tracker.update(obs({ timestamp: 800 }));
    // Gap shorter than gapCommitMs: track stays alive
    const early = tracker.update({ timestamp: 1400, found: false });
    expect(early.events).toEqual([]);
    expect(early.activeTrackId).toBe('track-1');
    // Sustained gap: committed
    const late = tracker.update({ timestamp: 1900, found: false });
    expect(late.events).toEqual([
      { type: 'committed', trackId: 'track-1', frames: 3, bestScore: expect.any(Number) },
    ]);
    expect(late.activeTrackId).toBeNull();
  });

  it('discards tracks with too few frames instead of committing', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    const result = tracker.update({ timestamp: 1200, found: false });
    expect(result.events).toEqual([
      { type: 'discarded', trackId: 'track-1', reason: 'too-few-frames' },
    ]);
  });

  it('cuts to a new track on a sustained signature break (piece swap without gap)', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    tracker.update(obs({ timestamp: 400 }));
    tracker.update(obs({ timestamp: 800 }));

    // First divergent frame: hysteresis holds the old track
    const first = tracker.update(obs({ timestamp: 1200, signature: SIG_B }));
    expect(first.events).toEqual([]);
    expect(first.activeTrackId).toBe('track-1');

    // Second consecutive divergent frame: old track commits, new one starts
    const second = tracker.update(obs({ timestamp: 1600, signature: SIG_B }));
    expect(second.events).toEqual([
      { type: 'committed', trackId: 'track-1', frames: 3, bestScore: expect.any(Number) },
      { type: 'started', trackId: 'track-2' },
    ]);
    expect(second.snapshotRequested).toBe(true);
    expect(second.activeTrackId).toBe('track-2');
  });

  it('does not split a track on a single-frame appearance flash', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    tracker.update(obs({ timestamp: 400 }));
    tracker.update(obs({ timestamp: 800, signature: SIG_B })); // flash
    const recovered = tracker.update(obs({ timestamp: 1200, signature: SIG_A }));
    expect(recovered.events).toEqual([]);
    expect(recovered.activeTrackId).toBe('track-1');
  });

  it('cuts to a new track when the bbox jumps without overlap', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    tracker.update(obs({ timestamp: 400 }));
    tracker.update(obs({ timestamp: 800 }));
    const far: NormalizedBBox = { x: 0.05, y: 0.05, width: 0.1, height: 0.1 };
    tracker.update(obs({ timestamp: 1200, bbox: far }));
    const second = tracker.update(obs({ timestamp: 1600, bbox: far }));
    expect(second.events[0]).toEqual({
      type: 'committed',
      trackId: 'track-1',
      frames: 3,
      bestScore: expect.any(Number),
    });
    expect(second.activeTrackId).toBe('track-2');
  });

  it('flush commits the active track immediately', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    tracker.update(obs({ timestamp: 400 }));
    tracker.update(obs({ timestamp: 800 }));
    expect(tracker.flush()).toEqual([
      { type: 'committed', trackId: 'track-1', frames: 3, bestScore: expect.any(Number) },
    ]);
    expect(tracker.flush()).toEqual([]);
  });

  it('abandon drops the active track without events', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0 }));
    tracker.abandon();
    expect(tracker.flush()).toEqual([]);
    // Next detection starts a fresh track
    const next = tracker.update(obs({ timestamp: 400 }));
    expect(next.events).toEqual([{ type: 'started', trackId: 'track-2' }]);
  });

  it('a larger, equally sharp view of the piece scores better', () => {
    const tracker = makeTracker();
    tracker.update(obs({ timestamp: 0, bbox: { x: 0.45, y: 0.45, width: 0.1, height: 0.1 } }));
    const bigger = tracker.update(
      obs({ timestamp: 400, bbox: { x: 0.4, y: 0.4, width: 0.2, height: 0.2 } })
    );
    expect(bigger.snapshotRequested).toBe(true);
  });
});
