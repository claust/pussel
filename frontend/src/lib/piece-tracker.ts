/**
 * Piece track lifecycle for the live capture pipeline.
 *
 * Consumes one observation per detection tick (piece found or not, plus
 * quality/appearance measures) and decides when a physical piece's "track"
 * starts, when the current frame is its best capture so far, and when the
 * track ends and should be committed to the capture queue.
 *
 * Identity layers:
 *  - Temporal continuity: consecutive detections that overlap (bbox IoU) are
 *    the same piece; a detection gap ends the track.
 *  - Appearance: a running HSV-histogram signature per track. If the observed
 *    signature diverges for `breakFrames` consecutive ticks (hysteresis, so a
 *    hand or rotation flash doesn't split a track), the old track is
 *    committed and a new one starts — this catches quick piece swaps that
 *    never produce a detection gap.
 *
 * Pure logic: timestamps are injected, no DOM or timers, unit-testable.
 */

import { blendSignature, signatureDistance } from './frame-quality';

export interface NormalizedBBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TrackerObservation {
  timestamp: number; // ms
  found: boolean;
  bbox?: NormalizedBBox;
  signature?: number[];
  sharpness?: number;
}

export type TrackerEvent =
  | { type: 'started'; trackId: string }
  | { type: 'committed'; trackId: string; frames: number; bestScore: number }
  | { type: 'discarded'; trackId: string; reason: 'too-few-frames' };

export interface TrackerUpdateResult {
  events: TrackerEvent[];
  /** True when the current observation is the new best frame of the active track — snapshot it. */
  snapshotRequested: boolean;
  activeTrackId: string | null;
}

export interface TrackerConfig {
  /** Piece unseen for this long ends the track (ms). */
  gapCommitMs: number;
  /** Minimum observed frames for a track to be committed instead of discarded. */
  minFrames: number;
  /** BBox IoU below this counts toward an identity break. */
  iouBreakThreshold: number;
  /** Signature distance above this counts toward an identity break. */
  signatureBreakThreshold: number;
  /** Consecutive break ticks required before cutting to a new track. */
  breakFrames: number;
  /** EMA rate for the running track signature. */
  signatureAlpha: number;
}

export const DEFAULT_TRACKER_CONFIG: TrackerConfig = {
  gapCommitMs: 1000,
  minFrames: 3,
  iouBreakThreshold: 0.05,
  signatureBreakThreshold: 0.4,
  breakFrames: 2,
  signatureAlpha: 0.3,
};

export function bboxIoU(a: NormalizedBBox, b: NormalizedBBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Frame quality: sharpness weighted by how large the piece is in frame. */
function frameScore(obs: TrackerObservation): number {
  const area = obs.bbox ? obs.bbox.width * obs.bbox.height : 0;
  return (obs.sharpness ?? 0) * Math.sqrt(area);
}

interface ActiveTrack {
  id: string;
  frames: number;
  lastSeenAt: number;
  lastBBox: NormalizedBBox;
  signature: number[];
  bestScore: number;
  breakStreak: number;
}

export class PieceTracker {
  private config: TrackerConfig;
  private track: ActiveTrack | null = null;
  private nextId = 1;

  constructor(config: Partial<TrackerConfig> = {}) {
    this.config = { ...DEFAULT_TRACKER_CONFIG, ...config };
  }

  get activeTrackId(): string | null {
    return this.track?.id ?? null;
  }

  update(obs: TrackerObservation): TrackerUpdateResult {
    const events: TrackerEvent[] = [];
    let snapshotRequested = false;

    if (!obs.found || !obs.bbox) {
      // Piece not visible: end the track after a sustained gap
      if (this.track && obs.timestamp - this.track.lastSeenAt >= this.config.gapCommitMs) {
        events.push(this.finalize(this.track));
        this.track = null;
      }
      return { events, snapshotRequested: false, activeTrackId: this.activeTrackId };
    }

    if (!this.track) {
      this.track = this.startTrack(obs);
      events.push({ type: 'started', trackId: this.track.id });
      this.track.bestScore = frameScore(obs);
      snapshotRequested = true;
      return { events, snapshotRequested, activeTrackId: this.track.id };
    }

    // Identity check against the active track. The signature only contributes
    // once the track has actually learned one; comparing against an empty
    // running signature would always yield distance 0 and silently disable
    // appearance-based swap detection.
    const iou = bboxIoU(this.track.lastBBox, obs.bbox);
    const hasSignatures = this.track.signature.length > 0 && Boolean(obs.signature?.length);
    const sigDist = hasSignatures ? signatureDistance(this.track.signature, obs.signature!) : 0;
    const isBreak =
      iou < this.config.iouBreakThreshold || sigDist > this.config.signatureBreakThreshold;

    if (isBreak) {
      this.track.breakStreak += 1;
      if (this.track.breakStreak >= this.config.breakFrames) {
        // Sustained identity change: this is a different piece
        events.push(this.finalize(this.track));
        this.track = this.startTrack(obs);
        events.push({ type: 'started', trackId: this.track.id });
        this.track.bestScore = frameScore(obs);
        snapshotRequested = true;
      }
      // Within hysteresis: hold the track unchanged (don't pollute its
      // signature or best frame with what may be a different piece)
      return { events, snapshotRequested, activeTrackId: this.track.id };
    }

    // Same piece: fold the observation into the track
    this.track.breakStreak = 0;
    this.track.frames += 1;
    this.track.lastSeenAt = obs.timestamp;
    this.track.lastBBox = obs.bbox;
    if (obs.signature?.length) {
      // Adopt the first available signature outright; a track that started
      // before any signature was computed would otherwise stay empty forever
      // (blending into an empty array keeps it empty).
      this.track.signature =
        this.track.signature.length === 0
          ? [...obs.signature]
          : blendSignature(this.track.signature, obs.signature, this.config.signatureAlpha);
    }
    const score = frameScore(obs);
    if (score > this.track.bestScore) {
      this.track.bestScore = score;
      snapshotRequested = true;
    }

    return { events, snapshotRequested, activeTrackId: this.track.id };
  }

  /** End the active track immediately (e.g. camera stopping). */
  flush(): TrackerEvent[] {
    if (!this.track) return [];
    const event = this.finalize(this.track);
    this.track = null;
    return [event];
  }

  /** Drop the active track without committing (e.g. after a manual capture). */
  abandon(): void {
    this.track = null;
  }

  private startTrack(obs: TrackerObservation): ActiveTrack {
    return {
      id: `track-${this.nextId++}`,
      frames: 1,
      lastSeenAt: obs.timestamp,
      lastBBox: obs.bbox as NormalizedBBox,
      signature: obs.signature ? [...obs.signature] : [],
      bestScore: 0,
      breakStreak: 0,
    };
  }

  private finalize(track: ActiveTrack): TrackerEvent {
    if (track.frames < this.config.minFrames) {
      return { type: 'discarded', trackId: track.id, reason: 'too-few-frames' };
    }
    return {
      type: 'committed',
      trackId: track.id,
      frames: track.frames,
      bestScore: track.bestScore,
    };
  }
}
