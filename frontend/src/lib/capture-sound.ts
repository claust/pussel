/**
 * Short confirmation blip played when a piece capture is auto-committed.
 * Synthesized with WebAudio so no asset is needed. Fails silently where
 * audio is unavailable or blocked (e.g. before any user gesture).
 */

let audioContext: AudioContext | null = null;

function getContext(): AudioContext | null {
  if (typeof window === 'undefined' || !('AudioContext' in window)) return null;
  audioContext ??= new AudioContext();
  return audioContext;
}

export function playCommitSound(): void {
  try {
    const ctx = getContext();
    if (!ctx) return;
    if (ctx.state === 'suspended') {
      void ctx.resume();
    }

    // Two quick ascending tones — a friendly "got it" chirp
    const tones: Array<{ freq: number; at: number }> = [
      { freq: 880, at: 0 },
      { freq: 1318.5, at: 0.09 },
    ];
    for (const { freq, at } of tones) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const start = ctx.currentTime + at;
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.18, start + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.001, start + 0.12);
      osc.connect(gain).connect(ctx.destination);
      osc.start(start);
      osc.stop(start + 0.14);
    }
  } catch {
    // Audio is a nicety; never let it break the capture flow
  }
}
