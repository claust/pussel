import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { useCaptureQueueStore } from './capture-queue-store';

describe('CaptureQueueStore', () => {
  // jsdom doesn't implement the Blob URL APIs; spy on them (rather than
  // replacing the global URL constructor wholesale) so revoke calls in the
  // store can be observed without leaking a stubbed URL into other tests.
  const urlRecord = URL as unknown as Record<string, unknown>;
  const polyfilled: string[] = [];

  beforeAll(() => {
    // Add the missing methods so vi.spyOn has something to wrap, remembering
    // which ones we added so they can be removed again in afterAll.
    if (typeof URL.createObjectURL !== 'function') {
      urlRecord.createObjectURL = () => '';
      polyfilled.push('createObjectURL');
    }
    if (typeof URL.revokeObjectURL !== 'function') {
      urlRecord.revokeObjectURL = () => {};
      polyfilled.push('revokeObjectURL');
    }
  });

  afterAll(() => {
    // Drop any methods we polyfilled so they don't leak into other test files
    for (const name of polyfilled) {
      delete urlRecord[name];
    }
  });

  let revokeObjectURL: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url');
    revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    useCaptureQueueStore.setState({ entries: [], isProcessing: false });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const makeBlob = () => new Blob(['data'], { type: 'image/png' });

  it('should have initial state', () => {
    const state = useCaptureQueueStore.getState();
    expect(state.entries).toEqual([]);
    expect(state.isProcessing).toBe(false);
  });

  it('should enqueue an entry as queued', () => {
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '1', blob: makeBlob(), imageUrl: 'blob:1', capturedAt: 123 });

    const { entries } = useCaptureQueueStore.getState();
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      id: '1',
      imageUrl: 'blob:1',
      capturedAt: 123,
      status: 'queued',
    });
  });

  it('should remove an entry and revoke its object URL', () => {
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '1', blob: makeBlob(), imageUrl: 'blob:1', capturedAt: 1 });
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '2', blob: makeBlob(), imageUrl: 'blob:2', capturedAt: 2 });

    useCaptureQueueStore.getState().remove('1');

    const { entries } = useCaptureQueueStore.getState();
    expect(entries).toHaveLength(1);
    expect(entries[0].id).toBe('2');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:1');
    expect(revokeObjectURL).toHaveBeenCalledTimes(1);
  });

  it('should be a no-op when removing an id that does not exist', () => {
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '1', blob: makeBlob(), imageUrl: 'blob:1', capturedAt: 1 });

    useCaptureQueueStore.getState().remove('missing');

    expect(useCaptureQueueStore.getState().entries).toHaveLength(1);
    expect(revokeObjectURL).not.toHaveBeenCalled();
  });

  it('retry should only affect entries in the error status', () => {
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '1', blob: makeBlob(), imageUrl: 'blob:1', capturedAt: 1 });
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '2', blob: makeBlob(), imageUrl: 'blob:2', capturedAt: 2 });

    useCaptureQueueStore.getState().setError('1', 'boom');
    useCaptureQueueStore.getState().setStatus('2', 'predicting');

    // Retrying the queued/predicting entry should have no effect
    useCaptureQueueStore.getState().retry('2');
    expect(useCaptureQueueStore.getState().entries.find((e) => e.id === '2')?.status).toBe(
      'predicting'
    );

    // Retrying the error entry resets it to queued and clears the error
    useCaptureQueueStore.getState().retry('1');
    const entry1 = useCaptureQueueStore.getState().entries.find((e) => e.id === '1');
    expect(entry1?.status).toBe('queued');
    expect(entry1?.error).toBeUndefined();
  });

  it('clear should revoke all object URLs and reset state', () => {
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '1', blob: makeBlob(), imageUrl: 'blob:1', capturedAt: 1 });
    useCaptureQueueStore
      .getState()
      .enqueue({ id: '2', blob: makeBlob(), imageUrl: 'blob:2', capturedAt: 2 });
    useCaptureQueueStore.getState().setProcessing(true);

    useCaptureQueueStore.getState().clear();

    expect(revokeObjectURL).toHaveBeenCalledWith('blob:1');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:2');
    expect(revokeObjectURL).toHaveBeenCalledTimes(2);

    const state = useCaptureQueueStore.getState();
    expect(state.entries).toEqual([]);
    expect(state.isProcessing).toBe(false);
  });
});
