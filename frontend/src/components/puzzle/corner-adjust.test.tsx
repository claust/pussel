import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CornerAdjust } from './corner-adjust';
import type { QuadCorners } from '@/types';

const CORNERS: QuadCorners = {
  topLeft: { x: 0.1, y: 0.2 },
  topRight: { x: 0.9, y: 0.2 },
  bottomRight: { x: 0.9, y: 0.8 },
  bottomLeft: { x: 0.1, y: 0.8 },
};

function renderCornerAdjust(overrides: Partial<Parameters<typeof CornerAdjust>[0]> = {}) {
  const onApply = vi.fn();
  const onCancel = vi.fn();
  render(
    <CornerAdjust
      imageUrl="data:image/jpeg;base64,test"
      initialCorners={CORNERS}
      onApply={onApply}
      onCancel={onCancel}
      {...overrides}
    />
  );
  return { onApply, onCancel };
}

describe('CornerAdjust', () => {
  it('renders four corner handles at the initial positions', () => {
    renderCornerAdjust();

    const topLeft = screen.getByTestId('corner-handle-topLeft');
    expect(topLeft).toHaveStyle({ left: '10%', top: '20%' });

    const bottomRight = screen.getByTestId('corner-handle-bottomRight');
    expect(bottomRight).toHaveStyle({ left: '90%', top: '80%' });

    expect(screen.getByTestId('corner-handle-topRight')).toBeInTheDocument();
    expect(screen.getByTestId('corner-handle-bottomLeft')).toBeInTheDocument();
  });

  it('calls onApply with the current corners', () => {
    const { onApply } = renderCornerAdjust();

    fireEvent.click(screen.getByRole('button', { name: /apply/i }));

    expect(onApply).toHaveBeenCalledWith(CORNERS);
  });

  it('calls onCancel when cancelled', () => {
    const { onCancel } = renderCornerAdjust();

    fireEvent.click(screen.getByRole('button', { name: /cancel/i }));

    expect(onCancel).toHaveBeenCalled();
  });

  it('disables buttons while loading', () => {
    renderCornerAdjust({ isLoading: true });

    expect(screen.getByRole('button', { name: /apply/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /cancel/i })).toBeDisabled();
  });
});
