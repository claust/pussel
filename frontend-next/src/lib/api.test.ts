import { describe, it, expect } from 'vitest';
import { API_BASE } from './api';

describe('API Client', () => {
  it('should have correct API base URL', () => {
    expect(API_BASE).toBe('http://localhost:8000');
  });
});
