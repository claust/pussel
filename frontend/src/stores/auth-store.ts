import { create } from 'zustand';
import type { User } from '@/types';

interface AuthState {
  user: User | null;
  backendToken: string | null;
  isLoading: boolean;
  error: string | null;
  setUser: (user: User | null) => void;
  setBackendToken: (token: string | null) => void;
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  backendToken: null,
  isLoading: false,
  error: null,
  setUser: (user) => set({ user }),
  setBackendToken: (backendToken) => set({ backendToken }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () =>
    set({
      user: null,
      backendToken: null,
      isLoading: false,
      error: null,
    }),
}));
