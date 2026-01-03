'use client';

import { SessionProvider, useSession } from 'next-auth/react';
import { useEffect, type ReactNode } from 'react';
import { useAuthStore } from '@/stores/auth-store';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function AuthSync() {
  const { data: session, status } = useSession();
  const { setUser, setBackendToken, setLoading, setError, reset } = useAuthStore();

  useEffect(() => {
    const syncAuth = async () => {
      if (status === 'loading') {
        setLoading(true);
        return;
      }

      if (status === 'unauthenticated' || !session) {
        reset();
        return;
      }

      // Session is authenticated
      setLoading(true);
      setError(null);

      try {
        // Exchange Google ID token for backend JWT
        const idToken = (session as { idToken?: string }).idToken;
        if (!idToken) {
          throw new Error('No ID token available');
        }

        const response = await fetch(`${API_URL}/api/v1/auth/google`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ id_token: idToken }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to authenticate with backend');
        }

        const data = await response.json();

        setUser({
          id: data.user.id,
          email: data.user.email,
          name: data.user.name,
          picture: data.user.picture,
        });
        setBackendToken(data.access_token);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Authentication failed');
        reset();
      } finally {
        setLoading(false);
      }
    };

    syncAuth();
  }, [session, status, setUser, setBackendToken, setLoading, setError, reset]);

  return null;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  return (
    <SessionProvider>
      <AuthSync />
      {children}
    </SessionProvider>
  );
}
