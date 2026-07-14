'use client';

import { SessionProvider, signOut, useSession } from 'next-auth/react';
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
        // Token refresh failed server-side: the session can never authenticate again,
        // so clear it to make the sign-in flow reachable (proxy redirects
        // authenticated sessions away from /login).
        const sessionError = (session as { error?: string }).error;
        if (sessionError === 'RefreshTokenError') {
          reset();
          await signOut({ redirect: false });
          return;
        }

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

        if (response.status === 401) {
          // Stale or rejected Google token: drop the NextAuth session so the
          // user can sign in fresh instead of being stuck in a redirect loop.
          reset();
          await signOut({ redirect: false });
          return;
        }

        if (!response.ok) {
          let message = 'Failed to authenticate with backend';

          try {
            const errorBody = (await response.json()) as {
              detail?: string;
            };

            if (errorBody && typeof errorBody.detail === 'string') {
              message = errorBody.detail;
            }
          } catch {
            // Ignore JSON parsing errors and fall back to default message.
          }

          throw new Error(message);
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

    void syncAuth();
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
