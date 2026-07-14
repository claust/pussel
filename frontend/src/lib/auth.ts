import type { Account, Session } from 'next-auth';
import type { JWT } from 'next-auth/jwt';
import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';

declare module 'next-auth' {
  interface Session {
    accessToken?: string;
    idToken?: string;
    error?: 'RefreshTokenError';
    user: {
      id: string;
      email: string;
      name: string;
      image?: string | null;
    };
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    accessToken?: string;
    idToken?: string;
    refreshToken?: string;
    expiresAt?: number; // Unix seconds when the Google tokens expire
    error?: 'RefreshTokenError';
  }
}

// Validate required environment variables at startup
const googleClientId = process.env.GOOGLE_CLIENT_ID;
const googleClientSecret = process.env.GOOGLE_CLIENT_SECRET;

if (!googleClientId) {
  throw new Error(
    'GOOGLE_CLIENT_ID environment variable is not set. ' +
      'Please configure it in your .env.local file. ' +
      'Get your client ID from https://console.cloud.google.com/apis/credentials'
  );
}

if (!googleClientSecret) {
  throw new Error(
    'GOOGLE_CLIENT_SECRET environment variable is not set. ' +
      'Please configure it in your .env.local file. ' +
      'Get your client secret from https://console.cloud.google.com/apis/credentials'
  );
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Google({
      clientId: googleClientId,
      clientSecret: googleClientSecret,
      authorization: {
        params: {
          prompt: 'consent',
          access_type: 'offline',
          response_type: 'code',
        },
      },
    }),
  ],
  callbacks: {
    async jwt({ token, account }: { token: JWT; account?: Account | null }) {
      if (account) {
        // Initial sign-in: store the Google tokens and their expiry
        token.accessToken = account.access_token;
        token.idToken = account.id_token;
        token.refreshToken = account.refresh_token;
        token.expiresAt = account.expires_at;
        token.error = undefined;
        return token;
      }

      // Google ID tokens expire after ~1 hour; refresh shortly before expiry so the
      // backend exchange in AuthSync never receives an expired token.
      const expiresAt = token.expiresAt ?? 0;
      if (Date.now() < (expiresAt - 60) * 1000) {
        return token;
      }

      if (!token.refreshToken) {
        token.error = 'RefreshTokenError';
        return token;
      }

      try {
        const response = await fetch('https://oauth2.googleapis.com/token', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({
            client_id: googleClientId,
            client_secret: googleClientSecret,
            grant_type: 'refresh_token',
            refresh_token: token.refreshToken,
          }),
        });

        const refreshed = (await response.json()) as {
          access_token?: string;
          id_token?: string;
          expires_in?: number;
          refresh_token?: string;
          error?: string;
        };
        if (!response.ok || !refreshed.id_token) {
          throw new Error(refreshed.error || 'Failed to refresh Google token');
        }

        token.accessToken = refreshed.access_token;
        token.idToken = refreshed.id_token;
        token.expiresAt = Math.floor(Date.now() / 1000) + (refreshed.expires_in ?? 3600);
        // Google may or may not rotate the refresh token
        token.refreshToken = refreshed.refresh_token ?? token.refreshToken;
        token.error = undefined;
      } catch {
        token.error = 'RefreshTokenError';
      }
      return token;
    },
    async session({ session, token }: { session: Session; token: JWT }) {
      return {
        ...session,
        accessToken: token.accessToken,
        idToken: token.idToken,
        error: token.error,
      };
    },
  },
  pages: {
    signIn: '/login',
  },
});
