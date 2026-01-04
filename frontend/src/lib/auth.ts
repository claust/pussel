import type { Account, Session } from 'next-auth';
import type { JWT } from 'next-auth/jwt';
import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';

declare module 'next-auth' {
  interface Session {
    accessToken?: string;
    idToken?: string;
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
        token.accessToken = account.access_token;
        token.idToken = account.id_token;
      }
      return token;
    },
    async session({ session, token }: { session: Session; token: JWT }) {
      return {
        ...session,
        accessToken: token.accessToken,
        idToken: token.idToken,
      };
    },
  },
  pages: {
    signIn: '/login',
  },
});
