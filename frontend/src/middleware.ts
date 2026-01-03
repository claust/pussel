import type { NextRequest } from 'next/server';
import { auth } from '@/lib/auth';
import { NextResponse } from 'next/server';

interface AuthRequest extends NextRequest {
  auth: { user?: { email?: string | null } } | null;
}

export default auth((req: AuthRequest) => {
  const isLoggedIn = !!req.auth;
  const { pathname } = req.nextUrl;

  // Define protected routes
  const protectedRoutes = ['/puzzle', '/profile'];
  const isProtectedRoute = protectedRoutes.some((route) => pathname.startsWith(route));

  // Define public routes that should redirect authenticated users
  const authRoutes = ['/login'];
  const isAuthRoute = authRoutes.some((route) => pathname.startsWith(route));

  // Redirect unauthenticated users to login
  if (isProtectedRoute && !isLoggedIn) {
    const loginUrl = new URL('/login', req.nextUrl.origin);
    loginUrl.searchParams.set('callbackUrl', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Redirect authenticated users away from auth routes
  if (isAuthRoute && isLoggedIn) {
    return NextResponse.redirect(new URL('/', req.nextUrl.origin));
  }

  return NextResponse.next();
});

export const config = {
  matcher: ['/puzzle/:path*', '/profile/:path*', '/login'],
};
