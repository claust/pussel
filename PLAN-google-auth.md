# Google Authentication Implementation Plan

## Overview

Implement Google OAuth authentication across the Pussel application with:
- **Backend**: FastAPI with JWT-based session management
- **Frontend**: Next.js with Auth.js (NextAuth v5) for Google OAuth
- **Route Guards**: Middleware-based protection for authenticated routes
- **Profile Page**: Display user information from Google account

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Frontend      │         │   Auth.js       │         │   Google        │
│   (Next.js)     │◄───────►│   (NextAuth)    │◄───────►│   OAuth 2.0     │
└────────┬────────┘         └─────────────────┘         └─────────────────┘
         │
         │ JWT Token (in Authorization header)
         ▼
┌─────────────────┐
│   Backend       │
│   (FastAPI)     │
│   - Validates   │
│     Google JWT  │
│   - Issues app  │
│     JWT tokens  │
└─────────────────┘
```

## Implementation Steps

### Phase 1: Backend Authentication Setup

#### 1.1 Add Dependencies
Add to `backend/requirements.txt`:
```
python-jose[cryptography]>=3.3.0
google-auth>=2.27.0
```

#### 1.2 Create Auth Configuration
Create `backend/app/auth/config.py`:
- Add settings for JWT secret, algorithm, token expiry
- Add Google OAuth client ID for token verification

#### 1.3 Create Auth Models
Create `backend/app/models/user_model.py`:
- `User` model with id, email, name, picture, created_at
- `TokenPayload` for JWT claims
- `TokenResponse` for login endpoint response

#### 1.4 Create Auth Service
Create `backend/app/auth/service.py`:
- `verify_google_token()`: Verify Google ID token
- `create_access_token()`: Generate app JWT
- `decode_token()`: Decode and validate app JWT

#### 1.5 Create Auth Dependencies
Create `backend/app/auth/dependencies.py`:
- `get_current_user()`: FastAPI dependency for extracting user from JWT
- `get_optional_user()`: Same but returns None if not authenticated

#### 1.6 Create Auth Endpoints
Add to `backend/app/main.py` or create `backend/app/routers/auth.py`:
- `POST /api/v1/auth/google`: Exchange Google token for app JWT
- `GET /api/v1/auth/me`: Get current user profile

#### 1.7 Protect Existing Endpoints
Update `backend/app/main.py`:
- Add `get_current_user` dependency to puzzle endpoints
- Associate puzzles with users (optional enhancement)

### Phase 2: Frontend Authentication Setup

#### 2.1 Add Dependencies
```bash
cd frontend
bun add next-auth@beta @auth/core
```

#### 2.2 Configure Auth.js
Create `frontend/src/lib/auth.ts`:
```typescript
import NextAuth from "next-auth"
import Google from "next-auth/providers/google"

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [Google],
  callbacks: {
    async jwt({ token, account }) {
      if (account) {
        token.accessToken = account.access_token
        token.idToken = account.id_token
      }
      return token
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken
      session.idToken = token.idToken
      return session
    }
  }
})
```

#### 2.3 Create Auth Route Handler
Create `frontend/src/app/api/auth/[...nextauth]/route.ts`:
```typescript
import { handlers } from "@/lib/auth"
export const { GET, POST } = handlers
```

#### 2.4 Create Auth Store
Create `frontend/src/stores/auth-store.ts`:
- Store user state, loading state, backend token
- Methods: login, logout, refreshSession

#### 2.5 Create Auth Provider Component
Create `frontend/src/components/auth-provider.tsx`:
- Wrap SessionProvider from next-auth/react
- Initialize auth store on mount
- Exchange Google token for backend JWT

#### 2.6 Update Layout
Update `frontend/src/app/layout.tsx`:
- Wrap with AuthProvider

### Phase 3: Route Guards

#### 3.1 Create Middleware
Create `frontend/src/middleware.ts`:
```typescript
import { auth } from "@/lib/auth"

export default auth((req) => {
  const isLoggedIn = !!req.auth
  const isProtectedRoute = req.nextUrl.pathname.startsWith('/puzzle') ||
                          req.nextUrl.pathname.startsWith('/profile')

  if (isProtectedRoute && !isLoggedIn) {
    return Response.redirect(new URL('/login', req.nextUrl))
  }
})

export const config = {
  matcher: ['/puzzle/:path*', '/profile/:path*']
}
```

#### 3.2 Create Login Page
Create `frontend/src/app/login/page.tsx`:
- Google Sign-In button
- Redirect to original destination after login

### Phase 4: Profile Page

#### 4.1 Create Profile Page
Create `frontend/src/app/profile/page.tsx`:
- Display user avatar, name, email
- Show account creation date
- Logout button

#### 4.2 Add Navigation
Update navigation to include:
- Profile link (when authenticated)
- Login button (when not authenticated)
- User avatar in header

### Phase 5: API Client Updates

#### 5.1 Update API Client
Update `frontend/src/lib/api.ts`:
- Add Authorization header with JWT token
- Handle 401 responses with redirect to login

### Phase 6: Testing & Quality

#### 6.1 Backend Tests
Add `backend/tests/test_auth.py`:
- Test Google token verification (mocked)
- Test JWT creation and validation
- Test protected endpoints

#### 6.2 Frontend Tests
Add auth-related tests:
- Test login flow
- Test protected route redirects
- Test profile page rendering

#### 6.3 Environment Variables
Add to `.env.example`:
```
# Backend
JWT_SECRET=your-secret-key
GOOGLE_CLIENT_ID=your-google-client-id

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
AUTH_SECRET=your-nextauth-secret
```

## File Changes Summary

### New Files
- `backend/app/auth/__init__.py`
- `backend/app/auth/config.py`
- `backend/app/auth/service.py`
- `backend/app/auth/dependencies.py`
- `backend/app/models/user_model.py`
- `backend/tests/test_auth.py`
- `frontend/src/lib/auth.ts`
- `frontend/src/app/api/auth/[...nextauth]/route.ts`
- `frontend/src/stores/auth-store.ts`
- `frontend/src/components/auth-provider.tsx`
- `frontend/src/middleware.ts`
- `frontend/src/app/login/page.tsx`
- `frontend/src/app/profile/page.tsx`

### Modified Files
- `backend/requirements.txt` - Add auth dependencies
- `backend/app/config.py` - Add auth settings
- `backend/app/main.py` - Add auth routes and protect endpoints
- `frontend/package.json` - Add next-auth
- `frontend/src/app/layout.tsx` - Add AuthProvider
- `frontend/src/lib/api.ts` - Add auth headers
- `frontend/src/types/index.ts` - Add User type

## Security Considerations

1. **JWT Secret**: Use a strong, random secret (min 32 chars)
2. **Token Expiry**: Access tokens expire in 1 hour
3. **HTTPS**: Required in production for secure cookie handling
4. **CORS**: Restrict to frontend origin in production
5. **Token Storage**: Store JWT in httpOnly cookie or secure storage
6. **Google Token Verification**: Verify audience and issuer claims

## Environment Setup for Development

1. Create Google OAuth credentials at console.cloud.google.com
2. Add `http://localhost:3000` to authorized JavaScript origins
3. Add `http://localhost:3000/api/auth/callback/google` to redirect URIs
4. Copy Client ID and Secret to `.env` files
