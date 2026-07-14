'use client';

import Link from 'next/link';
import { Camera, FlaskConical, Info, LogIn, ScanLine, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ThemeToggle } from '@/components/theme-toggle';
import { useAuthStore } from '@/stores/auth-store';

export default function HomePage() {
  const { user, isLoading } = useAuthStore();

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-6">
      {/* Header with theme toggle and auth */}
      <div className="absolute top-4 right-4 flex items-center gap-2">
        {!isLoading && (
          <>
            {user ? (
              <Link href="/profile">
                <Button variant="ghost" size="sm" className="gap-2">
                  {user.picture ? (
                    <img
                      src={user.picture}
                      alt={user.name || 'User profile picture'}
                      className="h-6 w-6 rounded-full"
                    />
                  ) : (
                    <User className="h-4 w-4" />
                  )}
                  {user.name.split(' ')[0]}
                </Button>
              </Link>
            ) : (
              <Link href="/login">
                <Button variant="ghost" size="sm" className="gap-2">
                  <LogIn className="h-4 w-4" />
                  Sign In
                </Button>
              </Link>
            )}
          </>
        )}
        <ThemeToggle />
      </div>

      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-primary text-3xl font-bold">Pussel</CardTitle>
          <CardDescription>AI-powered puzzle piece position detection</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Button asChild className="w-full gap-2" size="lg">
            <Link href="/play">
              <Camera className="h-5 w-5" />
              New Puzzle
            </Link>
          </Button>

          <Button asChild variant="secondary" className="w-full gap-2" size="lg">
            <Link href="/real">
              <ScanLine className="h-5 w-5" />
              Solve Real Puzzle
            </Link>
          </Button>

          <Button asChild variant="secondary" className="w-full gap-2" size="lg">
            <Link href="/test-mode">
              <FlaskConical className="h-5 w-5" />
              Test Mode
            </Link>
          </Button>

          <Button asChild variant="outline" className="w-full gap-2" size="lg">
            <Link href="/about">
              <Info className="h-5 w-5" />
              About
            </Link>
          </Button>

          <p className="text-muted-foreground mt-4 text-center text-sm">
            Capture your puzzle and pieces to find where each piece belongs.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
