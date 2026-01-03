'use client';

import Link from 'next/link';
import { Camera, FlaskConical, Info, LogIn, User } from 'lucide-react';
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
                    <img src={user.picture} alt={user.name} className="h-6 w-6 rounded-full" />
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
          <Link href="/play" className="w-full">
            <Button className="w-full gap-2" size="lg">
              <Camera className="h-5 w-5" />
              New Puzzle
            </Button>
          </Link>

          <Link href="/test-mode" className="w-full">
            <Button variant="secondary" className="w-full gap-2" size="lg">
              <FlaskConical className="h-5 w-5" />
              Test Mode
            </Button>
          </Link>

          <Link href="/about" className="w-full">
            <Button variant="outline" className="w-full gap-2" size="lg">
              <Info className="h-5 w-5" />
              About
            </Button>
          </Link>

          <p className="text-muted-foreground mt-4 text-center text-sm">
            Capture your puzzle and pieces to find where each piece belongs.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
