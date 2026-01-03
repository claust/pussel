'use client';

import { signOut } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, LogOut, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuthStore } from '@/stores/auth-store';

export default function ProfilePage() {
  const router = useRouter();
  const { user, isLoading } = useAuthStore();

  const handleSignOut = async () => {
    await signOut({ callbackUrl: '/' });
  };

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="mx-auto max-w-2xl">
        <Button variant="ghost" onClick={() => router.back()} className="mb-6">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        <div className="rounded-lg border bg-card p-8 shadow-lg">
          <div className="flex flex-col items-center space-y-6">
            <div className="relative">
              {user?.picture ? (
                <img
                  src={user.picture}
                  alt={user.name}
                  className="h-24 w-24 rounded-full border-4 border-primary"
                />
              ) : (
                <div className="flex h-24 w-24 items-center justify-center rounded-full border-4 border-primary bg-muted">
                  <User className="h-12 w-12 text-muted-foreground" />
                </div>
              )}
            </div>

            <div className="text-center">
              <h1 className="text-2xl font-bold text-foreground">{user?.name || 'User'}</h1>
              <p className="text-muted-foreground">{user?.email || 'No email'}</p>
            </div>

            <div className="w-full space-y-4 border-t pt-6">
              <div className="flex justify-between">
                <span className="text-muted-foreground">User ID</span>
                <span className="font-mono text-sm text-foreground">{user?.id || 'N/A'}</span>
              </div>
            </div>

            <Button onClick={handleSignOut} variant="destructive" className="w-full" size="lg">
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
