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
        <div className="border-primary h-8 w-8 animate-spin rounded-full border-4 border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="bg-background min-h-screen p-4">
      <div className="mx-auto max-w-2xl">
        <Button variant="ghost" onClick={() => router.back()} className="mb-6">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>

        <div className="bg-card rounded-lg border p-8 shadow-lg">
          <div className="flex flex-col items-center space-y-6">
            <div className="relative">
              {user?.picture ? (
                <img
                  src={user.picture}
                  alt={user?.name || 'User profile picture'}
                  className="border-primary h-24 w-24 rounded-full border-4"
                />
              ) : (
                <div className="border-primary bg-muted flex h-24 w-24 items-center justify-center rounded-full border-4">
                  <User className="text-muted-foreground h-12 w-12" />
                </div>
              )}
            </div>

            <div className="text-center">
              <h1 className="text-foreground text-2xl font-bold">{user?.name || 'User'}</h1>
              <p className="text-muted-foreground">{user?.email || 'No email'}</p>
            </div>

            <div className="w-full space-y-4 border-t pt-6">
              <div className="flex justify-between">
                <span className="text-muted-foreground">User ID</span>
                <span className="text-foreground font-mono text-sm">{user?.id || 'N/A'}</span>
              </div>
            </div>

            <Button
              onClick={() => void handleSignOut()}
              variant="destructive"
              className="w-full"
              size="lg"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
