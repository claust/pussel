'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Grid2X2, Grid3X3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { usePuzzleStore } from '@/stores/puzzle-store';
import type { GridSize } from '@/types';

export default function TestModePage() {
  const router = useRouter();
  const { setGridSize } = usePuzzleStore();

  const handleSelectGrid = (size: GridSize) => {
    setGridSize(size);
    router.push('/test-mode/select');
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-4 border-b p-4">
        <Link href="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <h1 className="text-lg font-semibold">Test Mode</h1>
      </header>

      {/* Content */}
      <main className="flex flex-1 flex-col items-center justify-center gap-6 p-6">
        <p className="text-muted-foreground text-center">Select the grid size for testing</p>

        <div className="flex w-full max-w-md flex-col gap-4">
          <Card
            className="cursor-pointer transition-shadow hover:shadow-lg"
            onClick={() => handleSelectGrid('2x2')}
          >
            <CardHeader className="flex-row items-center gap-4 space-y-0">
              <div className="bg-primary/10 rounded-lg p-3">
                <Grid2X2 className="text-primary h-8 w-8" />
              </div>
              <div>
                <CardTitle>2×2 Grid</CardTitle>
                <CardDescription>4 pieces total</CardDescription>
              </div>
            </CardHeader>
          </Card>

          <Card
            className="cursor-pointer transition-shadow hover:shadow-lg"
            onClick={() => handleSelectGrid('3x3')}
          >
            <CardHeader className="flex-row items-center gap-4 space-y-0">
              <div className="bg-secondary/30 rounded-lg p-3">
                <Grid3X3 className="text-secondary-foreground h-8 w-8" />
              </div>
              <div>
                <CardTitle>3×3 Grid</CardTitle>
                <CardDescription>9 pieces total</CardDescription>
              </div>
            </CardHeader>
          </Card>
        </div>
      </main>
    </div>
  );
}
