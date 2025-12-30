'use client';

import Link from 'next/link';
import Image from 'next/image';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { TEST_PUZZLES } from '@/lib/test-puzzles';
import { usePuzzleStore } from '@/stores/puzzle-store';

export default function PuzzleSelectionPage() {
  const { gridSize } = usePuzzleStore();

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-4 border-b p-4">
        <Link href="/test-mode">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <div>
          <h1 className="text-lg font-semibold">Select Puzzle</h1>
          <p className="text-muted-foreground text-sm">Grid: {gridSize}</p>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 p-4">
        <div className="grid grid-cols-2 gap-3">
          {TEST_PUZZLES.map((puzzle) => (
            <Link key={puzzle.id} href={`/test-mode/puzzle/${puzzle.id}`}>
              <Card className="cursor-pointer overflow-hidden transition-shadow hover:shadow-lg">
                <CardContent className="p-0">
                  <div className="bg-muted relative aspect-square">
                    <Image
                      src={puzzle.path}
                      alt={puzzle.name}
                      fill
                      className="object-cover"
                      sizes="(max-width: 768px) 50vw, 25vw"
                    />
                  </div>
                  <div className="p-2 text-center">
                    <p className="text-sm font-medium">{puzzle.name}</p>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
