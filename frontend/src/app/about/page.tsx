'use client';

import Link from 'next/link';
import { ArrowLeft, Github, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function AboutPage() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-4 border-b p-4">
        <Link href="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <h1 className="text-lg font-semibold">About</h1>
      </header>

      {/* Content */}
      <main className="flex-1 p-4">
        <div className="mx-auto max-w-md space-y-6">
          <Card>
            <CardHeader className="text-center">
              <CardTitle className="text-primary text-2xl">Pussel</CardTitle>
              <CardDescription>AI-Powered Puzzle Solver</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground text-sm">
                Pussel uses computer vision and deep learning to help you solve puzzles. Simply take
                a photo of your complete puzzle, then capture individual pieces to find where they
                belong.
              </p>

              <div className="space-y-2">
                <h3 className="font-medium">How it works:</h3>
                <ol className="text-muted-foreground list-inside list-decimal space-y-1 text-sm">
                  <li>Capture a photo of the complete puzzle</li>
                  <li>Take photos of individual puzzle pieces</li>
                  <li>AI predicts the position and rotation of each piece</li>
                  <li>See the results overlaid on your puzzle image</li>
                </ol>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium">Technology:</h3>
                <ul className="text-muted-foreground list-inside list-disc space-y-1 text-sm">
                  <li>PyTorch/PyTorch Lightning for ML training</li>
                  <li>FastAPI backend for image processing</li>
                  <li>Next.js + React frontend</li>
                  <li>CNN-based position and rotation prediction</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Version</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground text-sm">0.1.0 (Next.js)</p>
            </CardContent>
          </Card>

          <div className="flex justify-center">
            <Button variant="outline" className="gap-2" asChild>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                <Github className="h-4 w-4" />
                View on GitHub
                <ExternalLink className="h-3 w-3" />
              </a>
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
