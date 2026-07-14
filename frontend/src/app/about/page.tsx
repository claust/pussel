'use client';

import Link from 'next/link';
import { ArrowLeft, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

// lucide-react v1 dropped brand icons (including GitHub), so inline the mark.
function GithubIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={className}>
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

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
                <GithubIcon className="h-4 w-4" />
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
