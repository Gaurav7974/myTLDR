'use client';

import React, { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import { DualPaneWorkspace } from '@/components/DualPaneWorkspace';
import { UploadPageView } from '@/components/UploadPageView';
import { TRANSFORMER_PAPER_DATA } from '@/lib/mockData';
import { PaperData } from '@/types/paper';

const PAPER_PARAM = 'paper';
const STORAGE_PREFIX = 'mytldr_paper_';
const SLUG_RE = /^[A-Za-z0-9.\-_/]{1,60}$/;

/** Stable, URL-safe identity for a decoded paper (never raw user input). */
function paperSlug(paper: PaperData): string | null {
  const raw =
    paper.metadata.arxivId || paper.metadata.contentHash || paper.metadata.id;
  if (!raw || typeof raw !== 'string') return null;
  const slug = raw.replace(/[^A-Za-z0-9.\-_/]/g, '').slice(0, 60);
  return slug || null;
}

function readCachedPaper(slug: string): PaperData | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_PREFIX + slug);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PaperData;
    if (paperSlug(parsed) !== slug) return null;
    return parsed;
  } catch {
    return null;
  }
}

function HomePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Deep-link restore: ?paper=<slug> reopens the decoded paper when this
  // browser still holds it in sessionStorage (refresh/back-button safe).
  const [currentPaper, setCurrentPaper] = useState<PaperData>(TRANSFORMER_PAPER_DATA);
  const [viewState, setViewState] = useState<'workspace' | 'upload'>('upload');

  useEffect(() => {
    try {
      const slug = searchParams.get(PAPER_PARAM);
      if (slug && SLUG_RE.test(slug)) {
        const cached = readCachedPaper(slug);
        if (cached) {
          setCurrentPaper(cached);
          setViewState('workspace');
        }
      }
    } catch {
      // storage unavailable — fall through to default
    }
  }, [searchParams]);

  const syncUrlToPaper = (paper: PaperData | null) => {
    const slug = paper ? paperSlug(paper) : null;
    try {
      if (paper && slug) {
        try {
          sessionStorage.setItem(STORAGE_PREFIX + slug, JSON.stringify(paper));
        } catch {
          // quota/full — URL still updates, restore just won't survive reload
        }
        router.replace(`/?${PAPER_PARAM}=${encodeURIComponent(slug)}`, { scroll: false });
      } else {
        router.replace('/', { scroll: false });
      }
    } catch {
      // navigation unavailable — state still works in-memory
    }
  };

  if (viewState === 'upload') {
    return (
      <UploadPageView
        onSelectPaper={(paper) => {
          setCurrentPaper(paper);
          setViewState('workspace');
          syncUrlToPaper(paper);
        }}
      />
    );
  }

  return (
    <main className="h-[calc(100vh/0.9)] flex flex-col overflow-hidden bg-[#FAFAFA] text-zinc-900 selection:bg-zinc-200">
      {/* Top Minimal Navigation */}
      <Navbar
        activePaperTitle={currentPaper.metadata.title}
        onOpenUpload={() => setViewState('upload')}
      />

      {/* Main Dual Pane Workspace */}
      <DualPaneWorkspace
        paper={currentPaper}
        onGoBack={() => {
          setViewState('upload');
          syncUrlToPaper(null);
        }}
      />
    </main>
  );
}

export default function HomePage() {
  // useSearchParams requires a Suspense boundary for prerendering.
  return (
    <Suspense fallback={null}>
      <HomePageInner />
    </Suspense>
  );
}
