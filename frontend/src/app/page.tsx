'use client';

import React, { useState } from 'react';
import { Navbar } from '@/components/Navbar';
import { DualPaneWorkspace } from '@/components/DualPaneWorkspace';
import { UploadPageView } from '@/components/UploadPageView';
import { TRANSFORMER_PAPER_DATA } from '@/lib/mockData';
import { PaperData } from '@/types/paper';

export default function HomePage() {
  const [currentPaper, setCurrentPaper] = useState<PaperData>(TRANSFORMER_PAPER_DATA);
  const [viewState, setViewState] = useState<'workspace' | 'upload'>('upload');

  if (viewState === 'upload') {
    return (
      <UploadPageView
        onSelectPaper={(paper) => {
          setCurrentPaper(paper);
          setViewState('workspace');
        }}
      />
    );
  }

  return (
    <main className="h-screen flex flex-col overflow-hidden bg-[#FAFAFA] text-zinc-900 selection:bg-zinc-200">
      {/* Top Minimal Navigation */}
      <Navbar
        activePaperTitle={currentPaper.metadata.title}
        onOpenUpload={() => setViewState('upload')}
      />

      {/* Main Dual Pane Workspace */}
      <DualPaneWorkspace
        paper={currentPaper}
        onGoBack={() => setViewState('upload')}
      />
    </main>
  );
}

