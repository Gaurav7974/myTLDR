'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { UploadPageView } from '@/components/UploadPageView';

export default function UploadPage() {
  const router = useRouter();

  const handleSelectPaper = () => {
    // Navigate back to workspace
    router.push('/');
  };

  return <UploadPageView onSelectPaper={handleSelectPaper} />;
}
