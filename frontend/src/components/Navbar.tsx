'use client';

import { FileText, Upload } from 'lucide-react';

interface NavbarProps {
  onOpenUpload: () => void;
  activePaperTitle?: string;
}

export const Navbar: React.FC<NavbarProps> = ({ onOpenUpload, activePaperTitle }) => {
  return (
    <header className="sticky top-0 z-50 w-full bg-[#FFFFFF]/90 backdrop-blur-md border-b border-zinc-200/80 px-6 py-2.5 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Brand Logo with Modern Geometric Research Emblem */}
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-zinc-950 text-white flex items-center justify-center shadow-xs">
            <svg className="w-4 h-4 text-zinc-100" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m12 3-8 4.5 8 4.5 8-4.5L12 3Z"/>
              <path d="m4 12 8 4.5 8-4.5"/>
              <path d="m4 16.5 8 4.5 8-4.5"/>
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-zinc-950 text-base tracking-tight">myTLDR</span>
              <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-zinc-100 text-zinc-700 font-semibold border border-zinc-200">
                Paper Decoder
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 font-medium">Understand any research paper</p>
          </div>
        </div>

        {/* Center: Current Active Paper Snippet */}
        {activePaperTitle && (
          <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-md bg-zinc-50 border border-zinc-200/80 max-w-md truncate shadow-2xs">
            <FileText className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span className="text-xs text-zinc-800 font-medium truncate">{activePaperTitle}</span>
          </div>
        )}

        {/* Right Action Bar - Minimal & Clean */}
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-zinc-900 hover:text-zinc-950 bg-zinc-100 hover:bg-zinc-200/80 rounded-md border border-zinc-200/90 shadow-2xs transition-all active:scale-95"
            title="Upload or select another paper"
          >
            <Upload className="w-3.5 h-3.5 text-zinc-600" />
            <span>Upload Paper</span>
          </button>
        </div>

      </div>
    </header>
  );
};
