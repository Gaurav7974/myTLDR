'use client';

import React, { useMemo } from 'react';
import { KeywordDefinition } from '@/types/paper';
import { X, BookOpen, MessageSquare, ArrowUpRight } from 'lucide-react';
import katex from 'katex';

interface KeywordPopoverProps {
  keyword: KeywordDefinition;
  onClose: () => void;
  onAskAI?: (prompt: string) => void;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
}

export const KeywordPopover: React.FC<KeywordPopoverProps> = ({
  keyword,
  onClose,
  onAskAI,
  onJumpToPage,
}) => {
  const getCategoryBadge = (cat: string) => {
    switch (cat) {
      case 'architecture':
        return 'bg-emerald-50 text-emerald-800 border-emerald-200/80';
      case 'mechanism':
        return 'bg-purple-50 text-purple-800 border-purple-200/80';
      case 'core':
        return 'bg-amber-50 text-amber-800 border-amber-200/80';
      case 'math':
        return 'bg-sky-50 text-sky-800 border-sky-200/80';
      default:
        return 'bg-zinc-100 text-zinc-800 border-zinc-200';
    }
  };

  const renderedFormula = useMemo(() => {
    if (!keyword.formula) return null;
    try {
      return katex.renderToString(keyword.formula, {
        displayMode: true,
        throwOnError: false,
      });
    } catch {
      return null;
    }
  }, [keyword.formula]);

  const cleanTerm = keyword.term.replace(/^#/, '');

  return (
    <div className="relative z-30 animate-fadeIn inline-block text-left">
      <div className="w-72 sm:w-80 p-3 rounded-lg bg-white/95 backdrop-blur-md border border-zinc-200/90 shadow-xl shadow-zinc-950/8 text-zinc-900 space-y-2">
        
        {/* Header: Clean Term & Category */}
        <div className="flex items-center justify-between border-b border-zinc-100 pb-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="font-bold text-[13px] text-zinc-950 tracking-tight truncate">
              {cleanTerm}
            </span>
            <span className={`text-[9px] uppercase font-mono px-1.5 py-0.5 rounded-full border font-semibold shrink-0 ${getCategoryBadge(keyword.category)}`}>
              {keyword.category}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-0.5 rounded text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors shrink-0 ml-1"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Core Intuition / Clear Explanation */}
        <p className="text-[12px] text-zinc-700 leading-relaxed font-normal">
          {keyword.simpleExplanation || keyword.definition}
        </p>

        {/* Mathematical Formula (KaTeX Rendered) */}
        {keyword.formula && (
          <div className="p-2 rounded-md bg-zinc-50/90 border border-zinc-200/70 overflow-x-auto text-zinc-900 text-center">
            {renderedFormula ? (
              <div
                className="text-[12px] leading-normal"
                dangerouslySetInnerHTML={{ __html: renderedFormula }}
              />
            ) : (
              <code className="text-[10.5px] font-mono text-zinc-800">{keyword.formula}</code>
            )}
          </div>
        )}

        {/* Footer: Section Reference & Ask AI */}
        <div className="pt-1.5 border-t border-zinc-100 flex items-center justify-between gap-2">
          {keyword.relatedSection ? (
            <button
              onClick={() => {
                const pageMatch = keyword.relatedSection?.match(/Page\s*(\d+)/i) || 
                                 keyword.relatedSection?.match(/Section\s*(\d+)/i);
                const targetPage = pageMatch ? parseInt(pageMatch[1], 10) : 3;
                onJumpToPage?.(targetPage, keyword.relatedSection);
              }}
              className="text-[10.5px] font-mono text-zinc-500 hover:text-zinc-900 transition-colors flex items-center gap-1 truncate text-left"
              title="Jump to Section in PDF"
            >
              <BookOpen className="w-3 h-3 text-zinc-400 shrink-0" />
              <span className="truncate">{keyword.relatedSection}</span>
            </button>
          ) : (
            <span />
          )}

          <button
            onClick={() => onAskAI?.(`Explain "${cleanTerm}" in detail with its core intuition and mathematical formulation.`)}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-zinc-950 hover:bg-zinc-800 text-white text-[11px] font-medium shadow-2xs transition-all active:scale-95 shrink-0 ml-auto"
          >
            <MessageSquare className="w-3 h-3" />
            <span>Ask AI</span>
            <ArrowUpRight className="w-3 h-3 opacity-70" />
          </button>
        </div>

      </div>
    </div>
  );
};
