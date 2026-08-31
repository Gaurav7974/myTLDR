'use client';

import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { KeywordDefinition } from '@/types/paper';
import { BookOpen, MessageSquare, ArrowUpRight, X } from 'lucide-react';
import katex from 'katex';

interface InteractiveHighlightedTextProps {
  text: string;
  keywords?: KeywordDefinition[];
  onAskAI?: (prompt: string) => void;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
  className?: string;
  defaultHighlightColor?: 'purple' | 'emerald' | 'amber' | 'blue' | 'rose';
}

interface TermMatch {
  keyword: KeywordDefinition;
  startIndex: number;
  endIndex: number;
  matchedText: string;
  color: 'purple' | 'emerald' | 'amber' | 'blue' | 'rose';
}

export const InteractiveHighlightedText: React.FC<InteractiveHighlightedTextProps> = ({
  text,
  keywords = [],
  onAskAI,
  onJumpToPage,
  className = '',
}) => {
  const [activePopover, setActivePopover] = useState<{
    keyword: KeywordDefinition;
    rect: { top: number; bottom: number; left: number; right: number; width: number; height: number };
    color: string;
    isPinned?: boolean;
  } | null>(null);

  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const activePopoverRef = useRef(activePopover);

  useEffect(() => {
    activePopoverRef.current = activePopover;
  }, [activePopover]);

  // Close popover when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (
        popoverRef.current &&
        !popoverRef.current.contains(target) &&
        !target.closest('.highlight-keyword-span')
      ) {
        setActivePopover(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!text || keywords.length === 0) {
    return <span className={className}>{text}</span>;
  }

  // Assign distinct pastel colors based on category or term
  const getColorForKeyword = (kw: KeywordDefinition): 'purple' | 'emerald' | 'amber' | 'blue' | 'rose' => {
    switch (kw.category) {
      case 'mechanism':
        return 'purple';
      case 'architecture':
        return 'emerald';
      case 'core':
        return 'amber';
      case 'math':
        return 'blue';
      case 'optimization':
        return 'rose';
      default:
        return 'purple';
    }
  };

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

  // Find all non-overlapping matches in the text
  const matches: TermMatch[] = [];

  // Sort keywords by length descending so longer phrases match first
  const sortedKeywords = [...keywords].sort(
    (a, b) => b.term.length - a.term.length
  );

  for (const kw of sortedKeywords) {
    const escaped = kw.term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escaped}\\b`, 'gi');
    let match;

    while ((match = pattern.exec(text)) !== null) {
      const startIndex = match.index;
      const endIndex = startIndex + match[0].length;

      const overlaps = matches.some(
        (m) =>
          (startIndex >= m.startIndex && startIndex < m.endIndex) ||
          (endIndex > m.startIndex && endIndex <= m.endIndex)
      );

      if (!overlaps) {
        matches.push({
          keyword: kw,
          startIndex,
          endIndex,
          matchedText: match[0],
          color: getColorForKeyword(kw),
        });
      }
    }
  }

  matches.sort((a, b) => a.startIndex - b.startIndex);

  if (matches.length === 0) {
    return <span className={className}>{text}</span>;
  }

  // Build segments
  const segments: React.ReactNode[] = [];
  let currentIndex = 0;

  const handleMouseEnter = (
    kw: KeywordDefinition,
    color: string,
    e: React.MouseEvent<HTMLSpanElement>
  ) => {
    if (activePopoverRef.current?.isPinned) return;
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    const domRect = e.currentTarget.getBoundingClientRect();
    const rect = {
      top: domRect.top,
      bottom: domRect.bottom,
      left: domRect.left,
      right: domRect.right,
      width: domRect.width,
      height: domRect.height,
    };
    setActivePopover({ keyword: kw, rect, color, isPinned: false });
  };

  const handleMouseLeave = () => {
    if (activePopoverRef.current?.isPinned) return;
    hoverTimerRef.current = setTimeout(() => {
      if (!activePopoverRef.current?.isPinned) {
        setActivePopover(null);
      }
    }, 300);
  };

  const getHighlightStyle = (color: string) => {
    switch (color) {
      case 'purple':
        return 'bg-[#E9D5FF]/90 text-[#4C1D95] hover:bg-[#DDD6FE] border-b-2 border-[#A855F7]/60';
      case 'emerald':
        return 'bg-[#A7F3D0]/90 text-[#064E3B] hover:bg-[#6EE7B7] border-b-2 border-[#10B981]/60';
      case 'amber':
        return 'bg-[#FEF08A]/90 text-[#713F12] hover:bg-[#FDE047] border-b-2 border-[#F59E0B]/60';
      case 'blue':
        return 'bg-[#BAE6FD]/90 text-[#0C4A6E] hover:bg-[#7DD3FC] border-b-2 border-[#0EA5E9]/60';
      case 'rose':
        return 'bg-[#FECDD3]/90 text-[#881337] hover:bg-[#FDA4AF] border-b-2 border-[#F43F5E]/60';
      default:
        return 'bg-[#E9D5FF]/90 text-[#4C1D95] hover:bg-[#DDD6FE] border-b-2 border-[#A855F7]/60';
    }
  };

  matches.forEach((m, idx) => {
    if (m.startIndex > currentIndex) {
      segments.push(
        <React.Fragment key={`plain-${currentIndex}`}>
          {text.substring(currentIndex, m.startIndex)}
        </React.Fragment>
      );
    }

    segments.push(
      <span
        key={`match-${idx}-${m.startIndex}`}
        onMouseEnter={(e) => handleMouseEnter(m.keyword, m.color, e)}
        onMouseLeave={handleMouseLeave}
        onClick={(e) => {
          e.stopPropagation();
          const domRect = e.currentTarget.getBoundingClientRect();
          const rect = {
            top: domRect.top,
            bottom: domRect.bottom,
            left: domRect.left,
            right: domRect.right,
            width: domRect.width,
            height: domRect.height,
          };
          setActivePopover({ keyword: m.keyword, rect, color: m.color, isPinned: true });
        }}
        className={`highlight-keyword-span inline-block px-1.5 py-0.2 mx-0.5 rounded-[3px] font-medium transition-all duration-150 cursor-pointer select-text ${getHighlightStyle(
          m.color
        )}`}
      >
        {m.matchedText}
      </span>
    );

    currentIndex = m.endIndex;
  });

  if (currentIndex < text.length) {
    segments.push(
      <React.Fragment key={`plain-end`}>
        {text.substring(currentIndex)}
      </React.Fragment>
    );
  }

  const renderFormulaHtml = (formula?: string) => {
    if (!formula) return null;
    try {
      return katex.renderToString(formula, {
        displayMode: true,
        throwOnError: false,
      });
    } catch {
      return null;
    }
  };

  return (
    <span className={`relative inline ${className}`}>
      {segments}

      {/* Sleek, Compact Floating Popover Card via Portal to document.body */}
      {activePopover &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            ref={popoverRef}
            onMouseEnter={() => {
              if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
            }}
            onMouseLeave={() => {
              if (!activePopover.isPinned) {
                hoverTimerRef.current = setTimeout(() => {
                  setActivePopover(null);
                }, 250);
              }
            }}
            className="fixed z-[99999] animate-fadeIn pointer-events-auto"
            style={{
              top: `${Math.min(
                window.innerHeight - 300,
                Math.max(12, activePopover.rect.bottom + 6)
              )}px`,
              left: `${Math.min(
                window.innerWidth - 340,
                Math.max(16, activePopover.rect.left - 16)
              )}px`,
            }}
          >
            <div className="w-72 sm:w-80 p-3 rounded-lg bg-white/95 backdrop-blur-md border border-zinc-200/90 shadow-xl shadow-zinc-950/8 text-zinc-900 space-y-2">
              
              {/* Header: Term & Category */}
              <div className="flex items-center justify-between border-b border-zinc-100 pb-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="font-bold text-[13px] text-zinc-950 tracking-tight truncate">
                    {activePopover.keyword.term.replace(/^#/, '')}
                  </span>
                  <span className={`text-[9px] uppercase font-mono px-1.5 py-0.5 rounded-full border font-semibold shrink-0 ${getCategoryBadge(activePopover.keyword.category)}`}>
                    {activePopover.keyword.category}
                  </span>
                </div>

                <button
                  onClick={() => setActivePopover(null)}
                  className="p-0.5 rounded text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors shrink-0 ml-1"
                  title="Close"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Core Intuition / Clear Explanation */}
              <p className="text-[12px] text-zinc-700 leading-relaxed font-normal">
                {activePopover.keyword.simpleExplanation || activePopover.keyword.definition}
              </p>

              {/* Formula (KaTeX Rendered) */}
              {activePopover.keyword.formula && (
                <div className="p-2 rounded-md bg-zinc-50/90 border border-zinc-200/70 overflow-x-auto text-zinc-900 text-center">
                  {(() => {
                    const html = renderFormulaHtml(activePopover.keyword.formula);
                    return html ? (
                      <div
                        className="text-[12px] leading-normal"
                        dangerouslySetInnerHTML={{ __html: html }}
                      />
                    ) : (
                      <code className="text-[10.5px] font-mono text-zinc-800">{activePopover.keyword.formula}</code>
                    );
                  })()}
                </div>
              )}

              {/* Footer / Actions */}
              <div className="pt-1.5 border-t border-zinc-100 flex items-center justify-between gap-2">
                {activePopover.keyword.relatedSection ? (
                  <button
                    onClick={() => {
                      setActivePopover(null);
                      const pageMatch = activePopover.keyword.relatedSection?.match(/Page\s*(\d+)/i) || 
                                       activePopover.keyword.relatedSection?.match(/Section\s*(\d+)/i);
                      const targetPage = pageMatch ? parseInt(pageMatch[1], 10) : 3;
                      onJumpToPage?.(targetPage, activePopover.keyword.relatedSection);
                    }}
                    className="text-[10.5px] font-mono text-zinc-500 hover:text-zinc-900 transition-colors flex items-center gap-1 truncate text-left"
                    title="Jump to Section in PDF"
                  >
                    <BookOpen className="w-3 h-3 text-zinc-400 shrink-0" />
                    <span className="truncate">{activePopover.keyword.relatedSection}</span>
                  </button>
                ) : (
                  <span />
                )}

                <button
                  onClick={() => {
                    const term = activePopover.keyword.term.replace(/^#/, '');
                    const prompt = `Explain "${term}" in detail with its core intuition and mathematical formulation.`;
                    setActivePopover(null);
                    onAskAI?.(prompt);
                  }}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-zinc-950 hover:bg-zinc-800 text-white text-[11px] font-medium shadow-2xs transition-all active:scale-95 shrink-0 ml-auto"
                >
                  <MessageSquare className="w-3 h-3" />
                  <span>Ask AI</span>
                  <ArrowUpRight className="w-3 h-3 opacity-70" />
                </button>
              </div>

            </div>
          </div>,
          document.body
        )}
    </span>
  );
};
