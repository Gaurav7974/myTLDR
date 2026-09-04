'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { PaperData } from '@/types/paper';
import { SummaryTab } from './tabs/SummaryTab';
import { Architecture2DTab } from './tabs/Architecture2DTab';
import { MathDeconstructedTab, MathFormula } from './tabs/MathDeconstructedTab';
import { CodeWalkthroughTab } from './tabs/CodeWalkthroughTab';
import { AIChatTab } from './tabs/AIChatTab';
import { TabErrorBoundary } from './TabErrorBoundary';
import {
  Lightbulb,
  Layers,
  Calculator,
  Code,
  MessageSquare,
  ExternalLink,
  BookOpen,
  Share2,
  Bookmark,
  ArrowLeft,
  ArrowUp,
  Search,
  Send,
  Check,
  Quote,
  Copy,
  FileText,
  List,
  CornerDownLeft,
  Sparkles
} from 'lucide-react';

// Tracks the Tailwind `lg` breakpoint (1024px). The draggable pane-split
// widths are desktop-only: inline `style` widths ignore breakpoints, so they
// must be omitted below `lg` or the panes stay squeezed at 50% on mobile.
function useDesktop(breakpoint = 1024) {
  const [isDesktop, setIsDesktop] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth >= breakpoint : true
  );
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${breakpoint}px)`);
    const sync = () => setIsDesktop(window.innerWidth >= breakpoint);
    const onChange = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    sync();
    // `resize` is a fallback for environments where matchMedia
    // change events are unreliable (older WebKit, CDP emulation).
    window.addEventListener('resize', sync);
    mq.addEventListener('change', onChange);
    return () => {
      window.removeEventListener('resize', sync);
      mq.removeEventListener('change', onChange);
    };
  }, [breakpoint]);
  return isDesktop;
}

// Helper to highlight search keywords
function HighlightQuery({ text, query }: { text: string; query: string }) {
  if (!query || !query.trim() || !text) return <>{text}</>;

  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'));
  return (
    <>
      {parts.map((part, idx) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <mark key={idx} className="bg-amber-200 text-zinc-950 font-semibold px-0.5 rounded">
            {part}
          </mark>
        ) : (
          <span key={idx}>{part}</span>
        )
      )}
    </>
  );
}

// Helper to render markdown paragraph with inline math, bold, code, images and highlights
function HighlightedText({ text, query }: { text: string; query: string }) {
  if (!text) return null;

  const tokens = text.split(/(\$\$[\s\S]*?\$\$|\$.*?\$|\*\*.*?\*\*|`.*?`|!\[[^\]]*\]\([^)]*\))/g);

  return (
    <>
      {tokens.map((token, tIdx) => {
        if (token.startsWith('$$') && token.endsWith('$$')) {
          return (
            <div key={tIdx} className="my-2.5 p-3 rounded-md bg-zinc-50 border border-zinc-200 overflow-x-auto text-center">
              <MathFormula latex={token.slice(2, -2)} displayMode={true} />
            </div>
          );
        }
        if (token.startsWith('$') && token.endsWith('$') && token.length > 2) {
          return <MathFormula key={tIdx} latex={token.slice(1, -1)} displayMode={false} className="inline" />;
        }
        if (token.startsWith('**') && token.endsWith('**')) {
          const inner = token.slice(2, -2);
          return (
            <strong key={tIdx} className="font-bold text-zinc-950">
              <HighlightQuery text={inner} query={query} />
            </strong>
          );
        }
        if (token.startsWith('`') && token.endsWith('`')) {
          const inner = token.slice(1, -1);
          return (
            <code key={tIdx} className="px-1.5 py-0.5 rounded bg-zinc-100 font-mono text-[11px] text-zinc-900 border border-zinc-200">
              <HighlightQuery text={inner} query={query} />
            </code>
          );
        }
        // Inline markdown image (e.g. Mistral OCR figure data URIs)
        const imgMatch = token.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
        if (imgMatch && imgMatch[2]) {
          return (
            <img
              key={tIdx}
              src={imgMatch[2]}
              alt={imgMatch[1] || 'Figure'}
              className="my-3 max-w-full max-h-96 w-auto mx-auto rounded-md border border-zinc-200 shadow-2xs bg-white"
              loading="lazy"
            />
          );
        }
        return <HighlightQuery key={tIdx} text={token} query={query} />;
      })}
    </>
  );
}

interface DualPaneWorkspaceProps {
  paper: PaperData;
  onGoBack?: () => void;
}

// ── Reader block model & parsers ──────────────────────────────────────────────

type ReaderBlock =
  | { type: 'h1'; title: string }
  | { type: 'h2'; title: string; id: string }
  | { type: 'h3'; title: string; id: string }
  | { type: 'math'; latex: string }
  | { type: 'list'; items: string[] }
  | { type: 'abstract'; text: string }
  | { type: 'p'; text: string };

const slugifySection = (title: string) => `sec-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;

// Parse markdown (Mistral OCR pages / fallback markdownContent) into blocks.
function parseMarkdownBlocks(raw: string): ReaderBlock[] {
  const blocks: ReaderBlock[] = [];
  const lines = raw.split('\n');
  let currentList: string[] = [];
  const flushList = () => {
    if (currentList.length > 0) {
      blocks.push({ type: 'list', items: [...currentList] });
      currentList = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) {
      flushList();
      continue;
    }

    // Display Math $$ ... $$
    if (line.startsWith('$$')) {
      flushList();
      let mathContent = line;
      if (!line.endsWith('$$') || line === '$$') {
        i++;
        while (i < lines.length && !lines[i].trim().endsWith('$$')) {
          mathContent += ' ' + lines[i].trim();
          i++;
        }
        if (i < lines.length) {
          mathContent += ' ' + lines[i].trim();
        }
      }
      blocks.push({ type: 'math', latex: mathContent.replace(/^\$\$|\$\$$/g, '').trim() });
      continue;
    }

    if (line.startsWith('# ')) {
      flushList();
      blocks.push({ type: 'h1', title: line.replace('# ', '').trim() });
    } else if (line.startsWith('## ')) {
      flushList();
      const title = line.replace('## ', '').trim();
      blocks.push({ type: 'h2', title, id: slugifySection(title) });
    } else if (line.startsWith('### ')) {
      flushList();
      const title = line.replace('### ', '').trim();
      blocks.push({ type: 'h3', title, id: slugifySection(title) });
    } else if (line.startsWith('- ') || line.startsWith('* ') || /^\d+\.\s/.test(line)) {
      currentList.push(line.replace(/^[-*]\s+|\d+\.\s+/, ''));
    } else {
      flushList();
      if (line.toLowerCase().includes('dominant sequence transduction') || line.toLowerCase().startsWith('the dominant sequence')) {
        blocks.push({ type: 'abstract', text: line });
      } else {
        blocks.push({ type: 'p', text: line });
      }
    }
  }
  flushList();

  // A heading titled "Abstract" turns its following paragraph into the quote box
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    if ((b.type === 'h1' || b.type === 'h2') && b.title.toLowerCase() === 'abstract') {
      const next = blocks[i + 1];
      blocks.splice(i, 1);
      if (next && next.type === 'p') blocks[i] = { type: 'abstract', text: next.text };
      break;
    }
  }
  return blocks;
}

// Parse a raw text layer page (PyMuPDF) into paragraphs: consecutive non-blank
// lines are joined — get_text() emits one line per visual line, not paragraphs.
function parsePlainBlocks(raw: string): ReaderBlock[] {
  const blocks: ReaderBlock[] = [];
  let buf: string[] = [];
  for (const line of raw.split('\n')) {
    const t = line.trim();
    if (!t) {
      if (buf.length > 0) {
        blocks.push({ type: 'p', text: buf.join(' ').trim() });
        buf = [];
      }
    } else {
      buf.push(t);
    }
  }
  if (buf.length > 0) blocks.push({ type: 'p', text: buf.join(' ').trim() });
  return blocks;
}

// Render one parsed block (shared by parsed-document and fallback paths).
function ReaderBlockView({ block, query }: { block: ReaderBlock; query: string }) {
  if (block.type === 'h1') return null; // Document title already rendered in header

  if (block.type === 'h2') {
    return (
      <div id={block.id} className="pt-5 border-t border-zinc-100 space-y-2 scroll-mt-12">
        <h2 className="text-sm md:text-base font-bold text-zinc-950 tracking-tight font-mono flex items-center gap-2">
          <span className="w-1.5 h-4 bg-zinc-950 rounded-xs inline-block" />
          <HighlightQuery text={block.title} query={query} />
        </h2>
      </div>
    );
  }

  if (block.type === 'h3') {
    return (
      <div id={block.id} className="pt-2 space-y-1.5 scroll-mt-12">
        <h3 className="text-xs md:text-sm font-bold text-zinc-900 font-mono flex items-center gap-1.5">
          <span className="text-zinc-400">§</span>
          <HighlightQuery text={block.title} query={query} />
        </h3>
      </div>
    );
  }

  if (block.type === 'math') {
    return (
      <div className="my-3 p-4 rounded-md bg-zinc-50 border border-zinc-200 overflow-x-auto text-center shadow-2xs">
        <MathFormula latex={block.latex} displayMode={true} />
      </div>
    );
  }

  if (block.type === 'list') {
    return (
      <ul className="space-y-1.5 pl-5 list-disc marker:text-zinc-400">
        {block.items.map((it, itIdx) => (
          <li key={itIdx} className="text-zinc-700 leading-relaxed pl-1">
            <HighlightedText text={it} query={query} />
          </li>
        ))}
      </ul>
    );
  }

  if (block.type === 'abstract') {
    return (
      <div className="p-5 rounded-lg bg-zinc-50 border border-zinc-200 text-zinc-800 space-y-2 font-serif shadow-2xs">
        <div className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider font-bold text-zinc-900">
          <Quote className="w-3.5 h-3.5 text-zinc-500" />
          <span>Abstract</span>
        </div>
        <p className="leading-relaxed text-zinc-700 italic text-sm md:text-[15px]">
          <HighlightedText text={block.text} query={query} />
        </p>
      </div>
    );
  }

  return (
    <div className="text-zinc-700 leading-relaxed">
      <HighlightedText text={block.text} query={query} />
    </div>
  );
}

export const DualPaneWorkspace: React.FC<DualPaneWorkspaceProps> = ({ paper, onGoBack }) => {
  const [activeTab, setActiveTab] = useState<'summary' | 'visualizer' | 'math' | 'code' | 'chat'>('summary');
  const [mobileTab, setMobileTab] = useState<'paper' | 'workspace'>('workspace');
  
  // Resizable Center Bar Split State (Percentage for Left Pane)
  const [splitPercent, setSplitPercent] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDesktop = useDesktop();

  // Computed: effective split (0% when expanded — workspace takes full width)
  const effectiveSplit = isExpanded ? 0 : splitPercent;

  // Tab click handler — preserve user's split position across all tabs
  const handleTabClick = useCallback((tab: typeof activeTab) => {
    setActiveTab(tab);
  }, []);

  // Toggle expand/collapse manually
  const toggleExpand = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  // Keyboard: Escape to collapse
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isExpanded) setIsExpanded(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isExpanded]);

  // Left Pane View Mode: 'pdf' (Real embedded PDF) or 'reader' (Parsed structured text)
  const [leftViewMode, setLeftViewMode] = useState<'pdf' | 'reader'>('pdf');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [fontSize, setFontSize] = useState<'sm' | 'base' | 'lg'>('sm');
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const [citeDropdownOpen, setCiteDropdownOpen] = useState(false);
  const [pdfLoadError, setPdfLoadError] = useState(false);
  const [pdfPage, setPdfPage] = useState<number>(1);
  const [pdfJumpNotice, setPdfJumpNotice] = useState<string | null>(null);

  // Bookmark — persisted to localStorage keyed by paper id / arxivId
  const bookmarkKey = `mytldr_saved_${paper.metadata.id || paper.metadata.arxivId || 'paper'}`;
  const [isBookmarked, setIsBookmarked] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try { return localStorage.getItem(bookmarkKey) === '1'; } catch { return false; }
  });

  const toggleBookmark = useCallback(() => {
    setIsBookmarked((prev) => {
      const next = !prev;
      try { next ? localStorage.setItem(bookmarkKey, '1') : localStorage.removeItem(bookmarkKey); } catch { /* ignore */ }
      return next;
    });
  }, [bookmarkKey]);

  const handleJumpToPdfPage = useCallback((page: number, sectionTitle?: string) => {
    setPdfPage(page);
    setLeftViewMode('pdf');
    setMobileTab('paper');
    setPdfJumpNotice(`📄 Jumped to PDF Page ${page}${sectionTitle ? ` (${sectionTitle})` : ''}`);
    setTimeout(() => setPdfJumpNotice(null), 4000);
  }, []);

  // Bottom Floating LLM Chat State & Active Prompt Transfer
  const [chatInput, setChatInput] = useState<string>('');
  const [pendingChatPrompt, setPendingChatPrompt] = useState<string | null>(null);

  // Handle Dragging Divider
  const startDragging = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const stopDragging = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrag = useCallback(
    (e: MouseEvent | TouchEvent) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const offset = clientX - rect.left;
      const newPercent = (offset / rect.width) * 100;
      
      // Clamp between 20% and 80%
      setSplitPercent(Math.min(Math.max(newPercent, 20), 80));
    },
    [isDragging]
  );

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleDrag);
      window.addEventListener('mouseup', stopDragging);
      window.addEventListener('touchmove', handleDrag);
      window.addEventListener('touchend', stopDragging);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      window.removeEventListener('mousemove', handleDrag);
      window.removeEventListener('mouseup', stopDragging);
      window.removeEventListener('touchmove', handleDrag);
      window.removeEventListener('touchend', stopDragging);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      window.removeEventListener('mousemove', handleDrag);
      window.removeEventListener('mouseup', stopDragging);
      window.removeEventListener('touchmove', handleDrag);
      window.removeEventListener('touchend', stopDragging);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleDrag, stopDragging]);

  // Handle Quick Chat Prompt Execution - Opens AI Assistant tab with smooth transition
  const handleExecutePrompt = (promptText: string) => {
    if (!promptText.trim()) return;
    setPendingChatPrompt(promptText.trim());
    setActiveTab('chat');
    setMobileTab('workspace');
    setChatInput('');
  };

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    handleExecutePrompt(chatInput.trim());
  };

  // Close cite dropdown on outside click
  useEffect(() => {
    if (!citeDropdownOpen) return;
    const close = (e: MouseEvent) => {
      const target = e.target as Element;
      if (!target.closest('[data-cite-dropdown]')) {
        setCiteDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [citeDropdownOpen]);


  // ── BibTeX helpers ──────────────────────────────────────────────────────────
  const buildBibtex = useCallback(() => {
    const m = paper.metadata;
    const arxivId = m.arxivId || '';
    const year = m.publishedDate ? new Date(m.publishedDate).getFullYear() : new Date().getFullYear();
    const firstAuthor = (m.authors?.[0] || 'unknown').split(' ').pop()?.toLowerCase() || 'unknown';
    const slug = (m.title || 'paper').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40);
    const key = `${firstAuthor}${year}${slug}`.replace(/[^a-z0-9]/g, '');
    const authorList = (m.authors || []).join(' and ');
    return `@article{${key},\n  title={${m.title || 'Untitled'}},\n  author={${authorList}},\n  journal={arXiv preprint${arxivId ? ` arXiv:${arxivId}` : ''}},\n  year={${year}}\n}`;
  }, [paper.metadata]);

  const copyBibtex = useCallback(() => {
    navigator.clipboard.writeText(buildBibtex());
    setCopiedBibtex(true);
    setCiteDropdownOpen(false);
    setTimeout(() => setCopiedBibtex(false), 2000);
  }, [buildBibtex]);

  const downloadBibtex = useCallback(() => {
    const bibtex = buildBibtex();
    const m = paper.metadata;
    const slug = (m.title || 'paper').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 40);
    const blob = new Blob([bibtex], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slug}.bib`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setCiteDropdownOpen(false);
  }, [buildBibtex, paper.metadata]);

  const downloadSummary = useCallback(() => {
    const m = paper.metadata;
    const s = paper.structuredSummary;
    const lines: string[] = [];
    lines.push(`# ${m.title || 'Research Paper'}`);
    lines.push('');
    if (m.authors?.length) lines.push(`**Authors:** ${m.authors.join(', ')}`);
    if (m.arxivId) lines.push(`**arXiv:** ${m.arxivId}`);
    if (m.publishedDate) lines.push(`**Published:** ${m.publishedDate}`);
    lines.push('');
    lines.push('---');
    lines.push('');
    if (s?.overview?.one_sentence) {
      lines.push('## One Sentence Summary');
      lines.push('');
      lines.push(`> ${s.overview.one_sentence}`);
      lines.push('');
    }
    if (s?.overview?.problem) {
      lines.push('## Problem');
      lines.push('');
      lines.push(s.overview.problem);
      lines.push('');
    }
    if (s?.overview?.solution) {
      lines.push('## Core Idea');
      lines.push('');
      lines.push(s.overview.solution);
      lines.push('');
    }
    if (s?.contributions?.length) {
      lines.push('## Contributions');
      lines.push('');
      s.contributions.forEach((c, i) => {
        lines.push(`### ${i + 1}. ${c.title}`);
        lines.push('');
        lines.push(c.summary);
        lines.push('');
      });
    }
    if (s?.key_findings?.length) {
      lines.push('## Key Results');
      lines.push('');
      s.key_findings.forEach((f) => {
        lines.push(`- **${f.title}** ${f.metric_value ? `(${f.metric_value})` : ''}: ${f.summary}`);
      });
      lines.push('');
    }
    if (s?.limitations?.length) {
      lines.push('## Limitations');
      lines.push('');
      s.limitations.forEach((l) => {
        lines.push(`- **${l.title}**: ${l.summary}`);
      });
      lines.push('');
    }
    lines.push('---');
    lines.push('');
    lines.push(`*Generated by myTLDR — ${new Date().toLocaleDateString()}*`);
    const slug = (m.title || 'summary').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 40);
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slug}_summary.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [paper.metadata, paper.structuredSummary]);

  // Dynamic Quick Chat Suggestions grounded in active paper
  const quickSuggestions = React.useMemo(() => {
    const kws = paper.metadata.keywords || [];
    if (kws.length >= 3) {
      return [
        `Explain ${kws[0].term.replace(/^#/, '')} in 1 min`,
        `Why is ${kws[1].term.replace(/^#/, '')} needed?`,
        `How does ${kws[2].term.replace(/^#/, '')} work?`,
        `Show tensor shapes & code`
      ];
    }
    return [
      `Summarize ${paper.metadata.title} in 2 min`,
      'What core bottleneck does this solve?',
      'Break down the mathematical equations',
      'Show PyTorch implementation'
    ];
  }, [paper.metadata.keywords, paper.metadata.title]);

  // Font size classes for structured reader
  const fontClass = {
    sm: 'text-xs leading-relaxed',
    base: 'text-sm leading-relaxed',
    lg: 'text-base leading-relaxed'
  }[fontSize];

  // Real parsed document from the backend (per-page extraction / OCR), if available
  const parsedDoc = paper.parsedDocument;

  // Jump-To entries: parsed headings when available, else ## lines of markdownContent
  const tocSections = React.useMemo(() => {
    const sections: { title: string; id: string }[] = [];
    if (parsedDoc) {
      for (const page of parsedDoc.pages) {
        if (page.source !== 'mistral_ocr') continue;
        for (const b of parseMarkdownBlocks(page.markdown)) {
          if ((b.type === 'h2' || b.type === 'h3') && !sections.some((s) => s.id === b.id)) {
            sections.push({ title: b.title, id: b.id });
          }
        }
      }
    } else {
      (paper.markdownContent || '')
        .split('\n')
        .filter((l) => l.startsWith('## '))
        .forEach((h2Line) => {
          const title = h2Line.replace('## ', '').trim();
          sections.push({ title, id: slugifySection(title) });
        });
    }
    return sections;
  }, [parsedDoc, paper.markdownContent]);

  // Search corpus for the match counter (images stripped so data URIs aren't scanned)
  const readerSearchText = parsedDoc
    ? parsedDoc.pages.map((p) => p.markdown.replace(/!\[[^\]]*\]\([^)]*\)/g, '')).join('\n')
    : paper.markdownContent || '';

  return (
    <div
      ref={containerRef}
      className="flex-1 flex flex-col lg:flex-row w-full min-h-0 overflow-hidden bg-[#FAFAFA]"
    >
      {/* Mobile / Small Screen View Switcher (< 1024px) */}
      <div className="flex lg:hidden items-center justify-center p-2 bg-white border-b border-zinc-200 z-20 shrink-0">
        <div className="flex items-center bg-zinc-100 rounded-md p-0.5 border border-zinc-200 w-full max-w-sm">
          <button
            onClick={() => setMobileTab('paper')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded transition-all text-center ${
              mobileTab === 'paper' ? 'bg-white text-zinc-950 shadow-2xs' : 'text-zinc-500 hover:text-zinc-900'
            }`}
          >
            📄 Original Paper
          </button>
          <button
            onClick={() => setMobileTab('workspace')}
            className={`flex-1 py-1.5 text-xs font-semibold rounded transition-all text-center ${
              mobileTab === 'workspace' ? 'bg-white text-zinc-950 shadow-2xs' : 'text-zinc-500 hover:text-zinc-900'
            }`}
          >
            🔬 3D Deconstructor
          </button>
        </div>
      </div>

      {/* =========================================================================
          LEFT PANE: Paper Document / PDF Viewer
          ========================================================================= */}
        <div
          style={isDesktop ? { width: `${effectiveSplit}%` } : undefined}
          className={`w-full lg:h-full flex-col bg-[#FFFFFF] border-r border-zinc-200/80 overflow-hidden shrink-0 ${
            isDragging ? 'transition-none' : 'transition-[width] duration-300 ease-in-out'
          } ${mobileTab === 'paper' ? 'flex flex-1' : 'hidden lg:flex'}`}
        >
        {/* Left Header Bar with Go Back & View Mode Switcher */}
        <div className="flex items-center justify-between px-3.5 py-2 border-b border-zinc-200/80 bg-[#FAFAFA] z-10 shrink-0">
          <div className="flex items-center gap-2">
            {onGoBack && (
              <button
                onClick={onGoBack}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-white hover:bg-zinc-100 border border-zinc-200 text-zinc-900 text-xs font-medium shadow-2xs transition-all active:scale-95"
                title="Return to Paper Selector"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>Go Back</span>
              </button>
            )}

            {/* Segmented View Mode Toggle */}
            <div className="flex items-center bg-zinc-200/70 rounded-md p-0.5 border border-zinc-300/60">
              <button
                onClick={() => setLeftViewMode('pdf')}
                className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-all ${
                  leftViewMode === 'pdf'
                    ? 'bg-white text-zinc-950 font-semibold shadow-2xs'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Original PDF</span>
              </button>

              <button
                onClick={() => setLeftViewMode('reader')}
                className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-all ${
                  leftViewMode === 'reader'
                    ? 'bg-white text-zinc-950 font-semibold shadow-2xs'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Parsed Text</span>
              </button>
            </div>
          </div>

          {/* Right Controls in Header */}
          <div className="flex items-center gap-1.5">
            {leftViewMode === 'reader' && (
              <>
                {/* Search in text */}
                <div className="relative hidden xl:block">
                  <Search className="w-3.5 h-3.5 text-zinc-400 absolute left-2.5 top-2" />
                  <input
                    type="text"
                    placeholder="Find in text..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8 pr-2.5 py-1 bg-white border border-zinc-200 rounded-md text-xs text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-400 w-28 focus:w-36 transition-all placeholder:text-zinc-400"
                  />
                </div>

                {/* Font Size Buttons */}
                <div className="flex items-center bg-zinc-100 rounded-md p-0.5 border border-zinc-200">
                  <button
                    onClick={() => setFontSize('sm')}
                    className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${fontSize === 'sm' ? 'bg-white text-zinc-950 font-bold shadow-2xs' : 'text-zinc-600'}`}
                  >
                    A-
                  </button>
                  <button
                    onClick={() => setFontSize('base')}
                    className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${fontSize === 'base' ? 'bg-white text-zinc-950 font-bold shadow-2xs' : 'text-zinc-600'}`}
                  >
                    A
                  </button>
                  <button
                    onClick={() => setFontSize('lg')}
                    className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${fontSize === 'lg' ? 'bg-white text-zinc-950 font-bold shadow-2xs' : 'text-zinc-600'}`}
                  >
                    A+
                  </button>
                </div>
              </>
            )}

            {/* Direct PDF Link */}
            {paper.metadata.pdfUrl && (
              <a
                href={paper.metadata.pdfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-medium transition-all shadow-2xs shrink-0"
                title="Open PDF in new browser tab"
              >
                <span>arXiv</span>
                <ExternalLink className="w-3 h-3 text-zinc-300" />
              </a>
            )}
          </div>
        </div>

        {/* LEFT PANE BODY */}
        {leftViewMode === 'pdf' ? (
          /* =========================================================================
             1. EMBEDDED ORIGINAL PDF VIEWER (VIA ARXIV STREAMING PROXY)
             ========================================================================= */
          <div className="flex-1 w-full h-full bg-zinc-100/70 relative overflow-hidden flex flex-col">
            {/* Provenance Page Jump Toast Overlay */}
            {pdfJumpNotice && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 px-3.5 py-1.5 rounded-full bg-zinc-950/90 backdrop-blur-md text-white text-xs font-mono font-medium shadow-lg border border-zinc-700 animate-fadeIn flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                <span>{pdfJumpNotice}</span>
              </div>
            )}

            {!pdfLoadError ? (
              <iframe
                key={`pdf-page-${pdfPage}-${paper.metadata.id}`}
                src={
                  paper.metadata.fileBlobUrl
                    ? `${paper.metadata.fileBlobUrl}#page=${pdfPage}&toolbar=1&navpanes=0&view=FitH`
                    : paper.metadata.pdfUrl
                      ? `${paper.metadata.pdfUrl}#page=${pdfPage}&toolbar=1&navpanes=0&view=FitH`
                      : `https://arxiv.org/pdf/${encodeURIComponent(
                          paper.metadata.arxivId || '—'
                        )}.pdf#page=${pdfPage}&toolbar=1&navpanes=0&view=FitH`
                }
                className={`w-full flex-1 border-0 bg-white ${isDragging ? 'pointer-events-none' : ''}`}
                title={`Original arXiv PDF - ${paper.metadata.title}`}
                onError={() => setPdfLoadError(true)}
              />
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-4">
                <FileText className="w-12 h-12 text-zinc-400" />
                <div className="space-y-1">
                  <h3 className="font-bold text-zinc-900 text-sm">PDF Embedded Viewer</h3>
                  <p className="text-xs text-zinc-500 max-w-sm">
                    If the PDF is taking long to stream, you can open it directly on arXiv or switch to Parsed Text mode.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <a
                    href={paper.metadata.pdfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-3.5 py-1.5 rounded-md bg-zinc-900 text-white text-xs font-medium shadow-xs"
                  >
                    Open on arXiv
                  </a>
                  <button
                    onClick={() => setLeftViewMode('reader')}
                    className="px-3.5 py-1.5 rounded-md bg-white border border-zinc-200 text-zinc-800 text-xs font-medium shadow-2xs"
                  >
                    Switch to Parsed Text
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* =========================================================================
             2. DYNAMIC STRUCTURED EDITORIAL READER WITH TOC & SEARCH
             ========================================================================= */
          <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#FFFFFF]">
            
            {/* Interactive Table of Contents (TOC) Horizontal Bar */}
            <div className="px-4 py-2 bg-zinc-50 border-b border-zinc-200/80 flex items-center gap-2 overflow-x-auto shrink-0 scrollbar-none">
              <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 font-semibold flex items-center gap-1 shrink-0">
                <List className="w-3 h-3 text-zinc-500" />
                <span>Jump to:</span>
              </span>

              {/* Section entries derived from the parsed document (or fallback markdown) */}
              {tocSections.map((section, hIdx) => (
                <button
                  key={hIdx}
                  onClick={() => {
                    const el = document.getElementById(section.id);
                    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                  }}
                  className="px-2.5 py-0.5 rounded-md bg-white hover:bg-zinc-100 border border-zinc-200 text-[11px] font-mono text-zinc-700 font-medium whitespace-nowrap transition-all shadow-2xs active:scale-95 shrink-0"
                >
                  {section.title}
                </button>
              ))}
            </div>

            {/* Search Match Counter Pill (if searching) */}
            {searchQuery.trim() && (
              <div className="px-6 py-1.5 bg-amber-50/80 border-b border-amber-200/60 text-amber-900 text-xs font-mono flex items-center justify-between">
                <span>Search results for: &ldquo;{searchQuery}&rdquo;</span>
                <span className="font-semibold">
                  {(readerSearchText.toLowerCase().match(new RegExp(searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').toLowerCase(), 'g')) || []).length} matches
                </span>
              </div>
            )}

            {/* Scrollable Reader Content Body */}
            <div className="flex-1 p-6 md:p-8 overflow-y-auto space-y-7 text-zinc-800 font-sans select-text">
              
              {/* Document Header Bar */}
              <div className="border-b border-zinc-200 pb-5 space-y-2.5">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded-md bg-zinc-100 text-zinc-700 border border-zinc-200 font-semibold">
                    Original Paper Text
                  </span>
                  <span className="text-xs font-mono text-zinc-400">
                    {paper.metadata.publishedDate}
                  </span>
                  {paper.metadata.arxivId && (
                    <span className="text-xs font-mono text-zinc-500">
                      arXiv:{paper.metadata.arxivId}
                    </span>
                  )}
                </div>
                <h1 className="text-2xl md:text-3xl font-extrabold text-zinc-950 tracking-tight leading-snug">
                  <HighlightQuery text={paper.metadata.title} query={searchQuery} />
                </h1>
                <p className="text-xs text-zinc-600 font-medium">
                  {paper.metadata.authors.join(', ')}
                </p>
              </div>

              {/* Dynamic Section Renderer */}
              <div className={`space-y-5 ${fontClass}`}>
                {(() => {
                  // Real parsed document: render per-page with "Page N of M" pills
                  if (parsedDoc) {
                    return parsedDoc.pages.map((page) => {
                      const blocks = (page.source === 'mistral_ocr'
                        ? parseMarkdownBlocks(page.markdown)
                        : parsePlainBlocks(page.markdown)
                      ).map((b) => (b.type === 'h1' ? { ...b, type: 'h2' as const, id: slugifySection(b.title) } : b));
                      return (
                        <div key={`page-${page.pageNumber}`}>
                          {/* Page pill separator — click syncs the Original PDF pane */}
                          <div className="flex items-center gap-3 pt-3 pb-1">
                            <button
                              onClick={() => setPdfPage(page.pageNumber)}
                              title="Sync the Original PDF pane to this page"
                              className="px-2.5 py-0.5 rounded-full bg-zinc-100 hover:bg-zinc-200 border border-zinc-200 text-[10px] font-mono font-semibold text-zinc-600 transition-all shrink-0 active:scale-95"
                            >
                              Page {page.pageNumber} of {parsedDoc.pageCount}
                            </button>
                            <div className="flex-1 h-px bg-zinc-100" />
                            <span className="text-[9px] font-mono uppercase tracking-wider text-zinc-400 shrink-0">
                              {page.source === 'mistral_ocr' ? 'OCR' : 'Text layer'}
                            </span>
                          </div>
                          <div className="space-y-3">
                            {blocks.map((block, idx) => (
                              <ReaderBlockView key={idx} block={block} query={searchQuery} />
                            ))}
                          </div>
                        </div>
                      );
                    });
                  }

                  // Fallback: single markdown string (mock data / older cached runs)
                  return parseMarkdownBlocks(paper.markdownContent || '').map((block, idx) => (
                    <ReaderBlockView key={idx} block={block} query={searchQuery} />
                  ));
                })()}

                {/* Interactive Bridge Cards */}
                <div className="pt-6 border-t border-zinc-200 grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-3.5 rounded-lg bg-zinc-50 border border-zinc-200 flex items-center justify-between gap-3 shadow-2xs">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-md bg-white border border-zinc-200 flex items-center justify-center text-zinc-900 shrink-0 shadow-2xs">
                        <Layers className="w-3.5 h-3.5" />
                      </div>
                      <div>
                        <span className="font-bold text-zinc-900 text-xs block">3D Architecture</span>
                        <span className="text-[11px] text-zinc-500">Interactive stacked visualizer</span>
                      </div>
                    </div>
                    <button
                      onClick={() => setActiveTab('visualizer')}
                      className="px-2.5 py-1 rounded-md bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-medium transition-all shrink-0"
                    >
                      Explore 3D
                    </button>
                  </div>

                  <div className="p-3.5 rounded-lg bg-zinc-50 border border-zinc-200 flex items-center justify-between gap-3 shadow-2xs">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-md bg-white border border-zinc-200 flex items-center justify-center text-zinc-900 shrink-0 shadow-2xs">
                        <Calculator className="w-3.5 h-3.5" />
                      </div>
                      <div>
                        <span className="font-bold text-zinc-900 text-xs block">Math & Equations</span>
                        <span className="text-[11px] text-zinc-500">Deconstructed variable shapes</span>
                      </div>
                    </div>
                    <button
                      onClick={() => setActiveTab('math')}
                      className="px-2.5 py-1 rounded-md bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-medium transition-all shrink-0"
                    >
                      View Math
                    </button>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sleek Minimal Center Divider */}
      {!isExpanded && (
      <div
        onMouseDown={startDragging}
        onTouchStart={startDragging}
        onDoubleClick={() => setSplitPercent(50)}
        className="hidden lg:flex relative w-1.5 hover:w-2 -mx-0.5 bg-transparent hover:bg-zinc-200/60 active:bg-zinc-300/80 items-center justify-center cursor-col-resize group select-none transition-all shrink-0 z-30"
        title="Drag to resize panels • Double-click to reset (50/50)"
      >
        {/* Subtle center line */}
        <div className={`w-[1px] h-full ${isDragging ? 'bg-zinc-900' : 'bg-zinc-200 group-hover:bg-zinc-400'} transition-colors`} />

        {/* Minimal pill handle */}
        <div className={`absolute top-1/2 -translate-y-1/2 w-1 h-7 rounded-full shadow-2xs transition-all ${
          isDragging ? 'bg-zinc-950 scale-y-110' : 'bg-zinc-300 group-hover:bg-zinc-700'
        }`} />
      </div>
      )}

      {/* =========================================================================
          RIGHT PANE: Knowledge, Visualizer, Tabs & Docked LLM Chat
          ========================================================================= */}
      <div
        style={isDesktop ? { width: `calc(100% - ${effectiveSplit}%)` } : undefined}
        className={`w-full lg:h-full flex-col bg-[#FFFFFF] overflow-hidden relative ${
          isDragging ? 'transition-none' : 'transition-[width] duration-300 ease-in-out'
        } ${mobileTab === 'workspace' ? 'flex flex-1' : 'hidden lg:flex'}`}
      >
        {/* ── RIGHT PANEL HEADER: 2 rows ───────────────────────────────────────
            Row 1: Paper title + action buttons (Save, Cite, Download, Expand)
            Row 2: Tab navigation bar (clean, no clutter)
            ──────────────────────────────────────────────────────────────────── */}
        <div className="shrink-0 bg-[#FAFAFA] border-b border-zinc-200/80">

          {/* Row 1 — paper identity + actions */}
          <div className="flex items-center justify-between gap-3 px-3.5 pt-2.5 pb-2 border-b border-zinc-100">
            {/* Paper title (truncated) */}
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-400 shrink-0 hidden sm:block">Paper</span>
              <h2 className="text-[12px] font-semibold text-zinc-800 truncate leading-tight" title={paper.metadata.title}>
                {paper.metadata.title}
              </h2>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-1.5 shrink-0">

              {/* Save / Bookmark (persisted to localStorage) */}
              <button
                onClick={toggleBookmark}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-[11px] font-medium transition-all active:scale-95 ${
                  isBookmarked
                    ? 'bg-zinc-900 text-white border-zinc-900'
                    : 'bg-white text-zinc-600 border-zinc-200 hover:bg-zinc-50 hover:border-zinc-300'
                }`}
                title={isBookmarked ? 'Saved — click to unsave' : 'Save this paper'}
              >
                <Bookmark className={`w-3.5 h-3.5 ${isBookmarked ? 'fill-current' : ''}`} />
                <span className="hidden sm:inline">{isBookmarked ? 'Saved' : 'Save'}</span>
              </button>

              {/* Download Summary .md */}
              <button
                onClick={downloadSummary}
                disabled={!paper.structuredSummary}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 bg-white hover:bg-zinc-50 hover:border-zinc-300 text-zinc-600 text-[11px] font-medium transition-all active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed"
                title="Download structured summary as Markdown"
              >
                <FileText className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Summary</span>
              </button>

              {/* Cite — dropdown: Copy BibTeX or Download .bib */}
              <div className="relative" data-cite-dropdown>
                <button
                  onClick={() => setCiteDropdownOpen((o) => !o)}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 bg-white hover:bg-zinc-50 hover:border-zinc-300 text-zinc-600 text-[11px] font-medium transition-all active:scale-95"
                  title="Cite this paper"
                >
                  {copiedBibtex ? (
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                  ) : (
                    <Quote className="w-3.5 h-3.5" />
                  )}
                  <span>{copiedBibtex ? 'Copied!' : 'Cite'}</span>
                </button>

                {citeDropdownOpen && (
                  <div className="absolute right-0 top-full mt-1.5 z-50 w-48 rounded-xl bg-white border border-zinc-200 shadow-lg overflow-hidden">
                    <button
                      onClick={copyBibtex}
                      className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-[12px] text-zinc-700 hover:bg-zinc-50 transition-colors text-left"
                    >
                      <Copy className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
                      <span>Copy BibTeX</span>
                    </button>
                    <div className="border-t border-zinc-100" />
                    <button
                      onClick={downloadBibtex}
                      className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-[12px] text-zinc-700 hover:bg-zinc-50 transition-colors text-left"
                    >
                      <Share2 className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
                      <span>Download .bib file</span>
                    </button>
                  </div>
                )}
              </div>

              {/* Expand/Collapse toggle */}
              <button
                onClick={toggleExpand}
                className={`p-1.5 rounded-lg border text-xs transition-all ${
                  isExpanded
                    ? 'bg-blue-50 text-blue-600 border-blue-200'
                    : 'bg-white text-zinc-500 hover:text-zinc-900 border-zinc-200 hover:bg-zinc-100'
                }`}
                title={isExpanded ? 'Collapse panel (Escape)' : 'Expand panel to full width'}
              >
                {isExpanded ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/></svg>
                ) : (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
                )}
              </button>
            </div>
          </div>

          {/* Row 2 — Tab bar (no actions here) */}
          <div className="flex items-center overflow-x-auto scrollbar-none px-3 max-lg:[mask-image:linear-gradient(to_right,black_calc(100%-1.5rem),transparent)]">
            <button
              onClick={() => handleTabClick('summary')}
              className={`flex items-center gap-1.5 px-3.5 py-2.5 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
                activeTab === 'summary'
                  ? 'border-zinc-950 text-zinc-950 bg-white font-semibold shadow-2xs'
                  : 'border-transparent text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100/60'
              }`}
            >
              <Lightbulb className="w-3.5 h-3.5" />
              <span>Summary</span>
            </button>

            <button
              onClick={() => handleTabClick('visualizer')}
              className={`flex items-center gap-1.5 px-3.5 py-2.5 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
                activeTab === 'visualizer'
                  ? 'border-zinc-950 text-zinc-950 bg-white font-semibold shadow-2xs'
                  : 'border-transparent text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100/60'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Architecture</span>
            </button>

            <button
              onClick={() => handleTabClick('math')}
              className={`flex items-center gap-1.5 px-3.5 py-2.5 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
                activeTab === 'math'
                  ? 'border-zinc-950 text-zinc-950 bg-white font-semibold shadow-2xs'
                  : 'border-transparent text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100/60'
              }`}
            >
              <Calculator className="w-3.5 h-3.5" />
              <span>Equations</span>
            </button>

            <button
              onClick={() => handleTabClick('code')}
              className={`flex items-center gap-1.5 px-3.5 py-2.5 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
                activeTab === 'code'
                  ? 'border-zinc-950 text-zinc-950 bg-white font-semibold shadow-2xs'
                  : 'border-transparent text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100/60'
              }`}
            >
              <Code className="w-3.5 h-3.5" />
              <span>PyTorch</span>
            </button>

            <button
              onClick={() => handleTabClick('chat')}
              className={`flex items-center gap-1.5 px-3.5 py-2.5 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
                activeTab === 'chat'
                  ? 'border-zinc-950 text-zinc-950 bg-white font-semibold shadow-2xs'
                  : 'border-transparent text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100/60'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>AI Chat</span>
            </button>
          </div>
        </div>


        {/* MAIN TAB CONTENT DISPLAY ("TEXT SUMMARY" / 3D / Math / Code / Chat) */}
        <div className={`flex-1 ${activeTab === 'chat' ? 'overflow-hidden flex flex-col' : 'overflow-y-auto pb-36'} bg-[#FAFAFA]`}>
          <TabErrorBoundary tabName="Summary">
            {activeTab === 'summary' && (
              <SummaryTab
                paper={paper}
                onAskAI={handleExecutePrompt}
                onJumpToPage={handleJumpToPdfPage}
              />
            )}
          </TabErrorBoundary>
          <TabErrorBoundary tabName="Architecture Visualizer">
            {activeTab === 'visualizer' && <Architecture2DTab paper={paper} onJumpToPage={handleJumpToPdfPage} />}
          </TabErrorBoundary>
          <TabErrorBoundary tabName="Equations & Math">
            {activeTab === 'math' && <MathDeconstructedTab paper={paper} />}
          </TabErrorBoundary>
          <TabErrorBoundary tabName="PyTorch Code">
            {activeTab === 'code' && <CodeWalkthroughTab paper={paper} />}
          </TabErrorBoundary>
          <TabErrorBoundary tabName="AI Assistant">
            {activeTab === 'chat' && (
              <AIChatTab
                paper={paper}
                initialPrompt={pendingChatPrompt}
                onPromptHandled={() => setPendingChatPrompt(null)}
                onAskAI={handleExecutePrompt}
                onJumpToPage={handleJumpToPdfPage}
              />
            )}
          </TabErrorBoundary>
        </div>

        {/* Floating Bottom Chat Dock — one glass layer done properly: the
            scrim fades scrolled content underneath, the form alone carries
            the frost. Pills stay solid (nested blurs compound into mud). */}
        {activeTab !== 'chat' && activeTab !== 'visualizer' && (
          <div className="absolute bottom-0 left-0 right-0 z-30 pointer-events-none pt-14 pb-4 px-4 md:px-6 bg-gradient-to-t from-[#FAFAFA] via-[#FAFAFA]/90 to-transparent">
            <div className="max-w-2xl mx-auto w-full space-y-2 pointer-events-auto">
              
              {/* Quick Suggested Prompt Pills (Floating Capsules) */}
              <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 scrollbar-none px-1">
                <span className="text-[10px] font-mono text-zinc-400 uppercase font-bold shrink-0">
                  Quick Ask:
                </span>
                {quickSuggestions.map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExecutePrompt(suggestion)}
                    className="px-2.5 py-1 rounded-full bg-white hover:bg-zinc-50 border border-zinc-200 text-[11px] text-zinc-700 font-medium whitespace-nowrap transition-all shadow-xs hover:shadow-sm active:scale-95 shrink-0"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>

              {/* LLM Chat Input Form - modern minimal input */}
              <form onSubmit={handleChatSubmit} className="relative flex items-center bg-white/95 backdrop-blur-xl border border-zinc-200 shadow-xl shadow-zinc-950/8 rounded-2xl p-1.5 transition-all focus-within:border-zinc-950/70 focus-within:ring-2 focus-within:ring-zinc-950/10 focus-within:shadow-2xl">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder={`Ask any question about ${paper.metadata.title}...`}
                  className="w-full pl-3.5 pr-12 py-2 bg-transparent text-xs md:text-sm text-zinc-900 focus:outline-none placeholder:text-zinc-400 font-normal"
                />
                <div className="absolute right-2 flex items-center">
                  <button
                    type="submit"
                    disabled={!chatInput.trim()}
                    className="w-8 h-8 rounded-full bg-zinc-950 hover:bg-zinc-800 disabled:opacity-20 text-white flex items-center justify-center transition-all shadow-xs active:scale-95"
                    title="Send to AI Assistant"
                  >
                    <ArrowUp className="w-4 h-4" />
                  </button>
                </div>
              </form>

            </div>
          </div>
        )}

      </div>

      {/* Global Drag Shield to prevent iframe mouse capture during resize */}
      {isDragging && (
        <div className="fixed inset-0 z-50 cursor-col-resize select-none bg-transparent" />
      )}
    </div>
  );
};