'use client';

import React, { useState, useCallback } from 'react';
import { PaperData } from '@/types/paper';
import { Code, Copy, Check, Terminal, FileCode2, ChevronDown, ChevronRight, ClipboardList } from 'lucide-react';

interface CodeWalkthroughTabProps {
  paper: PaperData;
}

// Lightweight syntax tokenizer for Python / PyTorch (Light theme)
function highlightPythonLine(line: string): React.ReactNode {
  // Full-line comments
  if (line.trim().startsWith('#')) {
    return <span className="text-zinc-400 italic">{line}</span>;
  }

  const tokenRegex = /(#.*$)|(\b(?:def|class|import|from|as|return|if|else|elif|for|while|in|not|is|None|True|False|with|try|except|finally|raise|yield|async|await|lambda|pass|break|continue)\b)|(\b(?:torch|nn|math|Tensor|F|self|super|Optional|List|Dict|Tuple|Any|Union|scaled_dot_product_attention|matmul|sqrt|softmax|transpose|size|masked_fill|view|reshape|permute|contiguous|zeros|ones|randn|arange|cat|stack|einsum|Linear|Conv2d|LayerNorm|MultiheadAttention|Dropout|ReLU|GELU|Softmax|Parameter|Module|ModuleList|Sequential)\b)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([()[\]{},.:=+\-*/])/g;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(line)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`t-${lastIndex}`} className="text-zinc-800">{line.slice(lastIndex, match.index)}</span>);
    }

    const [full, comment, keyword, builtin, string, number, operator] = match;

    if (comment) {
      parts.push(<span key={match.index} className="text-zinc-400 italic">{comment}</span>);
    } else if (keyword) {
      parts.push(<span key={match.index} className="text-purple-700 font-semibold">{keyword}</span>);
    } else if (builtin) {
      parts.push(<span key={match.index} className="text-blue-700 font-medium">{builtin}</span>);
    } else if (string) {
      parts.push(<span key={match.index} className="text-emerald-700">{string}</span>);
    } else if (number) {
      parts.push(<span key={match.index} className="text-amber-700">{number}</span>);
    } else if (operator) {
      parts.push(<span key={match.index} className="text-zinc-500">{operator}</span>);
    }

    lastIndex = match.index + full.length;
  }

  if (lastIndex < line.length) {
    parts.push(<span key={`tail-${lastIndex}`} className="text-zinc-800">{line.slice(lastIndex)}</span>);
  }

  return parts;
}

// Strip docstrings and ensure clean, concise human single-line comments
function sanitizeCode(code: string): string {
  return code
    .replace(/"""([\s\S]*?)"""|'''([\s\S]*?)'''/g, (_, g1, g2) => {
      const content = (g1 || g2 || '').trim();
      const firstLine = content.split('\n').map((l: string) => l.trim()).filter(Boolean)[0] || '';
      return firstLine ? `# ${firstLine}` : '';
    })
    .trim();
}

interface SnippetCardProps {
  snippet: { title: string; language: string; code: string };
  idx: number;
  copiedIdx: number | null;
  onCopy: (code: string, idx: number) => void;
}

function SnippetCard({ snippet, idx, copiedIdx, onCopy }: SnippetCardProps) {
  const [collapsed, setCollapsed] = useState(false);
  const cleanCode = sanitizeCode(snippet.code);
  const lines = cleanCode.split('\n');

  return (
    <div className="rounded-xl border border-zinc-200/90 bg-white overflow-hidden shadow-2xs transition-all">
      
      {/* Header Bar */}
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-zinc-50/90 border-b border-zinc-200/80 hover:bg-zinc-100/70 transition-colors group text-left"
      >
        <div className="flex items-center gap-3 min-w-0">
          <Terminal className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
          <span className="font-mono text-[13px] font-semibold text-zinc-900 truncate">{snippet.title}</span>
          <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-600 border border-zinc-200">
            {lines.length} lines
          </span>
          <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-200/70 text-zinc-700 border border-zinc-300/70 font-semibold">
            py
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {/* Copy button — stop propagation so it doesn't collapse */}
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); onCopy(cleanCode, idx); }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); onCopy(cleanCode, idx); } }}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white hover:bg-zinc-100 text-zinc-700 text-[11px] font-mono font-medium transition-all active:scale-95 border border-zinc-200 shadow-2xs cursor-pointer"
          >
            {copiedIdx === idx ? (
              <>
                <Check className="w-3 h-3 text-emerald-600" />
                <span className="text-emerald-600">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3 text-zinc-400" />
                <span>Copy</span>
              </>
            )}
          </span>
          {collapsed
            ? <ChevronRight className="w-4 h-4 text-zinc-400 group-hover:text-zinc-700 transition-colors" />
            : <ChevronDown className="w-4 h-4 text-zinc-400 group-hover:text-zinc-700 transition-colors" />
          }
        </div>
      </button>

      {/* Code Body */}
      {!collapsed && (
        <div className="overflow-x-auto bg-[#FBFBFC]">
          <table className="w-full border-collapse font-mono text-[12px] leading-[1.7]">
            <tbody>
              {lines.map((line, lineIdx) => (
                <tr
                  key={lineIdx}
                  className="hover:bg-zinc-100/60 transition-colors group/row"
                >
                  <td className="w-10 pr-3 pl-3 text-right select-none text-zinc-400 text-[10.5px] border-r border-zinc-200/70 group-hover/row:text-zinc-600 transition-colors bg-zinc-50/40">
                    {String(lineIdx + 1).padStart(2, '0')}
                  </td>
                  <td className="pl-4 pr-4 whitespace-pre text-zinc-800">
                    {highlightPythonLine(line)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Collapsed preview pill */}
      {collapsed && (
        <div className="px-4 py-2 text-[11px] font-mono text-zinc-400 italic bg-zinc-50 border-t border-zinc-200/60">
          {lines[0]?.trim().slice(0, 80)}{lines[0]?.trim().length > 80 ? '…' : ''}
        </div>
      )}
    </div>
  );
}

export const CodeWalkthroughTab: React.FC<CodeWalkthroughTabProps> = ({ paper }) => {
  const { codeSnippets } = paper;
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);

  const handleCopy = useCallback((code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  }, []);

  const handleCopyAll = useCallback(() => {
    if (!codeSnippets?.length) return;
    const combined = codeSnippets
      .map((s) => `# ── ${s.title} ──\n${sanitizeCode(s.code)}`)
      .join('\n\n\n');
    navigator.clipboard.writeText(combined);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  }, [codeSnippets]);

  // Honest empty state
  if (!codeSnippets || codeSnippets.length === 0) {
    return (
      <div className="p-6 md:p-8 animate-fadeIn max-w-2xl mx-auto pb-52">
        <div className="rounded-xl border border-zinc-200 bg-white p-10 text-center space-y-4 shadow-xs">
          <div className="w-12 h-12 rounded-xl bg-zinc-100 text-zinc-500 flex items-center justify-center mx-auto">
            <Code className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h4 className="font-bold text-zinc-950 text-sm font-mono uppercase tracking-wide">
              No reference code for this paper
            </h4>
            <p className="text-xs text-zinc-500 leading-relaxed max-w-sm mx-auto">
              {paper.codeNote || 'The extraction found no implementable architecture to turn into reference code.'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8 space-y-5 animate-fadeIn max-w-4xl mx-auto pb-52">
      
      {/* Tab Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-zinc-900 text-white flex items-center justify-center shadow-xs shrink-0">
            <Code className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h4 className="font-bold text-zinc-950 text-xs font-mono uppercase tracking-wide">
              Model Implementation
            </h4>
            <p className="text-[11px] text-zinc-500 truncate">
              Reference architecture implementation — review before use
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-100 border border-zinc-200 text-zinc-600 text-[11px] font-mono">
            <FileCode2 className="w-3.5 h-3.5" />
            <span>Python 3.10+ / PyTorch 2.x</span>
          </div>

          {/* Copy All button */}
          <button
            onClick={handleCopyAll}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-white text-[11px] font-mono font-medium transition-all active:scale-95 shadow-xs cursor-pointer"
            title={`Copy all ${codeSnippets.length} snippet${codeSnippets.length > 1 ? 's' : ''}`}
          >
            {copiedAll ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied all</span>
              </>
            ) : (
              <>
                <ClipboardList className="w-3.5 h-3.5" />
                <span>Copy all</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Snippet count badge */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-400 font-semibold">
          {codeSnippets.length} snippet{codeSnippets.length > 1 ? 's' : ''}
        </span>
        <div className="flex-1 h-px bg-zinc-200/80" />
      </div>

      {/* Code Snippet Cards */}
      <div className="space-y-4">
        {codeSnippets.map((snippet, idx) => (
          <SnippetCard
            key={idx}
            snippet={snippet}
            idx={idx}
            copiedIdx={copiedIdx}
            onCopy={handleCopy}
          />
        ))}
      </div>

      {/* Footer disclaimer */}
      <p className="text-[11px] text-zinc-400 font-mono text-center pt-2">
        AI-generated reference — verify correctness against the paper before running in production.
      </p>

    </div>
  );
};
