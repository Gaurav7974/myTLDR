'use client';

import React, { useState, useCallback } from 'react';
import { PaperData } from '@/types/paper';
import { Code, Copy, Check, Terminal, FileCode2, ChevronDown, ChevronRight, ClipboardList } from 'lucide-react';

interface CodeWalkthroughTabProps {
  paper: PaperData;
}

// Lightweight syntax tokenizer for Python / PyTorch
function highlightPythonLine(line: string): React.ReactNode {
  // Full-line comments
  if (line.trim().startsWith('#')) {
    return <span className="text-[#6B8C6B] italic">{line}</span>;
  }

  const tokenRegex = /(#.*$)|(\b(?:def|class|import|from|as|return|if|else|elif|for|while|in|not|is|None|True|False|with|try|except|finally|raise|yield|async|await|lambda|pass|break|continue)\b)|(\b(?:torch|nn|math|Tensor|F|self|super|Optional|List|Dict|Tuple|Any|Union|scaled_dot_product_attention|matmul|sqrt|softmax|transpose|size|masked_fill|view|reshape|permute|contiguous|zeros|ones|randn|arange|cat|stack|einsum|Linear|Conv2d|LayerNorm|MultiheadAttention|Dropout|ReLU|GELU|Softmax|Parameter|Module|ModuleList|Sequential)\b)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([()[\]{},.:=+\-*/])/g;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(line)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`t-${lastIndex}`}>{line.slice(lastIndex, match.index)}</span>);
    }

    const [full, comment, keyword, builtin, string, number, operator] = match;

    if (comment) {
      parts.push(<span key={match.index} className="text-[#6B8C6B] italic">{comment}</span>);
    } else if (keyword) {
      parts.push(<span key={match.index} className="text-[#CF7ADB] font-semibold">{keyword}</span>);
    } else if (builtin) {
      parts.push(<span key={match.index} className="text-[#79AADB]">{builtin}</span>);
    } else if (string) {
      parts.push(<span key={match.index} className="text-[#D4A96A]">{string}</span>);
    } else if (number) {
      parts.push(<span key={match.index} className="text-[#A8D4A0]">{number}</span>);
    } else if (operator) {
      parts.push(<span key={match.index} className="text-[#7DAFCC]">{operator}</span>);
    }

    lastIndex = match.index + full.length;
  }

  if (lastIndex < line.length) {
    parts.push(<span key={`tail-${lastIndex}`}>{line.slice(lastIndex)}</span>);
  }

  return parts;
}

interface SnippetCardProps {
  snippet: { title: string; language: string; code: string };
  idx: number;
  copiedIdx: number | null;
  onCopy: (code: string, idx: number) => void;
}

function SnippetCard({ snippet, idx, copiedIdx, onCopy }: SnippetCardProps) {
  const [collapsed, setCollapsed] = useState(false);
  const lines = snippet.code.trim().split('\n');

  return (
    <div className="rounded-xl border border-zinc-800 bg-[#0D0D10] overflow-hidden shadow-lg">
      
      {/* Header Bar */}
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center justify-between px-4 py-3 bg-[#131317] border-b border-zinc-800/80 hover:bg-[#18181c] transition-colors group"
      >
        <div className="flex items-center gap-3 min-w-0">
          <Terminal className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span className="font-mono text-[13px] font-semibold text-zinc-100 truncate">{snippet.title}</span>
          <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/80">
            {lines.length} lines
          </span>
          <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-500 border border-zinc-800">
            py
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {/* Copy button — stop propagation so it doesn't collapse */}
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); onCopy(snippet.code, idx); }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); onCopy(snippet.code, idx); } }}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-[11px] font-mono font-medium transition-all active:scale-95 border border-zinc-700 cursor-pointer"
          >
            {copiedIdx === idx ? (
              <>
                <Check className="w-3 h-3 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3 text-zinc-400" />
                <span>Copy</span>
              </>
            )}
          </span>
          {collapsed
            ? <ChevronRight className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
            : <ChevronDown className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
          }
        </div>
      </button>

      {/* Code Body */}
      {!collapsed && (
        <div className="overflow-x-auto bg-[#0D0D10]">
          <table className="w-full border-collapse font-mono text-[12px] leading-[1.65]">
            <tbody>
              {lines.map((line, lineIdx) => (
                <tr
                  key={lineIdx}
                  className="hover:bg-zinc-800/30 transition-colors group/row"
                >
                  <td className="w-10 pr-4 pl-4 text-right select-none text-zinc-600 text-[10.5px] border-r border-zinc-800/60 group-hover/row:text-zinc-500 transition-colors">
                    {String(lineIdx + 1).padStart(2, '0')}
                  </td>
                  <td className="pl-4 pr-4 whitespace-pre text-zinc-200">
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
        <div className="px-4 py-2 text-[11px] font-mono text-zinc-600 italic bg-[#0D0D10] border-t border-zinc-800/40">
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
      .map((s) => `# ── ${s.title} ──\n${s.code.trim()}`)
      .join('\n\n\n');
    navigator.clipboard.writeText(combined);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  }, [codeSnippets]);

  // Honest empty state
  if (!codeSnippets || codeSnippets.length === 0) {
    return (
      <div className="p-6 md:p-8 animate-fadeIn max-w-2xl mx-auto pb-36">
        <div className="rounded-xl border border-zinc-200 bg-white p-10 text-center space-y-4">
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
    <div className="p-6 md:p-8 space-y-5 animate-fadeIn max-w-4xl mx-auto pb-36">
      
      {/* Tab Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-zinc-900 text-white flex items-center justify-center shadow-xs shrink-0">
            <Code className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h4 className="font-bold text-zinc-950 text-xs font-mono uppercase tracking-wide">
              PyTorch Reference Implementation
            </h4>
            <p className="text-[11px] text-zinc-500 truncate">
              AI-generated from this paper&apos;s architecture — review before use
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
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-white text-[11px] font-mono font-medium transition-all active:scale-95 shadow-xs"
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
        <div className="flex-1 h-px bg-zinc-100" />
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
