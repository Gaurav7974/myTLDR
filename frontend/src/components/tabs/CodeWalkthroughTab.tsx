'use client';

import React, { useState } from 'react';
import { PaperData } from '@/types/paper';
import { Code, Copy, Check, Terminal, FileCode2, Layers } from 'lucide-react';

interface CodeWalkthroughTabProps {
  paper: PaperData;
}

// Lightweight syntax tokenizer for Python / PyTorch
function highlightPythonLine(line: string): React.ReactNode {
  // Check for full line comments
  if (line.trim().startsWith('#')) {
    return <span className="text-zinc-500 italic">{line}</span>;
  }

  // Tokenize regex
  const tokenRegex = /(#.*$)|(\b(?:def|class|import|from|as|return|if|else|elif|for|in|not|is|None|True|False)\b)|(\b(?:torch|nn|math|Tensor|F|scaled_dot_product_attention|matmul|sqrt|softmax|transpose|size|masked_fill)\b)|(["'].*?["'])|(\b\d+(?:\.\d+)?\b)|([()[\],:=+*/-])/g;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(line)) !== null) {
    // Push preceding plain text
    if (match.index > lastIndex) {
      parts.push(line.slice(lastIndex, match.index));
    }

    const [full, comment, keyword, builtin, string, number, operator] = match;

    if (comment) {
      parts.push(<span key={match.index} className="text-zinc-500 italic">{comment}</span>);
    } else if (keyword) {
      parts.push(<span key={match.index} className="text-purple-400 font-semibold">{keyword}</span>);
    } else if (builtin) {
      parts.push(<span key={match.index} className="text-blue-400 font-medium">{builtin}</span>);
    } else if (string) {
      parts.push(<span key={match.index} className="text-amber-300">{string}</span>);
    } else if (number) {
      parts.push(<span key={match.index} className="text-emerald-400 font-mono">{number}</span>);
    } else if (operator) {
      parts.push(<span key={match.index} className="text-zinc-400">{operator}</span>);
    }

    lastIndex = match.index + full.length;
  }

  if (lastIndex < line.length) {
    parts.push(line.slice(lastIndex));
  }

  return parts;
}

export const CodeWalkthroughTab: React.FC<CodeWalkthroughTabProps> = ({ paper }) => {
  const { codeSnippets } = paper;
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);

  const handleCopy = (code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  return (
    <div className="p-6 md:p-8 space-y-6 animate-fadeIn max-w-4xl mx-auto pb-36 font-sans">
      
      {/* Tab Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-zinc-900 text-white flex items-center justify-center shadow-xs">
            <Code className="w-4 h-4" />
          </div>
          <div>
            <h4 className="font-bold text-zinc-950 text-xs font-mono uppercase">PyTorch Implementation</h4>
            <p className="text-[11px] text-zinc-500">Executable reference tensor implementation with architectural annotations</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-100 border border-zinc-200 text-zinc-700 text-xs font-mono">
          <FileCode2 className="w-3.5 h-3.5 text-zinc-600" />
          <span>Python 3.10+ / PyTorch 2.x</span>
        </div>
      </div>

      {/* Code Snippet Cards */}
      {codeSnippets.map((snippet, idx) => {
        const lines = snippet.code.trim().split('\n');

        return (
          <div key={idx} className="rounded-lg border border-zinc-800 bg-[#09090B] text-zinc-100 overflow-hidden shadow-lg space-y-0">
            
            {/* Code Header Bar */}
            <div className="flex items-center justify-between px-4 py-3 bg-zinc-900/90 border-b border-zinc-800/90">
              <div className="flex items-center gap-2.5">
                <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
                <div className="h-3.5 w-px bg-zinc-700 mx-1"></div>
                <Terminal className="w-3.5 h-3.5 text-zinc-400" />
                <span className="font-mono text-xs font-semibold text-zinc-200">{snippet.title}</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                  {lines.length} lines
                </span>
              </div>

              <button
                onClick={() => handleCopy(snippet.code, idx)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono font-medium transition-all active:scale-95 border border-zinc-700"
              >
                {copiedIdx === idx ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Copy Code</span>
                  </>
                )}
              </button>
            </div>

            {/* Code Body with Line Numbers and Syntax Highlighting */}
            <div className="p-4 font-mono text-xs overflow-x-auto text-zinc-200 leading-relaxed bg-[#0a0a0c]">
              <table className="w-full border-collapse">
                <tbody>
                  {lines.map((line, lineIdx) => {
                    const lineNum = lineIdx + 1;
                    const isHovered = hoveredLine === lineNum;

                    return (
                      <tr
                        key={lineIdx}
                        onMouseEnter={() => setHoveredLine(lineNum)}
                        onMouseLeave={() => setHoveredLine(null)}
                        className={`transition-colors ${isHovered ? 'bg-zinc-800/40' : ''}`}
                      >
                        <td className="w-8 pr-4 text-right select-none text-zinc-600 text-[11px] font-mono border-r border-zinc-800/80">
                          {lineNum < 10 ? `0${lineNum}` : lineNum}
                        </td>
                        <td className="pl-4 whitespace-pre font-mono">
                          {highlightPythonLine(line)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Architectural Callout Banner */}
            <div className="p-3.5 bg-zinc-900/60 border-t border-zinc-800/80 flex items-center justify-between text-[11px] text-zinc-400 font-mono">
              <div className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-zinc-500" />
                <span>Matches Equation 1: Attention(Q, K, V) = softmax(QK^T / √d_k)V</span>
              </div>
              <span className="text-zinc-500">O(N²) Complexity</span>
            </div>

          </div>
        );
      })}

    </div>
  );
};
