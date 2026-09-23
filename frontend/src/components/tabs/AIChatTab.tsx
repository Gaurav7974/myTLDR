'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { PaperData, KeywordDefinition } from '@/types/paper';
import { 
  ArrowUp,
  Sparkles, 
  Copy,
  Check,
  ArrowRight,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import { MathFormula } from './MathDeconstructedTab';
import { InteractiveHighlightedText } from '../InteractiveHighlightedText';

interface AIChatTabProps {
  paper: PaperData;
  initialPrompt?: string | null;
  onPromptHandled?: () => void;
  onAskAI?: (prompt: string) => void;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
}

export interface ChatCitation {
  section_title: string;
  content: string;
  similarity: number;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
  citations?: ChatCitation[];
  indexingInProgress?: boolean;
}

// Locate the paper page holding a citation's topic by matching distinctive
// words against the parsed document. RAG chunks carry no page numbers, so
// this client-side match is what makes source pills hyperlink to the paper.
// Exported for unit testing.
export function findCitationPage(cit: ChatCitation, paper: PaperData): number | null {
  const pages = paper.parsedDocument?.pages;
  if (!pages?.length) return null;
  const tokens = cit.content.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 4).slice(0, 12);
  if (!tokens.length) return null;
  let best: number | null = null;
  let bestHits = 0;
  for (const p of pages) {
    const text = p.markdown.toLowerCase();
    const hits = tokens.filter(t => text.includes(t)).length;
    if (hits > bestHits) { bestHits = hits; best = p.pageNumber; }
  }
  return bestHits >= 2 ? best : null;
}

// Formatter to render Markdown bold, inline code, and KaTeX math in AI responses
function FormattedMessageContent({
  text,
  keywords,
  onAskAI,
  onJumpToPage,
}: {
  text: string;
  keywords?: KeywordDefinition[];
  onAskAI?: (prompt: string) => void;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
}) {
  const blocks = text.split('\n\n');

  return (
    <div className="space-y-3 leading-relaxed text-[13px] text-zinc-800">
      {blocks.map((block, bIdx) => {
        const trimmed = block.trim();
        if (!trimmed) return null;

        // KaTeX display math block $$...$$
        if (trimmed.startsWith('$$') && trimmed.endsWith('$$')) {
          const latex = trimmed.slice(2, -2).trim();
          return (
            <div key={bIdx} className="my-2.5 p-3.5 bg-zinc-50 border border-zinc-200/90 rounded-lg overflow-x-auto text-center shadow-2xs">
              <MathFormula latex={latex} displayMode={true} />
            </div>
          );
        }

        // Regular paragraph with inline math $...$, bold **text**, and `code`
        const lines = trimmed.split('\n');
        return (
          <div key={bIdx} className="space-y-1.5">
            {lines.map((line, lIdx) => {
              const parts = line.split(/(\$\$[\s\S]*?\$\$|\$.*?\$|\*\*.*?\*\*|`.*?`)/g);

              return (
                <p key={lIdx} className="leading-relaxed">
                  {parts.map((part, pIdx) => {
                    if (part.startsWith('$$') && part.endsWith('$$')) {
                      return <MathFormula key={pIdx} latex={part.slice(2, -2)} displayMode={true} />;
                    }
                    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
                      return <MathFormula key={pIdx} latex={part.slice(1, -1)} displayMode={false} className="inline" />;
                    }
                    if (part.startsWith('**') && part.endsWith('**')) {
                      return (
                        <strong key={pIdx} className="font-bold text-zinc-950">
                          {part.slice(2, -2)}
                        </strong>
                      );
                    }
                    if (part.startsWith('`') && part.endsWith('`')) {
                      return (
                        <code
                          key={pIdx}
                          className="px-1.5 py-0.5 rounded bg-zinc-100 font-mono text-[11.5px] text-zinc-900 border border-zinc-200/80 font-medium"
                        >
                          {part.slice(1, -1)}
                        </code>
                      );
                    }
                    return (
                      <InteractiveHighlightedText
                        key={pIdx}
                        text={part}
                        keywords={keywords ?? []}
                        onAskAI={onAskAI}
                        onJumpToPage={onJumpToPage}
                      />
                    );
                  })}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

export const AIChatTab: React.FC<AIChatTabProps> = ({ paper, initialPrompt, onPromptHandled, onAskAI, onJumpToPage }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const copyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMsgId(id);
    setTimeout(() => setCopiedMsgId(null), 2000);
  };

  const executeMessage = useCallback(async (userText: string) => {
    if (!userText.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: userText.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      // Always use contentHash — this is the key the backend indexed chunks under.
      // Fall back to arxivId so demo papers share one cache entry; when no
      // hash exists at all, also send paper text so the backend can run
      // on-the-fly BM25 instead of returning "still being indexed".
      const paperHash = paper.metadata.contentHash || paper.metadata.arxivId || '';
      if (!paperHash) {
        console.warn('[AIChatTab] contentHash missing from paper metadata — RAG will use BM25 fallback only');
      }
      const fullText = (
        paper.markdownContent ||
        paper.parsedDocument?.pages?.map((p) => p.markdown).join('\n\n') ||
        ''
      ).slice(0, 60000) || undefined;

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_hash: paperHash || '',
          paper_title: paper.metadata.title,
          query: userText.trim(),
          // Standard {role, content} format — backend also accepts legacy {sender, text}
          history: messages.slice(-6).map((m) => ({
            role: m.sender === 'user' ? 'user' : 'assistant',
            content: m.text,
          })),
          difficulty: 'intermediate',
          ...(fullText ? { full_text: fullText } : {}),
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: data.answer || 'Could not generate a response.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        citations: data.citations || [],
        indexingInProgress: data.indexing_in_progress === true,
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Unknown error';
      const fallbackMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: `Connection error: ${errMsg}. Please ensure the backend is running and retry.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, fallbackMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages, paper.metadata.contentHash, paper.metadata.title]);

  // Handle incoming initial prompt from bottom bar or summary tab
  useEffect(() => {
    if (initialPrompt && initialPrompt.trim()) {
      const promptToRun = initialPrompt.trim();
      const timer = setTimeout(() => {
        executeMessage(promptToRun);
        onPromptHandled?.();
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [initialPrompt, executeMessage, onPromptHandled]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    executeMessage(input);
  };

  const starterCards = [
    {
      title: 'Executive Summary',
      prompt: `Explain ${paper.metadata.title} in 2 minutes for a researcher.`
    },
    {
      title: 'Core Bottleneck Solved',
      prompt: 'What fundamental limitation of prior architectures does this paper solve?'
    },
    {
      title: 'Math & Tensor Shapes',
      prompt: 'Break down the mathematical formulation and tensor shapes step-by-step.'
    },
    {
      title: 'Engineering Tradeoffs',
      prompt: 'What are the main hardware limitations and compute tradeoffs mentioned by the authors?'
    }
  ];

  return (
    <div className="flex flex-col h-full bg-[#FAFAFA] animate-fadeIn relative overflow-hidden">
      
      {/* Scrollable Conversation Thread */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 pt-6 pb-36 space-y-6">
        <div className="max-w-3xl mx-auto w-full space-y-6">

          {/* Empty State / Centered Welcome Greeting */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[48vh] text-center px-4 py-8 space-y-4 animate-fadeIn max-w-lg mx-auto">
              <div className="w-12 h-12 rounded-2xl bg-zinc-950 text-white flex items-center justify-center shadow-md">
                <Sparkles className="w-6 h-6 text-zinc-100" />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-xl font-bold text-zinc-950 tracking-tight">
                  Deep dive with AI Assistant
                </h3>
                <p className="text-xs text-zinc-500 leading-relaxed max-w-md">
                  Ask anything about the architecture, mathematical formulations, or experimental results in <span className="font-semibold text-zinc-800">{paper.metadata.title}</span>.
                </p>
              </div>

              {/* Clean starter prompt pills */}
              <div className="flex flex-wrap items-center justify-center gap-2 pt-2 max-w-md">
                {starterCards.map((card, idx) => (
                  <button
                    key={idx}
                    onClick={() => executeMessage(card.prompt)}
                    className="px-3.5 py-1.5 rounded-full bg-white hover:bg-zinc-50 border border-zinc-200 text-xs text-zinc-700 font-medium transition-all shadow-2xs hover:border-zinc-300 active:scale-95 cursor-pointer"
                  >
                    {card.title}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Render Messages */}
          {messages.map((msg) => {
            const cleanText = msg.text.replace(/^Regarding\s+(\*\*"[^\n*]+"(\*\*)|"[^\n"]+")\s+in\s+\*[^\n*]+\*:\s*/i, '').trim();

            return (
              <div key={msg.id} className="space-y-2">
                {msg.sender === 'user' ? (
                  /* USER MESSAGE: Right-Aligned Pill */
                  <div className="flex justify-end">
                    <div className="max-w-2xl px-4 py-2.5 rounded-2xl rounded-tr-sm bg-zinc-950 text-white text-[13px] leading-relaxed shadow-sm">
                      {msg.text}
                    </div>
                  </div>
                ) : (
                  /* AI MESSAGE */
                  <div className="py-1 space-y-2.5 text-left max-w-3xl">
                    
                    {/* Slim message header */}
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-[11px] text-zinc-950 font-mono tracking-wider">myTLDR AI</span>
                      <span className="text-[10px] font-mono text-zinc-400">• {msg.timestamp}</span>
                    </div>

                    {/* Indexing-in-progress notice */}
                    {msg.indexingInProgress && (
                      <div className="flex items-start gap-2 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-[12px]">
                        <RefreshCw className="w-3.5 h-3.5 mt-0.5 shrink-0 animate-spin" />
                        <div>
                          <span className="font-semibold">Index may still be building</span>
                          <span className="text-amber-700"> — wait 10–15 seconds and ask again for better results.</span>
                        </div>
                      </div>
                    )}

                    {/* Message Body with KaTeX Math Rendering */}
                    <FormattedMessageContent
                      text={cleanText}
                      keywords={paper.metadata.keywords}
                      onAskAI={onAskAI}
                      onJumpToPage={onJumpToPage}
                    />

                    {/* Source pills — positioned below the response text */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="pt-2.5 pb-1 space-y-1.5">
                        <span className="text-zinc-400 font-mono text-[10px] uppercase tracking-wider block font-semibold">
                          Sources:
                        </span>
                        <div className="flex items-center gap-1.5 flex-wrap">
                          {msg.citations.map((cit, citIdx) => {
                            const page = findCitationPage(cit, paper);
                            const simPct = Math.round(cit.similarity * 100);
                            const pill = "px-2.5 py-1 rounded-md bg-white border border-zinc-200 text-[11px] text-zinc-700 font-mono font-medium max-w-[260px] shadow-2xs transition-all";
                            return page != null && onJumpToPage ? (
                              <button
                                key={citIdx}
                                onClick={() => onJumpToPage(page, cit.section_title)}
                                title={`Open page ${page} — similarity ${simPct}%`}
                                className={`${pill} hover:bg-zinc-100 hover:border-zinc-300 cursor-pointer flex items-center gap-1`}
                              >
                                <span className="text-zinc-400">§</span>
                                <span className="truncate">{cit.section_title}</span>
                                <span className="text-[10px] text-zinc-400 ml-1">p.{page}</span>
                                <span className="text-[10px] text-zinc-300 ml-0.5">{simPct}%</span>
                              </button>
                            ) : (
                              <span key={citIdx} title={`Passage cited — similarity ${simPct}%`} className={`${pill} cursor-default flex items-center gap-1`}>
                                <span className="text-zinc-400">§</span>
                                <span className="truncate">{cit.section_title}</span>
                                <span className="text-[10px] text-zinc-300 ml-1">{simPct}%</span>
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Action Bar Below Message */}
                    <div className="pt-2 flex items-center gap-2 flex-wrap text-[11px]">
                      <button
                        onClick={() => copyMessage(msg.id, cleanText)}
                        className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-white hover:bg-zinc-100 border border-zinc-200 text-zinc-700 font-mono transition-colors shadow-2xs"
                        title="Copy response"
                      >
                        {copiedMsgId === msg.id ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-600" />
                            <span className="text-emerald-700 font-medium">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3 text-zinc-500" />
                            <span>Copy response</span>
                          </>
                        )}
                      </button>

                      <span className="text-zinc-300">|</span>

                      <button
                        onClick={() => executeMessage(`Explain this simpler with a real-world analogy.`)}
                        className="px-2.5 py-1 rounded-md bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium transition-colors"
                      >
                        Explain simpler
                      </button>
                      <button
                        onClick={() => executeMessage(`Show the exact PyTorch implementation for this.`)}
                        className="px-2.5 py-1 rounded-md bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium transition-colors"
                      >
                        Show PyTorch code
                      </button>
                    </div>

                  </div>
                )}
              </div>
            );
          })}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex items-center gap-2 py-2 animate-pulse">
              <Sparkles className="w-3.5 h-3.5 text-zinc-400 animate-spin" />
              <span className="text-[11px] text-zinc-500 font-mono">
                Synthesizing grounded response…
              </span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Floating Island Composer Capsule */}
      <div className="absolute bottom-0 left-0 right-0 z-30 pointer-events-none pt-14 pb-4 px-4 md:px-6 bg-gradient-to-t from-[#FAFAFA] via-[#FAFAFA]/90 via-50% to-transparent backdrop-blur-[2px]">
        <div className="max-w-3xl mx-auto w-full space-y-2 pointer-events-auto">
          
          {/* Paper Keyword Chips */}
          {paper.metadata.keywords && paper.metadata.keywords.length > 0 && (
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 scrollbar-none px-1">
              <span className="text-[10px] font-mono uppercase font-bold text-zinc-400 shrink-0">
                Topics:
              </span>
              {paper.metadata.keywords.map((kw) => (
                <button
                  key={kw.id}
                  onClick={() => executeMessage(`Explain "${kw.term.replace(/^#/, '')}" in detail with its core intuition and mathematical formulation.`)}
                  className="px-2.5 py-1 rounded-full bg-white/95 hover:bg-white border border-zinc-200/90 text-[11px] text-zinc-700 font-medium whitespace-nowrap transition-all shadow-xs hover:shadow-sm active:scale-95 shrink-0 backdrop-blur-md"
                >
                  #{kw.term.replace(/^#/, '')}
                </button>
              ))}
            </div>
          )}

          {/* Floating Input Capsule Bar */}
          <form onSubmit={handleSend} className="relative flex items-center bg-white/95 backdrop-blur-xl border border-zinc-200/90 shadow-xl shadow-zinc-950/8 rounded-2xl p-1.5 transition-all focus-within:border-zinc-950/70 focus-within:ring-2 focus-within:ring-zinc-950/10 focus-within:shadow-2xl">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Ask any question about ${paper.metadata.title}...`}
              className="w-full pl-3.5 pr-12 py-2 bg-transparent text-xs md:text-sm text-zinc-900 focus:outline-none placeholder:text-zinc-400 font-normal"
            />
            <div className="absolute right-2 flex items-center">
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="w-8 h-8 rounded-full bg-zinc-950 hover:bg-zinc-800 disabled:opacity-20 text-white flex items-center justify-center transition-all shadow-xs active:scale-95"
                title="Send Message"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </form>

        </div>
      </div>

    </div>
  );
};
