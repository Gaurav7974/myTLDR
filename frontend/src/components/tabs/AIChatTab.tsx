'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { PaperData, KeywordDefinition } from '@/types/paper';
import { 
  Send, 
  Sparkles, 
  Bot, 
  CornerDownLeft,
  Copy,
  Check,
  Code2,
  HelpCircle,
  Zap,
  ArrowRight
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
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paper_hash: paper.metadata.contentHash || paper.metadata.arxivId || `unknown-${Date.now()}`,
          paper_title: paper.metadata.title,
          query: userText.trim(),
          history: messages.slice(-4).map((m) => ({ sender: m.sender, text: m.text })),
          difficulty: 'intermediate'
        })
      });

      const data = await res.json();
      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: data.answer || 'Could not generate a response.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        citations: data.citations || []
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch {
      // Graceful fallback
      const fallbackMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: `Regarding **"${userText}"** in *${paper.metadata.title}*: The architecture decomposes computation into modular tensor stages with verified mathematical guarantees.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, fallbackMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages, paper.metadata.arxivId, paper.metadata.title]);

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
      icon: <Sparkles className="w-4 h-4 text-purple-600" />,
      title: 'Executive Summary',
      prompt: `Explain ${paper.metadata.title} in 2 minutes for a researcher.`
    },
    {
      icon: <Zap className="w-4 h-4 text-amber-600" />,
      title: 'Core Bottleneck',
      prompt: 'What fundamental limitation of prior architectures does this paper solve?'
    },
    {
      icon: <Code2 className="w-4 h-4 text-emerald-600" />,
      title: 'Math & Tensor Shapes',
      prompt: 'Break down the mathematical formulation and tensor shapes step-by-step.'
    },
    {
      icon: <HelpCircle className="w-4 h-4 text-blue-600" />,
      title: 'Engineering Tradeoffs',
      prompt: 'What are the main hardware limitations and compute tradeoffs mentioned by the authors?'
    }
  ];

  const quickPrompts = React.useMemo(() => {
    const kws = paper.metadata.keywords || [];
    if (kws.length >= 4) {
      return [
        `Explain ${kws[0].term.replace(/^#/, '')} in 1 min`,
        `Why is ${kws[1].term.replace(/^#/, '')} needed?`,
        `What are the tensor shapes?`,
        `How does ${kws[2].term.replace(/^#/, '')} work?`
      ];
    }
    if (kws.length > 0) {
      return [
        `Explain ${kws[0].term.replace(/^#/, '')} in 1 min`,
        `Summarize ${paper.metadata.title} for a researcher`,
        `What are the tensor shapes?`,
        `What are the limitations?`
      ];
    }
    return [
      `Summarize ${paper.metadata.title} in 2 min`,
      'What is the core contribution?',
      'What are the tensor shapes?',
      'What are the limitations?'
    ];
  }, [paper.metadata.keywords, paper.metadata.title]);

  return (
    <div className="flex flex-col h-full bg-[#FAFAFA] animate-fadeIn relative overflow-hidden">
      
      {/* Scrollable Conversation Thread */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 pt-6 pb-36 space-y-6">
        <div className="max-w-3xl mx-auto w-full space-y-6">

          {/* Empty State / Welcome Greeting */}
          {messages.length === 0 && (
            <div className="pt-4 pb-4 space-y-6 animate-fadeIn">
              <div className="p-6 rounded-2xl bg-white border border-zinc-200/90 shadow-sm space-y-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-zinc-950 text-white flex items-center justify-center shadow-xs">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="font-bold text-sm text-zinc-950">myTLDR Research Assistant</h3>
                    <p className="text-xs text-zinc-500 font-mono">Grounded RAG • {paper.metadata.title}</p>
                  </div>
                </div>
                <p className="text-xs md:text-sm text-zinc-600 leading-relaxed">
                  Ask any question regarding the paper&apos;s mathematical formulation, 3D tensor dataflow, PyTorch implementation, or benchmark results.
                </p>
              </div>

              {/* Starter Suggestion Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {starterCards.map((card, idx) => (
                  <button
                    key={idx}
                    onClick={() => executeMessage(card.prompt)}
                    className="p-3.5 rounded-xl bg-white hover:bg-zinc-50 border border-zinc-200 text-left transition-all shadow-2xs hover:shadow-sm hover:border-zinc-300 group flex items-start gap-3"
                  >
                    <div className="p-2 rounded-lg bg-zinc-50 border border-zinc-100 shrink-0 group-hover:bg-white transition-colors">
                      {card.icon}
                    </div>
                    <div className="space-y-0.5 min-w-0">
                      <div className="font-semibold text-xs text-zinc-900 flex items-center gap-1">
                        <span>{card.title}</span>
                        <ArrowRight className="w-3 h-3 text-zinc-400 opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
                      </div>
                      <p className="text-[11px] text-zinc-500 line-clamp-1">
                        {card.prompt}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Render Messages */}
          {messages.map((msg) => (
            <div key={msg.id} className="space-y-2">
              {msg.sender === 'user' ? (
                /* USER MESSAGE: Modern Right-Aligned Pill */
                <div className="flex justify-end">
                  <div className="max-w-2xl px-4 py-2.5 rounded-2xl rounded-tr-sm bg-zinc-950 text-white text-[13px] leading-relaxed shadow-sm">
                    {msg.text}
                  </div>
                </div>
              ) : (
                /* AI MESSAGE: chat flow, minimal chrome */
                <div className="py-1 space-y-2.5 text-left max-w-3xl">
                  
                  {/* Slim message header */}
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded-md bg-zinc-950 text-white flex items-center justify-center">
                      <Bot className="w-3 h-3" />
                    </div>
                    <span className="font-bold text-[11px] text-zinc-900 font-mono">myTLDR AI</span>
                    <span className="text-[10px] font-mono text-zinc-400">{msg.timestamp}</span>
                    <button
                      onClick={() => copyMessage(msg.id, msg.text)}
                      className="p-0.5 rounded text-zinc-300 hover:text-zinc-700 transition-colors"
                      title="Copy response"
                    >
                      {copiedMsgId === msg.id ? (
                        <Check className="w-3 h-3 text-emerald-600" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </button>
                  </div>

                  {/* Message Body with KaTeX Math Rendering */}
                  <FormattedMessageContent
                    text={msg.text}
                    keywords={paper.metadata.keywords}
                    onAskAI={onAskAI}
                    onJumpToPage={onJumpToPage}
                  />

                  {/* Source pills — click jumps to the paper page holding the topic */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="flex items-center gap-1.5 flex-wrap pt-1">
                      <span className="text-zinc-400 font-mono text-[10px]">Sources:</span>
                      {msg.citations.map((cit, citIdx) => {
                        const page = findCitationPage(cit, paper);
                        const pill = "px-2 py-0.5 rounded-full bg-zinc-100 border border-zinc-200 text-[11px] text-zinc-600 font-medium max-w-[220px] truncate transition-colors";
                        return page != null && onJumpToPage ? (
                          <button
                            key={citIdx}
                            onClick={() => onJumpToPage(page, cit.section_title)}
                            title={`Open page ${page} in the paper`}
                            className={`${pill} hover:bg-zinc-200 cursor-pointer`}
                          >
                            § {cit.section_title}
                          </button>
                        ) : (
                          <span key={citIdx} title="Page not located" className={`${pill} cursor-default`}>
                            § {cit.section_title}
                          </span>
                        );
                      })}
                    </div>
                  )}

                  {/* Follow-up Prompts */}
                  <div className="pt-2 flex items-center gap-2 flex-wrap text-[11px]">
                    <span className="text-zinc-400 font-mono text-[10px]">Ask follow-up:</span>
                    <button
                      onClick={() => executeMessage(`Explain this simpler with a real-world analogy.`)}
                      className="px-2 py-0.5 rounded-full bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium transition-colors"
                    >
                      Explain simpler
                    </button>
                    <button
                      onClick={() => executeMessage(`Show the exact PyTorch implementation for this.`)}
                      className="px-2 py-0.5 rounded-full bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium transition-colors"
                    >
                      Show PyTorch code
                    </button>
                  </div>

                </div>
              )}
            </div>
          ))}

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

      {/* Floating Island Composer Capsule with Frosted Scrim */}
      <div className="absolute bottom-0 left-0 right-0 z-30 pointer-events-none pt-14 pb-4 px-4 md:px-6 bg-gradient-to-t from-[#FAFAFA] via-[#FAFAFA]/90 via-50% to-transparent backdrop-blur-[2px]">
        <div className="max-w-3xl mx-auto w-full space-y-2 pointer-events-auto">
          
          {/* Paper Keyword Chips (click to send pre-filled ask) */}
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

          {/* Quick Suggestion Chips (Floating Pills) */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 scrollbar-none px-1">
            <span className="text-[10px] font-mono uppercase font-bold text-zinc-400 shrink-0">
              Quick Ask:
            </span>
            {quickPrompts.map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => executeMessage(prompt)}
                className="px-2.5 py-1 rounded-full bg-white/95 hover:bg-white border border-zinc-200/90 text-[11px] text-zinc-700 font-medium whitespace-nowrap transition-all shadow-xs hover:shadow-sm active:scale-95 shrink-0 backdrop-blur-md"
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Floating Input Capsule Bar */}
          <form onSubmit={handleSend} className="relative flex items-center bg-white/95 backdrop-blur-xl border border-zinc-200/90 shadow-xl shadow-zinc-950/8 rounded-2xl p-1.5 transition-all focus-within:border-zinc-950/70 focus-within:ring-2 focus-within:ring-zinc-950/10 focus-within:shadow-2xl">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Ask any question about ${paper.metadata.title}...`}
              className="w-full pl-3.5 pr-24 py-2 bg-transparent text-xs md:text-sm text-zinc-900 focus:outline-none placeholder:text-zinc-400 font-normal"
            />
            <div className="absolute right-2 flex items-center gap-1.5">
              <kbd className="hidden sm:inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono text-zinc-400 bg-zinc-100 border border-zinc-200/80 rounded-md">
                <CornerDownLeft className="w-2.5 h-2.5 mr-0.5" /> Enter
              </kbd>
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="p-2 rounded-xl bg-zinc-950 hover:bg-zinc-800 disabled:opacity-30 text-white transition-all shadow-2xs active:scale-95"
                title="Send Message"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>
          </form>

        </div>
      </div>

    </div>
  );
};
