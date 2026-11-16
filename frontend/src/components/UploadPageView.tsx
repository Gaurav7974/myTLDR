'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { PaperData } from '@/types/paper';
import { TRANSFORMER_PAPER_DATA, MAMBA_PAPER_DATA, generateDynamicPaperData } from '@/lib/mockData';
import { processPaperStream, lookupArxiv, ArxivLookup } from '@/lib/api';
import { PixelHeaderGrid } from '@/components/PixelHeaderGrid';
import { Reveal } from '@/components/Reveal';
import { Typewriter, TypeOnce } from '@/components/Typewriter';
import { UglyShrimpLoader } from '@/components/UglyShrimpLoader';
import {
  ArrowRight,
  BookOpen,
  FileText,
  MessageSquare,
  Network,
  Paperclip,
} from 'lucide-react';

interface UploadPageViewProps {
  onSelectPaper: (paper: PaperData) => void;
}

const BENCHMARK_PAPERS = [
  {
    id: 'transformer',
    title: 'Attention Is All You Need',
    authors: 'Vaswani et al. (Google Brain)',
    arxivId: '1706.03762',
    citations: '124,500+',
    year: '2017',
    category: 'Transformers',
    description: 'Replaced recurrent neural networks with parallel self-attention, powering GPT-4, Claude and all modern LLMs.',
    badge: 'Benchmark',
    data: TRANSFORMER_PAPER_DATA,
  },
  {
    id: 'mamba',
    title: 'Mamba: Linear-Time Sequence Modeling',
    authors: 'Gu and Dao (CMU / Princeton)',
    arxivId: '2312.00752',
    citations: '3,800+',
    year: '2023',
    category: 'SSM',
    description: 'Selective state spaces computing 5x higher throughput with sub-quadratic O(N) context scaling.',
    badge: 'SSM Architecture',
    data: MAMBA_PAPER_DATA,
  },
  {
    id: 'llama2',
    title: 'Llama 2: Open Foundation Models',
    authors: 'Touvron et al. (Meta AI)',
    arxivId: '2307.09288',
    citations: '18,400+',
    year: '2023',
    category: 'Foundation',
    description: 'Introduced Grouped-Query Attention (GQA), RoPE positional embeddings and SwiGLU FFN at scale.',
    badge: 'Foundation',
    data: generateDynamicPaperData('Llama 2: Open Foundation Models', '2307.09288'),
  },
  {
    id: 'lora',
    title: 'LoRA: Low-Rank Adaptation of LLMs',
    authors: 'Hu et al. (Microsoft Research)',
    arxivId: '2106.09685',
    citations: '24,100+',
    year: '2021',
    category: 'Fine-Tuning',
    description: 'Freezes pretrained model weights and injects trainable rank decomposition matrices A and B.',
    badge: 'Fine-Tuning',
    data: generateDynamicPaperData('LoRA: Low-Rank Adaptation of LLMs', '2106.09685'),
  },
];

const CATEGORIES = ['All', 'Transformers', 'Foundation', 'SSM', 'Fine-Tuning'];

const EXAMPLE_LINKS = [
  { label: 'ATTENTION IS ALL YOU NEED', url: 'https://arxiv.org/abs/1706.03762' },
  { label: 'MAMBA', url: 'https://arxiv.org/abs/2312.00752' },
  { label: 'LLAMA 2', url: 'https://arxiv.org/abs/2307.09288' },
  { label: 'LORA', url: 'https://arxiv.org/abs/2106.09685' },
];

const MARQUEE_ITEMS = [
  'ATTENTION IS ALL YOU NEED',
  'MAMBA',
  'FLASHATTENTION',
  'RESNET',
  'BERT',
  'GPT-3',
  'LLAMA 2',
  'LORA',
  'DIFFUSION',
  'CLIP',
  'WHISPER',
  'ALEXNET',
  'TRANSFORMER',
  'RETENTION',
];

const SOURCES = [
  { name: 'arXiv', slug: 'arxiv' },
  { name: 'Semantic Scholar', slug: 'semanticscholar' },
  { name: 'Hugging Face', slug: 'huggingface' },
  { name: 'PubMed', slug: 'pubmed' },
  { name: 'IEEE', slug: 'ieee' },
  { name: 'Google Scholar', slug: 'googlescholar' },
];

const TRUST_BADGES = ['NO ACCOUNT NEEDED', 'FREE IN BETA', 'BEGINNER → EXPERT', 'NEVER A WRONG DIAGRAM'];

const DEPTHS = [
  { id: 'beginner' as const, label: 'BEGINNER' },
  { id: 'intermediate' as const, label: 'INTERMEDIATE' },
  { id: 'expert' as const, label: 'EXPERT' },
];

function SourceLogo({ name, slug }: { name: string; slug: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <span className="font-mono text-sm font-bold tracking-tight text-ink transition-colors group-hover:text-white">
        {name.toUpperCase()}
      </span>
    );
  }
  return (
    <span className="relative inline-flex h-6 items-center">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`https://cdn.simpleicons.org/${slug}/0A0A0A`}
        alt={name}
        className="h-6 w-auto max-w-[130px] object-contain transition-opacity duration-200 group-hover:opacity-0"
        loading="lazy"
        onError={() => setFailed(true)}
      />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`https://cdn.simpleicons.org/${slug}/FFFFFF`}
        alt=""
        aria-hidden="true"
        className="absolute inset-0 m-auto h-6 w-auto max-w-[130px] object-contain opacity-0 transition-opacity duration-200 group-hover:opacity-100"
        loading="lazy"
        onError={() => setFailed(true)}
      />
    </span>
  );
}

export const UploadPageView: React.FC<UploadPageViewProps> = ({ onSelectPaper }) => {
  const [url, setUrl] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressStatus, setProgressStatus] = useState<string>('');
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [progressStage, setProgressStage] = useState<string>('extracting');
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>('intermediate');
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Smooth progress bar advancement while processing
  useEffect(() => {
    if (!isProcessing) {
      return;
    }
    const timer = setInterval(() => {
      setProgressPercent((prev) => {
        if (progressStage === 'extracting') {
          // Tick to just below the 'explaining' SSE floor (40) so no jump occurs
          if (prev < 38) return prev + 0.6;
        } else if (progressStage === 'explaining') {
          // Tick to just below the 'diagramming' SSE floor (70)
          if (prev < 68) return prev + 0.5;
        } else if (progressStage === 'diagramming') {
          if (prev < 94) return prev + 0.3;
        }
        return prev;
      });
    }, 150);
    return () => clearInterval(timer);
  }, [isProcessing, progressStage]);

  const filteredPapers = useMemo(() => {
    if (selectedCategory === 'All') return BENCHMARK_PAPERS;
    return BENCHMARK_PAPERS.filter((p) => p.category === selectedCategory);
  }, [selectedCategory]);

  // Strict arXiv validator
  const isArxivInput = (val: string): boolean => {
    const t = val.trim();
    if (!t) return false;
    if (/^\d{4}\.\d{4,5}(v\d+)?$/i.test(t)) return true;
    if (/^[a-zA-Z\-]+(\.[a-zA-Z\-]+)?\/\d{7}(v\d+)?$/i.test(t)) return true;
    if (/^(https?:\/\/)?(www\.)?arxiv\.org\/(abs|pdf)\/([a-zA-Z\-]+(\.[a-zA-Z\-]+)?\/\d{7}|\d{4}\.\d{4,5})(v\d+)?(\.pdf)?([?#].*)?$/i.test(t)) {
      return true;
    }
    return false;
  };

  const isInvalidUrl = useMemo(() => {
    const trimmed = url.trim();
    return trimmed.length > 0 && !isArxivInput(trimmed);
  }, [url]);

  // Identity verification: resolve the typed ID against arXiv (debounced) so
  // the user confirms the REAL title before we stream a deconstruction.
  // A typo'd-but-valid ID would otherwise silently decode the wrong paper.
  const [lookup, setLookup] = useState<ArxivLookup | null>(null);
  const [lookupLoading, setLookupLoading] = useState(false);
  const [lookupError, setLookupError] = useState<string | null>(null);

  useEffect(() => {
    const trimmed = url.trim();
    if (!trimmed || !isArxivInput(trimmed) || isProcessing) {
      setLookup(null);
      setLookupError(null);
      setLookupLoading(false);
      return;
    }
    let cancelled = false;
    setLookup(null);
    setLookupError(null);
    setLookupLoading(true);
    const timer = setTimeout(async () => {
      try {
        const meta = await lookupArxiv(trimmed);
        if (!cancelled) {
          setLookup(meta);
          setLookupError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setLookup(null);
          setLookupError(err instanceof Error ? err.message : 'arXiv lookup failed.');
        }
      } finally {
        if (!cancelled) setLookupLoading(false);
      }
    }, 600);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, isProcessing]);

  // Lookup is confirmed only when it resolved AND still matches the input —
  // typing a new ID after verification must re-verify, never reuse.
  const isLookupConfirmed = useMemo(() => {
    if (!lookup || lookupError) return false;
    const trimmed = url.trim();
    return !!trimmed && trimmed.includes(lookup.arxiv_id);
  }, [lookup, lookupError, url]);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim() || isProcessing) return;

    if (!isArxivInput(url)) {
      setSubmitError("Invalid URL: Only arXiv papers are supported (e.g. 1706.03762 or https://arxiv.org/abs/1706.03762). Please try again.");
      return;
    }

    if (!isLookupConfirmed) {
      setSubmitError(
        lookupError
          ? `arXiv lookup failed: ${lookupError} — check the ID for typos and try again.`
          : "Verifying this arXiv ID… please wait for the paper identity check to finish."
      );
      return;
    }

    setIsProcessing(true);
    setSubmitError(null);
    setProgressPercent(1);
    setProgressStage('extracting');
    setProgressStatus('Initializing ingestion pipeline...');

    try {
      const paper = await processPaperStream(
        {
          source: 'arxiv',
          url: url.trim(),
          difficulty,
        },
        (evt) => {
          if (evt.message) setProgressStatus(evt.message);
          if (evt.stage === 'extracting') {
            setProgressStage('extracting');
            // Only nudge forward if we're still at the very start
            setProgressPercent((prev) => Math.max(prev, 10));
          } else if (evt.stage === 'explaining') {
            setProgressStage('explaining');
            setProgressPercent((prev) => Math.max(prev, 40));
          } else if (evt.stage === 'diagramming') {
            setProgressStage('diagramming');
            setProgressPercent((prev) => Math.max(prev, 70));
          } else if (evt.stage === 'done') {
            setProgressStage('done');
            setProgressPercent(100);
          }
        }
      );
      onSelectPaper(paper);
    } catch (err) {
      // Surface the failure — never substitute mock content for a real paper.
      setSubmitError(err instanceof Error ? err.message : 'Processing failed. Please retry.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileProcess = async (file: File) => {
    // One ingestion at a time: ignore file drops while a URL parse is running.
    if (isProcessing) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setSubmitError("Invalid file type: Only PDF research papers (.pdf) are supported. Please upload a .pdf file.");
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setSubmitError("File too large: Maximum supported paper size is 25MB.");
      return;
    }

    setIsProcessing(true);
    setSubmitError(null);
    setProgressPercent(1);
    setProgressStage('extracting');
    setProgressStatus(`Extracting ${file.name} via OCR...`);

    try {
      const paper = await processPaperStream(
        {
          source: 'upload',
          file,
          difficulty,
        },
        (evt) => {
          if (evt.message) setProgressStatus(evt.message);
          if (evt.stage === 'extracting') {
            setProgressStage('extracting');
            setProgressPercent((prev) => Math.max(prev, 10));
          } else if (evt.stage === 'explaining') {
            setProgressStage('explaining');
            setProgressPercent((prev) => Math.max(prev, 40));
          } else if (evt.stage === 'diagramming') {
            setProgressStage('diagramming');
            setProgressPercent((prev) => Math.max(prev, 70));
          } else if (evt.stage === 'done') {
            setProgressStage('done');
            setProgressPercent(100);
          }
        }
      );
      onSelectPaper(paper);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Processing failed. Please retry.');
    } finally {
      setIsProcessing(false);
    }
  };

  const marqueeRow = (hidden: boolean) => (
    <div
      aria-hidden={hidden}
      className="flex shrink-0 items-center gap-8 pr-8"
    >
      {MARQUEE_ITEMS.map((item) => (
        <span key={item} className="flex items-center gap-8 whitespace-nowrap font-mono text-xs font-bold tracking-[0.14em] text-white">
          {item}
          <span className="text-white/50">/</span>
        </span>
      ))}
    </div>
  );

  return (
    <div className="flex min-h-screen flex-col bg-paper-warm font-sans text-ink">
      {/* Hidden file input for PDF uploads */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".pdf,application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileProcess(file);
        }}
      />

      {/* ============ DARK BAND: INTERACTIVE PIXEL GRID HEADER ============ */}
      <PixelHeaderGrid
        onExploreClick={() => {
          const el = document.getElementById('input-zone');
          el?.scrollIntoView({ behavior: 'smooth' });
        }}
        onContactClick={() => {
          const el = document.getElementById('input-zone');
          el?.scrollIntoView({ behavior: 'smooth' });
        }}
      />

      {/* ============ BLUE MARQUEE TICKER ============ */}
      <div className="overflow-hidden border-b-2 border-ink bg-[#0055FF] py-1.5">
        <div className="marquee-track flex w-max">
          {marqueeRow(false)}
          {marqueeRow(true)}
        </div>
      </div>

      {/* ============ HERO: HEADLINE + INPUT ZONE ============ */}
      <section id="input-zone" className="border-b-2 border-ink bg-paper-warm">
        <div className="mx-auto w-full max-w-[1400px] px-4 pb-8 pt-5 md:px-8 md:pb-10 md:pt-6 space-y-6 md:space-y-7">
          
          {/* Split 2-Column Hero Section matching aeye-saas.webflow.io */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-10 items-start">
            {/* Left Headline Column */}
            <div className="lg:col-span-7 space-y-3">
              <h1 className="select-none text-[clamp(2rem,4.5vw,3.8rem)] font-bold tracking-tight text-ink leading-[1.1] animate-slideUp">
                A Next-Gen Powerful
                <br />
                <span className="font-pixel text-[#0055FF] text-[0.58em] sm:text-[0.62em] tracking-normal inline-flex items-center gap-1.5 py-1 whitespace-nowrap">
                  <span>[</span>
                  <Typewriter />
                  <span className="inline-block w-2 sm:w-2.5 h-4 sm:h-5 bg-[#0055FF] animate-pulse ml-0.5" />
                  <span>]</span>
                </span>
                <br />
                for Research Papers.
              </h1>
            </div>
          </div>

          {/* Input + depth + submit */}
          <form id="input-form" onSubmit={handleUrlSubmit} className="max-w-4xl space-y-3 pt-1">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDraggingOver(true);
              }}
              onDragLeave={() => setIsDraggingOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDraggingOver(false);
                if (isProcessing) return;
                const file = e.dataTransfer.files?.[0];
                if (file) handleFileProcess(file);
              }}
              className={`flex items-stretch border-2 bg-white transition-colors ${
                isDraggingOver ? 'border-volt ring-4 ring-volt/20' : 'border-ink'
              }`}
            >
              <span className="flex items-center border-r-2 border-ink bg-ink px-4 font-mono text-sm font-bold text-volt">
                »
              </span>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Paste an arXiv link, e.g. arxiv.org/abs/1706.03762"
                aria-label="Paper URL or arXiv ID"
                disabled={isProcessing}
                className="min-w-0 flex-1 bg-transparent px-4 py-3.5 font-mono text-sm text-ink outline-none placeholder:text-fog disabled:opacity-50"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="flex items-center gap-1.5 border-l-2 border-ink px-3.5 font-mono text-[11px] font-bold tracking-[0.08em] text-ink/70 transition-colors hover:bg-band hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
                title="Attach a PDF research paper"
              >
                <Paperclip className="h-4 w-4" />
                <span className="hidden sm:inline">PDF</span>
              </button>
              <button
                type="submit"
                disabled={!url.trim() || isInvalidUrl || isProcessing || (!!url.trim() && !isInvalidUrl && !isLookupConfirmed)}
                className="flex items-center gap-2 border-l-2 border-ink bg-ink px-5 font-mono text-xs font-bold tracking-[0.14em] text-white transition-colors hover:bg-volt disabled:cursor-not-allowed disabled:opacity-40 md:px-7"
              >
                {isProcessing ? 'PARSING' : 'EXPLAIN'}
                <ArrowRight className={`h-4 w-4 ${isProcessing ? 'animate-pulse' : ''}`} />
              </button>
            </div>

            {/* Invalid URL Warning */}
            {isInvalidUrl && !isProcessing && (
              <div className="flex items-center gap-2 border-2 border-red-500 bg-red-50 px-3.5 py-2.5 text-xs font-mono text-red-700 animate-fadeIn">
                <span className="font-bold shrink-0">⚠ INVALID URL:</span>
                <span>Only arXiv papers are supported (e.g. 1706.03762 or https://arxiv.org/abs/1706.03762). Please provide a valid arXiv ID or URL.</span>
              </div>
            )}

            {/* Verified paper identity (replaces the old ID-echo preview):
                shows the REAL title from arXiv so a typo'd ID can't
                silently deconstruct the wrong paper. */}
            {lookupLoading && !isProcessing && (
              <div className="flex items-center gap-2 border-2 border-ink/20 bg-white px-3.5 py-2.5 font-mono text-xs text-fog animate-fadeIn">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-volt opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-volt"></span>
                </span>
                <span>Verifying arXiv ID…</span>
              </div>
            )}
            {lookupError && !isProcessing && (
              <div className="flex items-center gap-2 border-2 border-red-500 bg-red-50 px-3.5 py-2.5 text-xs font-mono text-red-700 animate-fadeIn">
                <span className="font-bold shrink-0">⚠ ID NOT FOUND:</span>
                <span>{lookupError}</span>
              </div>
            )}
            {/* Modern High-UX Progress Bar */}
            {isProcessing && (
              <div className="border-2 border-ink bg-white p-3.5 space-y-2.5 shadow-2xs animate-fadeIn" role="status">
                {/* Header: Stage and Percent */}
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="flex items-center gap-2 font-bold tracking-[0.08em] text-ink uppercase">
                    <UglyShrimpLoader size="sm" />
                    {progressStage === 'extracting' ? 'EXTRACTING & OCR' :
                     progressStage === 'explaining' ? 'GENERATING EXPLANATION' :
                     progressStage === 'diagramming' ? 'DECONSTRUCTING ARCHITECTURE' :
                     'FINALIZING'}
                  </span>
                  <span className="font-mono font-bold text-volt tracking-wider">
                    {Math.round(progressPercent)}%
                  </span>
                </div>

                {/* Bar Track & Moving Animated Stripes */}
                <div className="h-3.5 w-full border-2 border-ink bg-zinc-100 p-[1px] relative overflow-hidden">
                  <div
                    className="progress-stripes h-full bg-volt transition-all duration-300 ease-out shadow-[0_0_10px_rgba(37,99,235,0.4)]"
                    style={{ width: `${Math.max(6, Math.min(100, progressPercent))}%` }}
                  />
                </div>
              </div>
            )}

            {/* Submission error — single clean error message */}
            {submitError && !isProcessing && (
              <div className="border-2 border-ink bg-red-50 px-4 py-3 animate-fadeIn" role="alert">
                <p className="font-mono text-xs font-bold tracking-[0.06em] text-red-800 flex items-start gap-2">
                  <span className="shrink-0 text-red-600 font-bold">✕</span>
                  <span>{submitError}</span>
                </p>
              </div>
            )}

            {/* Depth selector */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="font-mono text-[11px] font-bold tracking-[0.14em] text-ink/60">DEPTH:</span>
              {DEPTHS.map((d) => (
                <button
                  type="button"
                  key={d.id}
                  onClick={() => setDifficulty(d.id)}
                  className={`border-2 px-3 py-1.5 font-mono text-[11px] font-bold tracking-[0.1em] transition-colors active:translate-y-[1px] ${
                    difficulty === d.id
                      ? 'border-ink bg-volt text-white'
                      : 'border-ink/30 bg-white text-ink/70 hover:border-ink hover:text-ink'
                  }`}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </form>

          {/* Examples + disclaimer */}
          <div className="mt-5 flex max-w-3xl flex-wrap items-center gap-x-2 gap-y-1.5 font-mono text-[11px] text-ink/60 animate-slideUp [animation-delay:480ms]">
            <span className="font-bold tracking-[0.14em] text-ink/80">TRY:</span>
            {EXAMPLE_LINKS.map((ex, i) => (
              <React.Fragment key={ex.label}>
                <button
                  type="button"
                  onClick={() => setUrl(ex.url)}
                  className="underline decoration-ink/30 underline-offset-4 transition-colors hover:decoration-volt hover:text-ink"
                >
                  {ex.label}
                </button>
                {i < EXAMPLE_LINKS.length - 1 && <span className="text-volt">/</span>}
              </React.Fragment>
            ))}
            <span className="w-full pt-1 text-[10px] tracking-[0.06em] text-fog">
              PASTE A LINK OR DROP A PDF. GET THE SUMMARY, THE DIAGRAM, THE EQUATIONS, AND CHAT.
            </span>
          </div>

          {/* Trust badges */}
          <div className="mt-8 flex flex-wrap gap-2 animate-slideUp [animation-delay:600ms]">
            {TRUST_BADGES.map((badge) => (
              <span
                key={badge}
                className="border-2 border-ink bg-ink px-3 py-1.5 font-mono text-[10px] font-bold tracking-[0.14em] text-white"
              >
                {badge}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ============ HOW IT WORKS: NUMBERED CARDS ============ */}
      <section id="how" className="border-b-2 border-ink">
        <div className="mx-auto w-full max-w-[1400px] px-4 py-14 md:px-6 md:py-20">
          <Reveal>
            <h2 className="max-w-3xl text-3xl font-bold uppercase leading-[1.02] tracking-[-0.02em] md:text-5xl">
              EVERYTHING YOU NEED TO <span className="font-mono font-bold text-volt">DECODE</span> A PAPER
            </h2>
          </Reveal>

          <div className="mt-10 grid grid-cols-1 gap-5 md:grid-cols-3 md:gap-6">
            {[
              {
                num: '/001',
                icon: BookOpen,
                title: 'PLAIN-ENGLISH EXPLANATION',
                desc: 'Every section rewritten at beginner, intermediate or expert depth. Beginner gets analogies, experts get tensor shapes. Switch depth anytime.',
              },
              {
                num: '/002',
                icon: Network,
                title: 'VERIFIED ARCHITECTURE DIAGRAM',
                desc: 'The model architecture is extracted and validated block by block. If a diagram cannot be verified, you get text only. Never a wrong diagram.',
              },
              {
                num: '/003',
                icon: MessageSquare,
                title: 'CHAT WITH THE PAPER',
                desc: 'Ask follow-up questions about methods, math or results. Every answer stays grounded in the paper you gave it.',
              },
            ].map((card, i) => (
              <Reveal key={card.num} delay={i * 120}>
                <div
                  className="group flex h-full flex-col border-2 border-ink bg-white p-6 transition-all duration-150 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:hard-shadow"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-bold text-volt">{card.num}</span>
                    <div className="flex h-9 w-9 items-center justify-center border-2 border-ink bg-paper-warm transition-all duration-200 group-hover:rotate-6 group-hover:bg-volt group-hover:text-white">
                      <card.icon className="h-4 w-4" />
                    </div>
                  </div>
                  <h3 className="mt-6 text-base font-bold uppercase tracking-tight">{card.title}</h3>
                  <p className="mt-3 font-mono text-xs leading-relaxed text-ink/60">{card.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ============ CLASSIC PAPERS ============ */}
      <section id="classics" className="border-b-2 border-ink">
        <div className="mx-auto w-full max-w-[1400px] px-4 py-14 md:px-6 md:py-20">
          <Reveal>
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 className="text-3xl font-bold uppercase leading-[1.02] tracking-[-0.02em] md:text-5xl">
                  START WITH A <span className="font-mono font-bold text-volt">CLASSIC</span>
                </h2>
                <p className="mt-3 font-mono text-xs text-ink/60">PRE-EXTRACTED. OPEN ONE INSTANTLY, NO PIPELINE WAIT.</p>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {CATEGORIES.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`border-2 px-2.5 py-1 font-mono text-[10px] font-bold tracking-[0.1em] uppercase transition-colors ${
                      selectedCategory === cat
                        ? 'border-ink bg-ink text-white'
                        : 'border-ink/25 bg-white text-ink/60 hover:border-ink hover:text-ink'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>
          </Reveal>

          <div className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-2">
            {filteredPapers.map((paper, i) => (
              <Reveal key={paper.id} delay={(i % 2) * 100}>
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => !isProcessing && onSelectPaper(paper.data)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      if (!isProcessing) onSelectPaper(paper.data);
                    }
                  }}
                  className={`group flex h-full cursor-pointer flex-col border-2 border-ink bg-white p-5 transition-all duration-150 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:hard-shadow focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#0055FF] ${
                    isProcessing ? 'pointer-events-none opacity-50' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="border border-ink/25 bg-paper-warm px-2 py-0.5 font-mono text-[10px] font-bold tracking-[0.1em] text-ink/70">
                      {paper.badge}
                    </span>
                    <span className="font-mono text-[11px] text-fog">{paper.citations} CITATIONS</span>
                  </div>

                  <h4 className="mt-4 text-lg font-bold leading-tight tracking-tight transition-colors group-hover:text-volt">
                    {paper.title}
                  </h4>
                  <p className="mt-1 font-mono text-[11px] text-ink/60">
                    {paper.authors} / <span className="font-bold">{paper.year}</span>
                  </p>
                  <p className="mt-3 flex-1 font-mono text-xs leading-relaxed text-ink/60">{paper.description}</p>

                  <div className="mt-5 flex items-center justify-between border-t-2 border-ink/10 pt-3">
                    <span className="font-mono text-[11px] font-bold text-ink/70">arXiv:{paper.arxivId}</span>
                    <span className="flex items-center gap-1 font-mono text-[11px] font-bold tracking-[0.1em] text-ink transition-colors group-hover:text-volt">
                      OPEN <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                    </span>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="mt-auto border-t-2 border-zinc-950 bg-[#0D0E11] text-white">
        <div className="mx-auto w-full max-w-[1400px] px-4 py-10 md:px-8 md:py-12">
          <div className="flex flex-col gap-8 md:flex-row md:items-start md:justify-between">
            {/* Identity — one line, no marketing paragraph */}
            <div className="max-w-sm">
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-lg font-bold leading-none text-[#0055FF]">✦</span>
                <span className="font-sans text-lg font-black tracking-tight">myTLDR</span>
                <span className="font-mono text-[11px] text-zinc-500">Paper Decoder</span>
              </div>
              <p className="mt-3 font-mono text-[11px] leading-relaxed text-zinc-500">
                A verified architecture diagram, a plain-English brief, and chat grounded in the paper.
              </p>
            </div>

            {/* Real anchors only */}
            <nav className="flex flex-wrap items-center gap-x-7 gap-y-2 font-mono text-xs text-zinc-400">
              <a href="#how" className="transition-colors hover:text-white">How it works</a>
              <a href="#classics" className="transition-colors hover:text-white">Classics</a>
            </nav>
          </div>

          <div className="mt-8 border-t border-zinc-800/80 pt-5 font-mono text-[11px] text-zinc-600">
            © 2026 myTLDR
          </div>
        </div>
      </footer>
    </div>
  );
};
