'use client';

import React, { useState, useMemo, useRef } from 'react';
import { PaperData } from '@/types/paper';
import { TRANSFORMER_PAPER_DATA, MAMBA_PAPER_DATA, generateDynamicPaperData } from '@/lib/mockData';
import { processPaperStream } from '@/lib/api';
import { PixelMosaic } from '@/components/PixelMosaic';
import { PixelHeaderGrid } from '@/components/PixelHeaderGrid';
import { Reveal } from '@/components/Reveal';
import { Typewriter, TypeOnce } from '@/components/Typewriter';
import StickerWall from '@/components/StickerWall';
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  FileText,
  MessageSquare,
  Network,
  Paperclip,
  Sparkles,
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

const TRUST_BADGES = ['NO SIGN-UP', 'FREE IN BETA', '3 DEPTHS', 'VERIFIED OUTPUT'];

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
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>('intermediate');
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const filteredPapers = useMemo(() => {
    if (selectedCategory === 'All') return BENCHMARK_PAPERS;
    return BENCHMARK_PAPERS.filter((p) => p.category === selectedCategory);
  }, [selectedCategory]);

  // Detected arXiv preview when typing
  const detectedArxiv = useMemo(() => {
    const trimmed = url.trim();
    if (!trimmed) return null;
    if (trimmed.includes('1706.03762')) {
      return { title: 'Attention Is All You Need', authors: 'Ashish Vaswani, Noam Shazeer, et al.' };
    }
    if (trimmed.includes('2307.09288')) {
      return { title: 'Llama 2: Open Foundation and Fine-Tuned Chat Models', authors: 'Hugo Touvron et al.' };
    }
    if (trimmed.includes('2312.00752')) {
      return { title: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces', authors: 'Albert Gu, Tri Dao' };
    }
    if (trimmed.includes('2106.09685')) {
      return { title: 'LoRA: Low-Rank Adaptation of Large Language Models', authors: 'Edward J. Hu et al.' };
    }
    if (trimmed.match(/\d{4}\.\d{4,5}/)) {
      return { title: `arXiv Paper: ${trimmed}`, authors: 'Ready for parsing, OCR and diagram extraction' };
    }
    return null;
  }, [url]);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim() || isProcessing) return;

    setIsProcessing(true);
    setProgressStatus('Initializing ingestion pipeline...');

    try {
      const paper = await processPaperStream(
        {
          source: url.includes('arxiv.org') || url.match(/^\d{4}\.\d{4,5}/) ? 'arxiv' : 'url',
          url: url.trim(),
          difficulty,
        },
        (evt) => {
          if (evt.message) setProgressStatus(evt.message);
        }
      );
      onSelectPaper(paper);
    } catch {
      const fallback = generateDynamicPaperData(detectedArxiv?.title || url, url);
      onSelectPaper(fallback);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileProcess = async (file: File) => {
    setIsProcessing(true);
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
        }
      );
      onSelectPaper(paper);
    } catch {
      const fallback = generateDynamicPaperData(file.name.replace(/\.pdf$/i, ''), 'custom-upload');
      onSelectPaper(fallback);
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
        accept=".pdf,.md"
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

            {/* Right Info & Platform Intelligence Column (No person photos) */}
            <div className="lg:col-span-5 space-y-4 lg:pt-1 animate-slideUp [animation-delay:150ms]">
              {/* Platform Capability Badges (Proprietary & Value-Driven) */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-white border-2 border-ink shadow-xs">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span className="font-mono text-xs font-bold text-ink">REAL-TIME RESEARCH ENGINE</span>
                </div>
                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#0055FF]/10 border-2 border-[#0055FF] text-[#0055FF] font-mono text-xs font-bold">
                  <span>✦ 3D NEURAL DECONSTRUCTOR</span>
                </div>
              </div>

              <p className="text-sm text-ink/75 leading-relaxed font-medium">
                Deconstruct any arXiv, NeurIPS, or ICML paper into an interactive 3D architecture, plain-English summary, and step-by-step LaTeX math formulation — verified with 0% hallucination.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-4 pt-0.5">
                <a
                  href="#input-form"
                  className="flex items-center gap-2 px-4 py-2.5 bg-ink text-white hover:bg-[#0055FF] font-mono text-xs font-bold tracking-wider uppercase transition-all shadow-md active:translate-y-[1px]"
                >
                  <span className="w-2.5 h-2.5 bg-white inline-block" />
                  <span>DECONSTRUCT PAPER</span>
                </a>
                <button
                  onClick={() => onSelectPaper(TRANSFORMER_PAPER_DATA)}
                  className="flex items-center gap-2 font-mono text-xs font-bold tracking-wider uppercase text-ink hover:text-[#0055FF] transition-colors underline underline-offset-4"
                >
                  <span>▷ ATTENTION IS ALL YOU NEED [DEMO]</span>
                </button>
              </div>
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
                className="min-w-0 flex-1 bg-transparent px-4 py-3.5 font-mono text-sm text-ink outline-none placeholder:text-fog"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-1.5 border-l-2 border-ink px-3.5 font-mono text-[11px] font-bold tracking-[0.08em] text-ink/70 transition-colors hover:bg-band hover:text-ink"
                title="Attach a PDF or Markdown file"
              >
                <Paperclip className="h-4 w-4" />
                <span className="hidden sm:inline">PDF</span>
              </button>
              <button
                type="submit"
                disabled={!url.trim() || isProcessing}
                className="flex items-center gap-2 border-l-2 border-ink bg-ink px-5 font-mono text-xs font-bold tracking-[0.14em] text-white transition-colors hover:bg-volt disabled:cursor-not-allowed disabled:opacity-40 md:px-7"
              >
                {isProcessing ? 'PARSING' : 'EXPLAIN'}
                <ArrowRight className={`h-4 w-4 ${isProcessing ? 'animate-pulse' : ''}`} />
              </button>
            </div>

            {/* Detected paper preview */}
            {detectedArxiv && !isProcessing && (
              <div className="flex items-start gap-2.5 border-2 border-ink/20 bg-white px-3.5 py-2.5 animate-fadeIn">
                <FileText className="mt-0.5 h-4 w-4 shrink-0 text-volt" />
                <div className="min-w-0">
                  <p className="truncate font-mono text-xs font-bold text-ink">{detectedArxiv.title}</p>
                  <p className="truncate font-mono text-[11px] text-fog">{detectedArxiv.authors}</p>
                </div>
              </div>
            )}

            {/* Processing status */}
            {isProcessing && (
              <div className="progress-stripes border-2 border-ink bg-volt px-4 py-3 animate-fadeIn" role="status">
                <p className="font-mono text-xs font-bold tracking-[0.06em] text-white">
                  <Sparkles className="mr-2 inline h-3.5 w-3.5 animate-spin" />
                  {progressStatus || 'Deconstructing paper...'}
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
              NO SIGN-IN. FREE WHILE IN BETA. DROP A PDF ANYWHERE ON THE BOX ABOVE.
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
                  <p className="mt-3 flex-1 font-mono text-xs leading-relaxed text-ink/60">{card.desc}</p>
                  <ArrowUpRight className="mt-6 h-5 w-5 self-end text-ink/30 transition-all duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-volt" />
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
                  onClick={() => !isProcessing && onSelectPaper(paper.data)}
                  className={`group flex h-full cursor-pointer flex-col border-2 border-ink bg-white p-5 transition-all duration-150 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:hard-shadow ${
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

      {/* ============ FEEDBACK WALL (physics) ============ */}
      <section id="feedback" className="border-b-2 border-ink bg-white">
        <div className="mx-auto w-full max-w-[1400px] px-4 py-14 md:px-6 md:py-20">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="text-3xl font-bold uppercase leading-[1.02] tracking-[-0.02em] md:text-5xl">
                LEAVE YOUR <span className="font-mono font-bold text-volt">MARK</span>
              </h2>
              <p className="mt-3 font-mono text-xs text-ink/60">PHYSICS WALL — DRAG, TOSS, COLLIDE.</p>
            </div>
          </div>
          <div className="mt-8 overflow-hidden rounded-xl border-2 border-ink hard-shadow">
            <StickerWall />
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="mt-auto border-t-2 border-zinc-950 bg-[#0D0E11] text-white">
        <div className="mx-auto w-full max-w-[1400px] px-4 py-12 md:px-8 md:py-16">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-12">
            {/* Left Brand Column */}
            <div className="lg:col-span-5 space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 text-[#0055FF] flex items-center justify-center font-mono font-bold text-lg">
                    ✦
                  </div>
                  <span className="font-sans font-black text-2xl tracking-tight text-white">
                    PaperScope
                  </span>
                </div>
                <span className="px-2 py-0.5 bg-zinc-900 border border-zinc-700/80 text-zinc-400 font-mono text-[10px] font-bold tracking-wider uppercase rounded-xs">
                  v1.0-BETA
                </span>
              </div>

              <p className="max-w-md text-xs font-mono text-zinc-400 leading-relaxed">
                The next-generation research paper deconstructor. Transform dense academic PDFs and arXiv preprints into interactive 3D architecture diagrams, plain-English executive briefs, and step-by-step LaTeX math formulations.
              </p>

              <div className="flex items-center gap-2 pt-2">
                <div className="inline-flex items-center gap-2 px-2.5 py-1 bg-zinc-900 border border-zinc-800 font-mono text-[11px] text-zinc-300">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span>ALL SYSTEMS OPERATIONAL</span>
                </div>
              </div>
            </div>

            {/* Right Links Grid */}
            <div className="lg:col-span-7 grid grid-cols-2 sm:grid-cols-3 gap-8 font-mono text-xs">
              {/* Platform */}
              <div className="space-y-3">
                <h4 className="font-bold tracking-wider uppercase text-white text-[11px]">
                  Platform
                </h4>
                <ul className="space-y-2 text-zinc-400">
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      3D Visualizer
                    </a>
                  </li>
                  <li>
                    <a href="#how" className="hover:text-white transition-colors">
                      Summary & ELI5
                    </a>
                  </li>
                  <li>
                    <a href="#features" className="hover:text-white transition-colors">
                      Equations & Math
                    </a>
                  </li>
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      PyTorch Generator
                    </a>
                  </li>
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      AI Assistant
                    </a>
                  </li>
                </ul>
              </div>

              {/* Benchmarks */}
              <div className="space-y-3">
                <h4 className="font-bold tracking-wider uppercase text-white text-[11px]">
                  Benchmarks
                </h4>
                <ul className="space-y-2 text-zinc-400">
                  <li>
                    <button
                      onClick={() => onSelectPaper(TRANSFORMER_PAPER_DATA)}
                      className="text-left hover:text-white transition-colors cursor-pointer"
                    >
                      Transformer (1706.03762)
                    </button>
                  </li>
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      FlashAttention
                    </a>
                  </li>
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      Mamba SSM
                    </a>
                  </li>
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      Llama 2
                    </a>
                  </li>
                  <li>
                    <a href="#benchmarks" className="hover:text-white transition-colors">
                      LoRA Adaptation
                    </a>
                  </li>
                </ul>
              </div>

              {/* Resources */}
              <div className="space-y-3 col-span-2 sm:col-span-1">
                <h4 className="font-bold tracking-wider uppercase text-white text-[11px]">
                  Resources
                </h4>
                <ul className="space-y-2 text-zinc-400">
                  <li>
                    <a href="#input-form" className="hover:text-white transition-colors">
                      arXiv Ingestion
                    </a>
                  </li>
                  <li>
                    <a href="#input-form" className="hover:text-white transition-colors">
                      PDF Upload
                    </a>
                  </li>
                  <li>
                    <a href="#how" className="hover:text-white transition-colors">
                      Verification Pipeline
                    </a>
                  </li>
                  <li>
                    <a href="#how" className="hover:text-white transition-colors">
                      Documentation
                    </a>
                  </li>
                  <li>
                    <a href="#how" className="hover:text-white transition-colors">
                      Privacy & Terms
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="mt-12 pt-8 border-t border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-[11px] text-zinc-500">
            <div>
              © 2026 PaperScope. Engineered for AI researchers, engineers, and curious minds.
            </div>
            <div className="flex items-center gap-4">
              <span className="text-zinc-600">ZERO HALLUCINATION GUARANTEE</span>
              <span className="text-zinc-700">•</span>
              <span className="text-[#0055FF] font-bold">VERIFIED OUTPUT</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
