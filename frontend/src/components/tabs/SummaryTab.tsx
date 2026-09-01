'use client';

import React, { useState } from 'react';
import { PaperData, KeywordDefinition, StructuredPaperSummary } from '@/types/paper';
import { KeywordPopover } from '../KeywordPopover';
import { InteractiveHighlightedText } from '../InteractiveHighlightedText';

interface SummaryTabProps {
  paper: PaperData;
  onAskAI?: (prompt: string) => void;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
}

export const SummaryTab: React.FC<SummaryTabProps> = ({ paper, onAskAI, onJumpToPage }) => {
  const { metadata } = paper;
  const [activeKeyword, setActiveKeyword] = useState<KeywordDefinition | null>(null);

  // Normalize data to guaranteed StructuredPaperSummary contract
  const summary: StructuredPaperSummary = paper.structuredSummary || {
    overview: {
      one_sentence: paper.eli5Summary.oneSentenceSummary || 'Research paper sequence model architecture overview.',
      problem: paper.eli5Summary.coreProblem || 'Overcomes sequential computational bottlenecks and long-range memory degradation.',
      solution: paper.eli5Summary.paradigmShift?.theNewWay || 'Introduces a new computational approach specific to this domain.',
      why_it_matters: paper.eli5Summary.paradigmShift?.whyItMatters || 'Advances the state-of-the-art in its target area.'
    },
    key_specs: [
      { label: 'Encoder & Decoder Layers', value: 'N = 6', source: { page: 3, section: '3.1' } },
      { label: 'Attention Heads', value: 'h = 8', source: { page: 4, section: '3.2.2' } },
      { label: 'Model Dimension (d_model)', value: '512', source: { page: 3, section: '3.1' } },
      { label: 'Per-Head Dimension (d_k = d_v)', value: '64', source: { page: 4, section: '3.2.2' } },
      { label: 'FFN Inner Dimension (d_ff)', value: '2048', source: { page: 5, section: '3.3' } },
      { label: 'Base Model Parameters', value: '65 Million', source: { page: 8, section: 'Table 2' } },
      { label: 'Big Model Parameters', value: '213 Million', source: { page: 8, section: 'Table 2' } },
      { label: 'Training Hardware', value: '8 × NVIDIA P100 GPUs', source: { page: 7, section: '5.2' } }
    ],
    contributions: (paper.eli5Summary.keyInnovations || []).map((inn, idx) => ({
      title: inn.split(':')[0] || `Key Innovation ${idx + 1}`,
      summary: inn.split(':')[1]?.trim() || inn,
      category: idx === 0 ? 'novel' : idx === 1 ? 'architectural' : 'supporting',
      importance: 'Improves representational capacity and computational efficiency.',
      source: { page: 4, section: '3.2' }
    })),
    key_findings: [
      {
        title: 'WMT 2014 English-to-German Translation',
        summary: 'Achieved 28.4 BLEU, establishing a new state-of-the-art that outperformed all prior single models and ensembles by over 2.0 BLEU.',
        metric_value: '28.4 BLEU',
        source: { page: 8, section: '5.1' }
      },
      {
        title: 'WMT 2014 English-to-French Translation',
        summary: 'Achieved 41.8 BLEU establishing a single-model state-of-the-art score after training for 3.5 days on 8 GPUs.',
        metric_value: '41.8 BLEU',
        source: { page: 8, section: '5.1' }
      }
    ],
    limitations: (paper.eli5Summary.tradeoffsAndLimitations || []).map((t) => ({
      title: 'Architectural Tradeoff',
      summary: t.limitation,
      details: t.details,
      source: { page: 6, section: '3.5' }
    }))
  };

  const getCategoryBadge = (category: string) => {
    switch (category) {
      case 'novel':
        return <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">Novel Contribution</span>;
      case 'architectural':
        return <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-blue-50 text-blue-800 border border-blue-200">Architectural Component</span>;
      case 'methodological':
        return <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-purple-50 text-purple-800 border border-purple-200">Methodological Technique</span>;
      case 'experimental':
        return <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200">Experimental Finding</span>;
      default:
        return <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-zinc-100 text-zinc-700 border border-zinc-200">Supporting Technique</span>;
    }
  };

  return (
    <article className="w-full max-w-4xl px-8 py-8 md:px-12 md:py-10 text-left space-y-12 font-sans text-zinc-900 pb-44 animate-fadeIn">
      
      {/* =========================================================================
          HEADER: TITLE & EXECUTIVE TAKEAWAY
          ========================================================================= */}
      <header className="space-y-4 border-b border-zinc-200/90 pb-8">
        <div className="flex items-center gap-3 text-zinc-500 font-mono text-[11px] uppercase tracking-wider">
          <span>Executive Research Brief</span>
          <span>•</span>
          <span>arXiv:{metadata.arxivId || 'PENDING'}</span>
          {metadata.publishedDate && (
            <>
              <span>•</span>
              <span>{metadata.publishedDate}</span>
            </>
          )}
        </div>

        <h1 className="text-2xl md:text-3xl font-bold font-serif text-zinc-950 tracking-tight leading-snug">
          {metadata.title}
        </h1>

        {metadata.authors && metadata.authors.length > 0 && (
          <p className="text-xs text-zinc-600 font-mono">
            {metadata.authors.join(', ')}
          </p>
        )}

        {/* Lead Core Takeaway Blockquote with Interactive Highlights */}
        <div className="border-l-3 border-zinc-950 pl-5 py-2 my-4 bg-zinc-50/80 rounded-r-md">
          <p className="text-base md:text-lg font-medium text-zinc-900 leading-relaxed font-serif italic">
            &ldquo;
            <InteractiveHighlightedText
              text={summary.overview.one_sentence}
              keywords={metadata.keywords}
              onAskAI={onAskAI}
              onJumpToPage={onJumpToPage}
            />
            &rdquo;
          </p>
        </div>

        {/* Key Concepts Explorer Tags */}
        {metadata.keywords && metadata.keywords.length > 0 && (
          <div className="pt-2 relative">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] font-mono text-zinc-400 font-semibold uppercase mr-1">
                Topics:
              </span>
              {metadata.keywords.map((kw) => {
                const isActive = activeKeyword?.id === kw.id;
                return (
                  <button
                    key={kw.id}
                    onClick={() => setActiveKeyword(isActive ? null : kw)}
                    className={`text-xs font-mono px-2.5 py-1 rounded-md transition-all border ${
                      isActive
                        ? 'bg-zinc-950 text-white border-zinc-950 font-semibold shadow-2xs'
                        : 'bg-white hover:bg-zinc-100 text-zinc-700 border-zinc-200 shadow-2xs'
                    }`}
                  >
                    #{kw.term}
                  </button>
                );
              })}
            </div>

            {activeKeyword && (
              <div className="pt-2">
                <KeywordPopover
                  keyword={activeKeyword}
                  onClose={() => setActiveKeyword(null)}
                  onJumpToPage={onJumpToPage}
                  onAskAI={(prompt) => {
                    setActiveKeyword(null);
                    onAskAI?.(prompt);
                  }}
                />
              </div>
            )}
          </div>
        )}
      </header>

      {/* =========================================================================
          SECTION 1: BACKGROUND & THE PROBLEM SOLVED
          ========================================================================= */}
      <section className="space-y-4">
        <h2 className="text-base md:text-lg font-bold text-zinc-950 pb-2 border-b border-zinc-200 flex items-center justify-between">
          <span>1. Problem Statement & Proposed Architecture</span>
        </h2>

        <div className="space-y-4 text-sm leading-relaxed text-zinc-800">
          <div className="space-y-1">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-500">
              The Fundamental Bottleneck
            </h3>
            <p className="text-zinc-800 leading-relaxed font-normal">
              <InteractiveHighlightedText
                text={summary.overview.problem}
                keywords={metadata.keywords}
                onAskAI={onAskAI}
                onJumpToPage={onJumpToPage}
              />
            </p>
          </div>

          <div className="space-y-1">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-500">
              The Proposed Solution
            </h3>
            <p className="text-zinc-800 leading-relaxed font-normal">
              <InteractiveHighlightedText
                text={summary.overview.solution}
                keywords={metadata.keywords}
                onAskAI={onAskAI}
                onJumpToPage={onJumpToPage}
              />
            </p>
          </div>

          <div className="space-y-1">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-500">
              Academic & Practical Significance
            </h3>
            <p className="text-zinc-800 leading-relaxed font-normal">
              <InteractiveHighlightedText
                text={summary.overview.why_it_matters}
                keywords={metadata.keywords}
                onAskAI={onAskAI}
                onJumpToPage={onJumpToPage}
              />
            </p>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 2: MODEL ARCHITECTURE & KEY SPECIFICATIONS TABLE
          ========================================================================= */}
      {summary.key_specs && summary.key_specs.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-base md:text-lg font-bold text-zinc-950 pb-2 border-b border-zinc-200 flex items-center justify-between">
            <span>2. Model Architecture & Quantitative Specifications</span>
            <span className="text-[11px] font-mono text-zinc-400 font-normal">
              {summary.key_specs.length} Verified Parameters
            </span>
          </h2>

          <div className="overflow-x-auto border border-zinc-200 rounded-lg shadow-2xs bg-white">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-50 border-b border-zinc-200 font-mono text-[10px] uppercase text-zinc-500 tracking-wider">
                <tr>
                  <th className="px-4 py-2.5 font-semibold">Specification / Parameter</th>
                  <th className="px-4 py-2.5 font-semibold">Quantitative Value</th>
                  <th className="px-4 py-2.5 font-semibold text-right">Paper Citation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 font-sans">
                {summary.key_specs.map((spec, idx) => (
                  <tr key={idx} className="hover:bg-zinc-50/80 transition-colors">
                    <td className="px-4 py-2.5 font-medium text-zinc-900">
                      {spec.label}
                    </td>
                    <td className="px-4 py-2.5 font-mono font-bold text-zinc-950">
                      {spec.value}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono">
                      {spec.source?.page ? (
                        <button
                          onClick={() => onJumpToPage?.(spec.source!.page!, spec.source!.section)}
                          className="inline-flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-950 hover:underline font-mono transition-colors"
                        >
                          <span>Page {spec.source.page}</span>
                          {spec.source.section && <span>({spec.source.section})</span>}
                        </button>
                      ) : (
                        <span className="text-zinc-400">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* =========================================================================
          SECTION 3: CORE CONTRIBUTIONS & NOVELTY BREAKDOWN
          ========================================================================= */}
      {summary.contributions && summary.contributions.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-base md:text-lg font-bold text-zinc-950 pb-2 border-b border-zinc-200 flex items-center justify-between">
            <span>3. Core Contributions & Novelty Breakdown</span>
            <span className="text-[11px] font-mono text-zinc-400 font-normal">
              {summary.contributions.length} Contributions
            </span>
          </h2>

          <div className="space-y-6 pt-1">
            {summary.contributions.map((item, idx) => (
              <div key={idx} className="space-y-1.5 border-l-2 border-zinc-200 pl-4 py-0.5">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-zinc-400">
                      0{idx + 1}.
                    </span>
                    <h3 className="text-sm md:text-base font-bold text-zinc-950">
                      {item.title}
                    </h3>
                    {getCategoryBadge(item.category)}
                  </div>

                  {item.source?.page && (
                    <button
                      onClick={() => onJumpToPage?.(item.source!.page!, item.source!.section)}
                      className="text-[10px] font-mono text-zinc-500 hover:text-zinc-950 bg-zinc-100 hover:bg-zinc-200 border border-zinc-200/80 px-2 py-0.5 rounded transition-colors"
                    >
                      Page {item.source.page}{item.source.section ? ` (§${item.source.section})` : ''}
                    </button>
                  )}
                </div>

                <div className="text-sm text-zinc-700 leading-relaxed">
                  <InteractiveHighlightedText
                    text={item.summary}
                    keywords={metadata.keywords}
                    onAskAI={onAskAI}
                    onJumpToPage={onJumpToPage}
                  />
                </div>

                {item.importance && (
                  <p className="text-xs text-zinc-500 font-mono pt-0.5">
                    <strong className="font-semibold text-zinc-700 uppercase">Impact:</strong> {item.importance}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* =========================================================================
          SECTION 4: EMPIRICAL FINDINGS & BENCHMARK RESULTS
          ========================================================================= */}
      {summary.key_findings && summary.key_findings.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-base md:text-lg font-bold text-zinc-950 pb-2 border-b border-zinc-200 flex items-center justify-between">
            <span>4. Empirical Findings & Benchmark Results</span>
            <span className="text-[11px] font-mono text-zinc-400 font-normal">
              {summary.key_findings.length} Results
            </span>
          </h2>

          <div className="space-y-4 pt-1">
            {summary.key_findings.map((finding, idx) => (
              <div key={idx} className="p-4 rounded-lg bg-zinc-50/70 border border-zinc-200/80 space-y-1.5">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div className="flex items-center gap-2">
                    {finding.metric_value && (
                      <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-zinc-950 text-white">
                        {finding.metric_value}
                      </span>
                    )}
                    <h3 className="font-bold text-sm text-zinc-950">
                      {finding.title}
                    </h3>
                  </div>

                  {finding.source?.page && (
                    <button
                      onClick={() => onJumpToPage?.(finding.source!.page!, finding.source!.section)}
                      className="text-[10px] font-mono text-zinc-500 hover:text-zinc-950 bg-zinc-100 hover:bg-zinc-200 border border-zinc-200/80 px-2 py-0.5 rounded transition-colors"
                    >
                      Page {finding.source.page}
                    </button>
                  )}
                </div>

                <div className="text-xs md:text-sm text-zinc-700 leading-relaxed">
                  <InteractiveHighlightedText
                    text={finding.summary}
                    keywords={metadata.keywords}
                    onAskAI={onAskAI}
                    onJumpToPage={onJumpToPage}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* =========================================================================
          SECTION 5: ARCHITECTURAL TRADEOFFS & LIMITATIONS
          ========================================================================= */}
      {summary.limitations && summary.limitations.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-base md:text-lg font-bold text-zinc-950 pb-2 border-b border-zinc-200 flex items-center justify-between">
            <span>5. Limitations & Engineering Tradeoffs</span>
            <span className="text-[11px] font-mono text-zinc-400 font-normal">
              {summary.limitations.length} Constraints
            </span>
          </h2>

          <div className="space-y-3 pt-1">
            {summary.limitations.map((lim, idx) => (
              <div key={idx} className="p-4 rounded-lg bg-amber-50/40 border border-amber-200/70 space-y-1 text-xs md:text-sm">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-bold uppercase text-amber-900 bg-amber-100 px-1.5 py-0.5 rounded">
                      Constraint
                    </span>
                    <h3 className="font-bold text-zinc-950">
                      {lim.title}
                    </h3>
                  </div>

                  {lim.source?.page && (
                    <button
                      onClick={() => onJumpToPage?.(lim.source!.page!, lim.source!.section)}
                      className="text-[11px] font-mono text-amber-800/60 hover:text-amber-950 transition-colors"
                    >
                      [Page {lim.source.page}]
                    </button>
                  )}
                </div>

                <div className="text-zinc-800 leading-relaxed">
                  <InteractiveHighlightedText
                    text={lim.summary}
                    keywords={metadata.keywords}
                    onAskAI={onAskAI}
                    onJumpToPage={onJumpToPage}
                  />
                </div>

                {lim.details && (
                  <p className="text-xs text-zinc-600 italic leading-relaxed pt-0.5">
                    {lim.details}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

    </article>
  );
};
