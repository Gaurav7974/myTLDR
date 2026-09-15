'use client';
import React, { useState } from 'react';
import { PaperData, KeywordDefinition, StructuredPaperSummary } from '@/types/paper';
import { KeywordPopover } from '../KeywordPopover';
import { InteractiveHighlightedText } from '../InteractiveHighlightedText';

interface SummaryTabProps { paper: PaperData; onAskAI?: (p:string)=>void; onJumpToPage?: (page:number, sec?:string)=>void; }

// Narrative section shell — editorial rule + numbered marker. No decorative
// icons: the numbering carries the structure.
function Section({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <div className="flex items-baseline gap-3 border-b-2 border-zinc-950 pb-2">
        <span className="font-mono text-xs font-bold text-zinc-400">{n}</span>
        <h2 className="text-base font-bold tracking-tight text-zinc-950">{title}</h2>
      </div>
      {children}
    </section>
  );
}

export const SummaryTab: React.FC<SummaryTabProps> = ({ paper, onAskAI, onJumpToPage }) => {
  const { metadata } = paper;
  const [kw, setKw] = useState<KeywordDefinition|null>(null);
  const s: StructuredPaperSummary | undefined = paper.structuredSummary;
  const badge = (c:string)=>({novel:'bg-emerald-50 text-emerald-700 border-emerald-200',architectural:'bg-blue-50 text-blue-700 border-blue-200',methodological:'bg-purple-50 text-purple-700 border-purple-200',experimental:'bg-amber-50 text-amber-700 border-amber-200'}[c]||'bg-zinc-100 text-zinc-600 border-zinc-200');

  // Honest unavailable state: when the backend produced no summary we say so.
  // Never substitute another paper's facts (the old fallback showed
  // Transformer specs for every paper).
  if (!s) {
    return (
      <article className="w-full max-w-3xl mx-auto px-6 md:px-8 py-8 pb-32 space-y-6 text-zinc-900 leading-relaxed">
        <header className="space-y-3">
          <p className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">{metadata.arxivId ? `arXiv:${metadata.arxivId}` : 'RESEARCH PAPER'}{metadata.publishedDate ? ` • ${metadata.publishedDate}` : ''}</p>
          <h1 className="font-serif text-[26px] md:text-[30px] font-bold tracking-tight leading-tight">{metadata.title}</h1>
          {metadata.authors?.length>0 && <p className="text-xs font-mono text-zinc-500">{metadata.authors.join(', ')}</p>}
        </header>
        <div className="rounded-lg border border-zinc-200 bg-white p-8 text-center space-y-2">
          <h3 className="text-sm font-semibold text-zinc-800">Summary unavailable for this paper</h3>
          <p className="text-xs text-zinc-500 leading-relaxed">
            The explanation step produced nothing usable. Parsed text and other tabs still work;
            we show nothing rather than a plausible-looking invented summary.
          </p>
        </div>
      </article>
    );
  }

  return (
    <article className="w-full max-w-3xl mx-auto px-6 md:px-8 py-8 pb-32 space-y-10 text-zinc-900 leading-relaxed">
      {/* header — paper identity, no summary text here */}
      <header className="space-y-3">
        <p className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">{metadata.arxivId ? `arXiv:${metadata.arxivId}` : 'RESEARCH PAPER'}{metadata.publishedDate ? ` • ${metadata.publishedDate}` : ''}</p>
        <h1 className="font-serif text-[26px] md:text-[30px] font-bold tracking-tight leading-tight">{metadata.title}</h1>
        {metadata.authors?.length>0 && <p className="text-xs font-mono text-zinc-500">{metadata.authors.join(', ')}</p>}
        {(metadata.keywords?.length||0)>0 && (
          <div className="flex flex-wrap items-center gap-1 pt-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 mr-1">Topics</span>
            {metadata.keywords!.map(k=>(
              <button key={k.id} onClick={()=>setKw(kw?.id===k.id?null:k)} className={`text-[11px] font-mono px-2 py-px rounded-full border transition ${kw?.id===k.id?'bg-zinc-900 text-white border-zinc-900':'bg-white border-zinc-200 hover:bg-zinc-50'}`}>#{k.term}</button>
            ))}
          </div>
        )}
        {kw && <KeywordPopover keyword={kw} onClose={()=>setKw(null)} onJumpToPage={onJumpToPage} onAskAI={p=>{setKw(null); onAskAI?.(p);}} />}
      </header>

      {/* 1 — the paper in one sentence */}
      {s.overview?.one_sentence && (
        <Section n="01" title="The Paper in One Sentence">
          <blockquote className="border-l-2 border-zinc-900 pl-4 py-1 text-[15px] font-serif italic text-zinc-800 leading-relaxed"><InteractiveHighlightedText text={s.overview.one_sentence} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></blockquote>
        </Section>
      )}

      {/* 2 — problem & why it matters */}
      {(s.overview?.problem || s.overview?.why_it_matters) && (
        <Section n="02" title="Problem & Motivation">
          {s.overview.problem && (
            <div><h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-zinc-500 mb-1">The problem</h3><p className="text-[14.5px] leading-7 text-zinc-800"><InteractiveHighlightedText text={s.overview.problem} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p></div>
          )}
          {s.overview.why_it_matters && (
            <div><h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-zinc-500 mb-1">Why it matters</h3><p className="text-[14.5px] leading-7 text-zinc-800"><InteractiveHighlightedText text={s.overview.why_it_matters} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p></div>
          )}
        </Section>
      )}

      {/* 3 — core idea */}
      {s.overview?.solution && (
        <Section n="03" title="Core Idea">
          <p className="text-[14.5px] leading-7 text-zinc-800"><InteractiveHighlightedText text={s.overview.solution} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p>
        </Section>
      )}

      {/* 4 — proposed approach: editorial index markers, no side rules */}
      {s.contributions?.length>0 && (
        <Section n="04" title="Proposed Approach">
          <ol className="space-y-6">
            {s.contributions.map((c,i)=>(
              <li key={i}>
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-1">
                  <span className="font-mono text-sm font-bold text-zinc-300">#{String(i+1).padStart(2,'0')}</span>
                  <h3 className="font-bold text-[15px] text-zinc-950 tracking-tight">{c.title}</h3>
                </div>
                <div className="flex flex-wrap items-center gap-2 mb-1.5">
                  <span className={`text-[10px] font-mono px-1.5 py-px rounded border ${badge(c.category)}`}>{c.category}</span>
                  {c.source?.page && <button onClick={()=>onJumpToPage?.(c.source!.page!,c.source!.section)} className="text-[11px] font-mono text-zinc-400 hover:text-zinc-900">p.{c.source.page} →</button>}
                </div>
                <p className="text-sm text-zinc-700 leading-6"><InteractiveHighlightedText text={c.summary} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p>
                {c.importance && <p className="text-xs text-zinc-500 mt-1">Why it matters: {c.importance}</p>}
              </li>
            ))}
          </ol>
        </Section>
      )}

      {/* 5 — results: observed vs interpreted */}
      {s.key_findings?.length>0 && (
        <Section n="05" title="Results">
          {s.key_findings.map((f,i)=>(
            <div key={i} className="flex gap-4">
              <span className="shrink-0 font-mono text-xs font-bold bg-zinc-900 text-white px-2 py-1 rounded h-fit">{f.metric_value}</span>
              <div><h3 className="font-semibold text-sm">{f.title}</h3><p className="text-sm text-zinc-700 leading-6"><InteractiveHighlightedText text={f.summary} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p></div>
            </div>
          ))}
          <p className="text-xs text-zinc-500 italic leading-relaxed border-l-2 border-zinc-200 pl-3">
            Read carefully: a metric measures performance under its evaluation setup.
            Strong numbers are evidence for that setup — not proof of universal superiority.
          </p>
        </Section>
      )}

      {/* 6 — limitations */}
      {s.limitations?.length>0 && (
        <Section n="06" title="Limitations">
          {s.limitations.map((l,i)=>(
            <div key={i} className="text-sm"><h3 className="font-semibold text-zinc-900">{l.title}</h3><p className="text-zinc-700 leading-6"><InteractiveHighlightedText text={l.summary} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p>{l.details && <p className="text-xs text-zinc-500 italic mt-1">{l.details}</p>}</div>
          ))}
        </Section>
      )}

      {/* 7 — specifications */}
      {s.key_specs?.length>0 && (
        <Section n="07" title="Specifications">
          <dl className="divide-y divide-zinc-100 text-sm">
            {s.key_specs.map((spec,i)=>(
              <div key={i} className="flex justify-between gap-4 py-2.5">
                <dt className="text-zinc-600">{spec.label}</dt>
                <dd className="flex items-center gap-3 font-mono font-semibold text-zinc-900">{spec.value}
                  {spec.source?.page && <button onClick={()=>onJumpToPage?.(spec.source!.page!, spec.source!.section)} className="text-[11px] font-mono text-zinc-400 hover:text-zinc-900">p.{spec.source.page} →</button>}
                </dd>
              </div>
            ))}
          </dl>
        </Section>
      )}
    </article>
  );
};
