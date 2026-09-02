'use client';
import React, { useState } from 'react';
import { PaperData, KeywordDefinition, StructuredPaperSummary } from '@/types/paper';
import { KeywordPopover } from '../KeywordPopover';
import { InteractiveHighlightedText } from '../InteractiveHighlightedText';

interface SummaryTabProps { paper: PaperData; onAskAI?: (p:string)=>void; onJumpToPage?: (page:number, sec?:string)=>void; }

export const SummaryTab: React.FC<SummaryTabProps> = ({ paper, onAskAI, onJumpToPage }) => {
  const { metadata } = paper;
  const [kw, setKw] = useState<KeywordDefinition|null>(null);
  const s: StructuredPaperSummary = paper.structuredSummary || {
    overview: { one_sentence: paper.eli5Summary.oneSentenceSummary||'Architecture overview.', problem: paper.eli5Summary.coreProblem||'', solution: paper.eli5Summary.paradigmShift?.theNewWay||'', why_it_matters: paper.eli5Summary.paradigmShift?.whyItMatters||'' },
    key_specs: [
      {label:'Encoder & Decoder Layers',value:'N = 6',source:{page:3,section:'3.1'}},{label:'Attention Heads',value:'h = 8',source:{page:4,section:'3.2.2'}},{label:'Model Dimension',value:'512',source:{page:3,section:'3.1'}},{label:'Per-Head Dimension',value:'64',source:{page:4,section:'3.2.2'}},{label:'FFN Dimension',value:'2048',source:{page:5,section:'3.3'}},{label:'Base Params',value:'65M',source:{page:8,section:'Table 2'}},{label:'Big Params',value:'213M',source:{page:8,section:'Table 2'}},{label:'Training',value:'8× P100 GPUs',source:{page:7,section:'5.2'}},
    ],
    contributions: (paper.eli5Summary.keyInnovations||[]).map((inn,i)=>({title:inn.split(':')[0]||`Innovation ${i+1}`,summary:inn.split(':')[1]?.trim()||inn,category:i===0?'novel':i===1?'architectural':'supporting',importance:'Improves capacity & efficiency.',source:{page:4,section:'3.2'}})),
    key_findings: [{title:'WMT En→De',summary:'28.4 BLEU — SOTA, +2.0 over prior ensembles.',metric_value:'28.4 BLEU',source:{page:8,section:'5.1'}},{title:'WMT En→Fr',summary:'41.8 BLEU single-model SOTA in 3.5 days on 8 GPUs.',metric_value:'41.8 BLEU',source:{page:8,section:'5.1'}}],
    limitations: (paper.eli5Summary.tradeoffsAndLimitations||[]).map(t=>({title:'Tradeoff',summary:t.limitation,details:t.details,source:{page:6,section:'3.5'}})),
  };
  const badge = (c:string)=>({novel:'bg-emerald-50 text-emerald-700 border-emerald-200',architectural:'bg-blue-50 text-blue-700 border-blue-200',methodological:'bg-purple-50 text-purple-700 border-purple-200',experimental:'bg-amber-50 text-amber-700 border-amber-200'}[c]||'bg-zinc-100 text-zinc-600 border-zinc-200');

  return (
    <article className="w-full max-w-3xl mx-auto px-6 md:px-8 py-8 pb-32 space-y-10 text-zinc-900 leading-relaxed">
      {/* header — editorial, no box */}
      <header className="space-y-3">
        <p className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">Executive Brief • arXiv:{metadata.arxivId||'PENDING'}{metadata.publishedDate?` • ${metadata.publishedDate}`:''}</p>
        <h1 className="font-serif text-[26px] md:text-[30px] font-bold tracking-tight leading-tight">{metadata.title}</h1>
        {metadata.authors?.length>0 && <p className="text-xs font-mono text-zinc-500">{metadata.authors.join(', ')}</p>}
        <blockquote className="border-l-2 border-zinc-900 pl-4 py-1 text-[15px] font-serif italic text-zinc-800 leading-relaxed"><InteractiveHighlightedText text={s.overview.one_sentence} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></blockquote>
        {(metadata.keywords?.length||0)>0 && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-zinc-400 mr-1">Topics</span>
            {metadata.keywords!.map(k=>(
              <button key={k.id} onClick={()=>setKw(kw?.id===k.id?null:k)} className={`text-xs font-mono px-2 py-0.5 rounded-full border transition ${kw?.id===k.id?'bg-zinc-900 text-white border-zinc-900':'bg-white border-zinc-200 hover:bg-zinc-50'}`}>#{k.term}</button>
            ))}
          </div>
        )}
        {kw && <KeywordPopover keyword={kw} onClose={()=>setKw(null)} onJumpToPage={onJumpToPage} onAskAI={p=>{setKw(null); onAskAI?.(p);}} />}
      </header>

      {/* 1. Narrative — no boxes, just prose */}
      <section className="space-y-6">
        <h2 className="font-mono text-xs font-bold uppercase tracking-widest text-zinc-900 border-b border-zinc-200 pb-2">1 — Problem & Solution</h2>
        {[
          ['The bottleneck', s.overview.problem],
          ['Proposed solution', s.overview.solution],
          ['Why it matters', s.overview.why_it_matters],
        ].map(([h,t])=>(
          <div key={h}><h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-zinc-500 mb-1">{h}</h3><p className="text-[14.5px] leading-7 text-zinc-800"><InteractiveHighlightedText text={t} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p></div>
        ))}
      </section>

      {/* 2. Specs — minimal table, no heavy border */}
      {s.key_specs?.length>0 && (
        <section className="space-y-3">
          <h2 className="font-mono text-xs font-bold uppercase tracking-widest text-zinc-900 border-b border-zinc-200 pb-2">2 — Specifications</h2>
          <dl className="divide-y divide-zinc-100 text-sm">
            {s.key_specs.map((spec,i)=>(
              <div key={i} className="flex justify-between gap-4 py-2.5">
                <dt className="text-zinc-600">{spec.label}</dt>
                <dd className="flex items-center gap-3 font-mono font-semibold text-zinc-900">{spec.value}
                  {spec.source?.page && <button onClick={()=>onJumpToPage?.(spec.source!.page!, spec.source!.section)} className="text-[11px] font-mono text-zinc-400 hover:text-zinc-900">p.{spec.source.page}</button>}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {/* 3. Contributions — flowing list, single left accent */}
      {s.contributions?.length>0 && (
        <section className="space-y-4">
          <h2 className="font-mono text-xs font-bold uppercase tracking-widest text-zinc-900 border-b border-zinc-200 pb-2">3 — Contributions</h2>
          <ol className="space-y-5">
            {s.contributions.map((c,i)=>(
              <li key={i} className="pl-4 border-l-2 border-zinc-200">
                <div className="flex flex-wrap items-center gap-2 mb-1"><span className="font-mono text-xs text-zinc-400">0{i+1}</span><span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${badge(c.category)}`}>{c.category}</span>{c.source?.page && <button onClick={()=>onJumpToPage?.(c.source!.page!,c.source!.section)} className="ml-auto text-[11px] font-mono text-zinc-400 hover:text-zinc-900">p.{c.source.page}</button>}</div>
                <h3 className="font-semibold text-sm text-zinc-900">{c.title}</h3>
                <p className="text-sm text-zinc-700 leading-6 mt-1"><InteractiveHighlightedText text={c.summary} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p>
                {c.importance && <p className="text-xs text-zinc-500 mt-1">{c.importance}</p>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {/* 4. Results — inline metric, no card */}
      {s.key_findings?.length>0 && (
        <section className="space-y-4">
          <h2 className="font-mono text-xs font-bold uppercase tracking-widest text-zinc-900 border-b border-zinc-200 pb-2">4 — Results</h2>
          {s.key_findings.map((f,i)=>(
            <div key={i} className="flex gap-4">
              <span className="shrink-0 font-mono text-xs font-bold bg-zinc-900 text-white px-2 py-1 rounded h-fit">{f.metric_value}</span>
              <div><h3 className="font-semibold text-sm">{f.title}</h3><p className="text-sm text-zinc-700 leading-6"><InteractiveHighlightedText text={f.summary} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p></div>
            </div>
          ))}
        </section>
      )}

      {/* 5. Limitations — subtle */}
      {s.limitations?.length>0 && (
        <section className="space-y-3">
          <h2 className="font-mono text-xs font-bold uppercase tracking-widest text-zinc-900 border-b border-zinc-200 pb-2">5 — Limitations</h2>
          {s.limitations.map((l,i)=>(
            <div key={i} className="text-sm"><h3 className="font-semibold text-zinc-900">{l.title}</h3><p className="text-zinc-700 leading-6"><InteractiveHighlightedText text={l.summary} keywords={metadata.keywords} onAskAI={onAskAI} onJumpToPage={onJumpToPage} /></p>{l.details && <p className="text-xs text-zinc-500 italic mt-1">{l.details}</p>}</div>
          ))}
        </section>
      )}
    </article>
  );
};
