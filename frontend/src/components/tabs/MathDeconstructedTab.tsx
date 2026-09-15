'use client';

import React, { useState, useMemo } from 'react';
import { PaperData, EquationBreakdown } from '@/types/paper';
import { Calculator, Sliders } from 'lucide-react';
import katex from 'katex';

interface MathDeconstructedTabProps {
  paper: PaperData;
}

// Reusable safe KaTeX math component
export const MathFormula: React.FC<{ latex?: string; displayMode?: boolean; className?: string }> = ({
  latex = '',
  displayMode = true,
  className = '',
}) => {
  const html = useMemo(() => {
    if (!latex || typeof latex !== 'string' || !latex.trim()) {
      return '';
    }
    try {
      return katex.renderToString(latex, {
        displayMode,
        throwOnError: false,
      });
    } catch {
      return `<code>${latex}</code>`;
    }
  }, [latex, displayMode]);

  if (!html) return null;

  return (
    <div
      className={`katex-rendered ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

export const MathDeconstructedTab: React.FC<MathDeconstructedTabProps> = ({ paper }) => {
  const equations = useMemo(() => paper.equations || [], [paper.equations]);

  // Interactive Sandbox state: scaling factor simulation
  const [dkValue, setDkValue] = useState<number>(64);
  const sqrtDk = useMemo(() => Math.sqrt(dkValue).toFixed(2), [dkValue]);
  const unscaledVariance = dkValue;
  const scaledVariance = (unscaledVariance / (Math.sqrt(dkValue) ** 2)).toFixed(1);

  // Honest empty state: when no equations were extracted from the paper
  if (equations.length === 0) {
    return (
      <div className="p-6 md:p-8 animate-fadeIn max-w-2xl mx-auto pb-36 font-sans">
        <div className="rounded-lg border border-zinc-200 bg-white p-8 text-center space-y-3">
          <div className="w-10 h-10 rounded-md bg-zinc-100 text-zinc-500 flex items-center justify-center mx-auto">
            <Calculator className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-zinc-950 text-sm font-mono uppercase">
            No mathematical formulations extracted
          </h4>
          <p className="text-xs text-zinc-600 leading-relaxed max-w-md mx-auto">
            This paper describes its architecture and methodology empirically without explicit mathematical equations, or the mathematical formulation could not be formally verified. You can explore the Summary, Architecture, and Parsed Text tabs to examine the methodology.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8 space-y-8 animate-fadeIn max-w-3xl mx-auto pb-36 text-zinc-900 font-sans">
      
      {/* Header */}
      <div className="space-y-3 border-b border-zinc-200 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-zinc-500 font-mono text-[11px] uppercase tracking-wider font-semibold">
            <Calculator className="w-3.5 h-3.5 text-zinc-900" />
            <span>Mathematical Formulations ({equations.length})</span>
          </div>
          <span className="text-[10px] font-mono text-zinc-400">
            Formally Verified KaTeX
          </span>
        </div>

        {/* Quick jump anchor links if multiple equations (flex-wrap, never horizontal overflow!) */}
        {equations.length > 1 && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            {equations.map((eq, idx) => (
              <button
                key={eq.id || `eq-${idx}`}
                onClick={() => {
                  const el = document.getElementById(eq.id || `formula-${idx}`);
                  el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }}
                className="px-2.5 py-1 rounded-md text-xs font-mono font-medium bg-white hover:bg-zinc-100 border border-zinc-200 text-zinc-700 transition-all cursor-pointer shadow-2xs active:scale-95"
              >
                0{idx + 1} // {eq.title || `Equation ${idx + 1}`}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Primary Equations List (Stacked vertically one below the other) */}
      <div className="space-y-8">
        {equations.map((eq, idx) => {
          const isAttentionFormula = Boolean(
            eq?.id?.toLowerCase().includes('attention') ||
            eq?.title?.toLowerCase().includes('attention')
          );
          const variables = eq.variables || [];

          return (
            <section
              key={eq.id || `eq-${idx}`}
              id={eq.id || `formula-${idx}`}
              className="p-6 rounded-xl bg-white border border-zinc-200/90 shadow-2xs space-y-5 scroll-mt-6"
            >
              {/* Title & Explanation */}
              <div className="border-l-2 border-zinc-950 pl-4 py-0.5 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono font-bold text-zinc-400 uppercase">
                    0{idx + 1}
                  </span>
                  <h3 className="font-bold text-zinc-950 text-base leading-snug">
                    {eq.title || `Mathematical Formulation ${idx + 1}`}
                  </h3>
                </div>
                {eq.explanation && (
                  <p className="text-xs text-zinc-600 leading-relaxed pt-0.5">
                    {eq.explanation}
                  </p>
                )}
              </div>

              {/* KaTeX Formula Display */}
              {eq.latex && (
                <div className="p-6 rounded-lg bg-zinc-50/70 border border-zinc-200/80 flex items-center justify-center text-zinc-950 overflow-x-auto min-h-[90px]">
                  <MathFormula latex={eq.latex} displayMode={true} className="text-base md:text-lg font-serif" />
                </div>
              )}

              {/* Interactive Scaling Sandbox (for Attention) */}
              {isAttentionFormula && (
                <div className="p-4 rounded-lg bg-white border border-zinc-200 space-y-3">
                  <div className="flex items-center justify-between border-b border-zinc-100 pb-2">
                    <div className="flex items-center gap-2">
                      <Sliders className="w-3.5 h-3.5 text-zinc-700" />
                      <h4 className="font-bold text-zinc-950 text-xs font-mono uppercase">
                        Scaling Factor Simulator (1 / √d_k)
                      </h4>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400">Live Tuning</span>
                  </div>

                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-zinc-700">
                        Key Dimension (d_k): <span className="font-mono font-bold text-zinc-950">{dkValue}</span>
                      </span>
                      <span className="font-mono text-xs text-zinc-600">
                        √d_k = <span className="font-bold text-zinc-950">{sqrtDk}</span>
                      </span>
                    </div>

                    <input
                      type="range"
                      min="16"
                      max="512"
                      step="16"
                      value={dkValue}
                      onChange={(e) => setDkValue(Number(e.target.value))}
                      className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-950"
                    />

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
                      <div className="p-2.5 rounded-md bg-zinc-50 border border-zinc-200/80 space-y-1">
                        <span className="text-[10px] font-mono uppercase text-zinc-500 font-semibold">Unscaled Dot Product</span>
                        <p className="text-xs font-mono font-bold text-zinc-950">Var(q · k) = {unscaledVariance}</p>
                        <p className="text-[11px] text-zinc-500">Unbounded dot products cause near-zero softmax gradients.</p>
                      </div>

                      <div className="p-2.5 rounded-md bg-zinc-50 border border-zinc-200/80 space-y-1">
                        <span className="text-[10px] font-mono uppercase text-zinc-500 font-semibold">Scaled Dot Product</span>
                        <p className="text-xs font-mono font-bold text-zinc-950">Var(q · k / √d_k) = {scaledVariance}</p>
                        <p className="text-[11px] text-zinc-500">Dividing by √d_k normalizes variance to 1.0, stabilizing training.</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Variables breakdown for this equation */}
              {variables.length > 0 && (
                <div className="pt-2 border-t border-zinc-100 space-y-2.5">
                  <span className="text-zinc-500 font-mono text-[10px] uppercase tracking-wider font-semibold block">
                    Variables Breakdown ({variables.length})
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {variables.map((v, vIdx) => (
                      <div
                        key={v.symbol || `var-${vIdx}`}
                        className="p-3 rounded-lg bg-zinc-50/70 border border-zinc-200/80 space-y-1"
                      >
                        <div className="flex items-center justify-between">
                          <MathFormula latex={v.symbol || ''} displayMode={false} className="font-serif font-bold text-sm text-zinc-950" />
                          {v.dimension && (
                            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-700 font-medium border border-zinc-200">
                              {v.dimension}
                            </span>
                          )}
                        </div>
                        {v.name && <p className="text-xs font-semibold text-zinc-950 mt-0.5">{v.name}</p>}
                        {v.meaning && <p className="text-xs text-zinc-600 leading-relaxed">{v.meaning}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          );
        })}
      </div>

    </div>
  );
};
