'use client';

import React, { useState, useMemo } from 'react';
import { PaperData, EquationBreakdown } from '@/types/paper';
import { Calculator, Sliders } from 'lucide-react';
import katex from 'katex';

interface MathDeconstructedTabProps {
  paper: PaperData;
}

// Reusable safe KaTeX math component
export const MathFormula: React.FC<{ latex: string; displayMode?: boolean; className?: string }> = ({
  latex,
  displayMode = true,
  className = '',
}) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        displayMode,
        throwOnError: false,
      });
    } catch {
      return `<code>${latex}</code>`;
    }
  }, [latex, displayMode]);

  return (
    <div
      className={`katex-rendered ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

export const MathDeconstructedTab: React.FC<MathDeconstructedTabProps> = ({ paper }) => {
  const { equations } = paper;
  const [selectedEq, setSelectedEq] = useState<EquationBreakdown>(equations[0]);
  const [activeVar, setActiveVar] = useState<string | null>(equations[0]?.variables[0]?.symbol || null);

  // Interactive Sandbox state: scaling factor simulation
  const [dkValue, setDkValue] = useState<number>(64);
  const sqrtDk = useMemo(() => Math.sqrt(dkValue).toFixed(2), [dkValue]);
  const unscaledVariance = dkValue;
  const scaledVariance = (unscaledVariance / (Math.sqrt(dkValue) ** 2)).toFixed(1);

  return (
    <div className="p-6 md:p-8 space-y-7 animate-fadeIn max-w-3xl mx-auto pb-36 text-zinc-900 font-sans">
      
      {/* Header & Equation Selector Pills */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-zinc-500 font-mono text-[11px] uppercase tracking-wider font-semibold">
            <Calculator className="w-3.5 h-3.5 text-zinc-900" />
            <span>Mathematical Formulations</span>
          </div>
          <span className="text-[10px] font-mono text-zinc-400">
            Formally Verified KaTeX
          </span>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
          {equations.map((eq) => {
            const isSelected = selectedEq.id === eq.id;
            return (
              <button
                key={eq.id}
                onClick={() => {
                  setSelectedEq(eq);
                  setActiveVar(eq.variables[0]?.symbol || null);
                }}
                className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium whitespace-nowrap transition-all ${
                  isSelected
                    ? 'bg-zinc-950 text-white shadow-2xs font-semibold'
                    : 'bg-white hover:bg-zinc-100 border border-zinc-200 text-zinc-700'
                }`}
              >
                {eq.title}
              </button>
            );
          })}
        </div>
      </div>

      {/* Primary Mathematical Equation Section */}
      <section className="space-y-3">
        <div className="border-l-2 border-zinc-950 pl-4 py-0.5">
          <h3 className="font-bold text-zinc-950 text-base">{selectedEq.title}</h3>
          <p className="text-xs text-zinc-600 leading-relaxed mt-1">
            {selectedEq.explanation}
          </p>
        </div>

        {/* Formula Display */}
        <div className="p-6 rounded-lg bg-white border border-zinc-200/90 shadow-2xs flex items-center justify-center text-zinc-950 overflow-x-auto min-h-[90px]">
          <MathFormula latex={selectedEq.latex} displayMode={true} className="text-base md:text-lg font-serif" />
        </div>
      </section>

      {/* Interactive Scaling Sandbox (for Softmax / Attention) */}
      {selectedEq.id.includes('attention') && (
        <section className="p-5 rounded-lg bg-white border border-zinc-200/90 shadow-2xs space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-100 pb-2.5">
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-zinc-700" />
              <h4 className="font-bold text-zinc-950 text-xs font-mono uppercase">
                Interactive Scaling Factor Simulator (1 / √d_k)
              </h4>
            </div>
            <span className="text-[10px] font-mono text-zinc-400">
              Live Tuning
            </span>
          </div>

          <div className="space-y-3">
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

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200/80 space-y-1">
                <span className="text-[10px] font-mono uppercase text-zinc-500 font-semibold">Unscaled Dot Product</span>
                <p className="text-xs font-mono font-bold text-zinc-950">Var(q · k) = {unscaledVariance}</p>
                <p className="text-[11px] text-zinc-500">Unbounded dot products cause near-zero softmax gradients.</p>
              </div>

              <div className="p-3 rounded-md bg-zinc-50 border border-zinc-200/80 space-y-1">
                <span className="text-[10px] font-mono uppercase text-zinc-500 font-semibold">Scaled Dot Product</span>
                <p className="text-xs font-mono font-bold text-zinc-950">Var(q · k / √d_k) = {scaledVariance}</p>
                <p className="text-[11px] text-zinc-500">Dividing by √d_k normalizes variance to 1.0, stabilizing training.</p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Variable Inspector Grid */}
      <section className="space-y-3">
        <div className="flex items-center justify-between border-b border-zinc-200 pb-2">
          <span className="text-zinc-500 font-mono text-[11px] uppercase tracking-wider font-semibold">
            Variable Inspector ({selectedEq.variables.length})
          </span>
          <span className="text-[10px] font-mono text-zinc-400">
            Click variable to inspect
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {selectedEq.variables.map((v) => {
            const isActive = activeVar === v.symbol;
            return (
              <div
                key={v.symbol}
                onClick={() => setActiveVar(v.symbol)}
                className={`p-3.5 rounded-md border transition-all cursor-pointer ${
                  isActive
                    ? 'bg-white border-zinc-950 ring-1 ring-zinc-950 shadow-2xs'
                    : 'bg-white hover:border-zinc-300 border-zinc-200/90'
                }`}
              >
                <div className="flex items-center justify-between">
                  <MathFormula latex={v.symbol} displayMode={false} className="font-serif font-bold text-base text-zinc-950" />
                  {v.dimension && (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-100 text-zinc-700 font-medium border border-zinc-200">
                      {v.dimension}
                    </span>
                  )}
                </div>
                <p className="text-xs font-semibold text-zinc-950 mt-1.5">{v.name}</p>
                <p className="text-xs text-zinc-600 mt-1 leading-relaxed">{v.meaning}</p>
              </div>
            );
          })}
        </div>
      </section>

    </div>
  );
};
