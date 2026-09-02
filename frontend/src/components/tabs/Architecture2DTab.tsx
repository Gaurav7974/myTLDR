'use client';

import React from 'react';
import { PaperData } from '@/types/paper';
import ArchitectureDiagram from '@/components/ArchitectureDiagram';
import { isVerifiedVisionGraph } from '@/types/architecture_graph';

interface Architecture2DTabProps {
  paper: PaperData;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
}

export const Architecture2DTab: React.FC<Architecture2DTabProps> = ({ paper }) => {
  const graph = paper.architectureGraph;
  const hasGraph = !!(graph && (graph.nodes?.length ?? 0) > 0);

  const isUnverified = hasGraph && !isVerifiedVisionGraph(graph);
  return (
    <div className="flex flex-col h-full bg-white animate-fadeIn">
      {hasGraph ? (
        <div className="relative flex-1">
          {isUnverified && (
            <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-mono font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              Unverified — may be incomplete
            </div>
          )}
          <ArchitectureDiagram graph={graph} height={720} />
        </div>
      ) : (
        // Trust-first fallback: no static Transformer mock. The
        // diagram tab shows an honest unavailable state and points
        // the user to the parsed text tab instead.
        <div className="w-full h-full flex items-center justify-center p-12">
          <div className="max-w-md text-center">
            <div className="w-12 h-12 rounded-full bg-zinc-50 border border-zinc-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-5 h-5 text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17V7l8 5-8 5z" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-zinc-800 mb-1">
              2D architecture diagram not available for this paper
            </h3>
            <p className="text-xs text-zinc-500 leading-relaxed">
              The pipeline did not return a verified figure-faithful graph
              (or the text-only graph had no architecture to extract).
              Other tabs (parsed text) still work; we deliberately
              do not render a placeholder diagram that could misrepresent
              the paper.
            </p>
            {isVerifiedVisionGraph(graph) === false &&
             graph &&
             graph.verification &&
             !graph.verification.valid && (
              <p className="text-[10px] text-amber-600 mt-3 italic">
                The figure-faithfulness verifier rejected the candidate graph
                (hallucinations or low confidence). See the parsed-text tab
                for the original figure.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};