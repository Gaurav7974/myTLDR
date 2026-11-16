import { describe, it, expect } from 'vitest';
import { transformBackendPayloadToPaperData } from './api';
import { selectDiagramGraph } from './diagramSelection';
import type { PaperData } from '../types/paper';

const IR = {
  version: '1.0',
  architecture: { name: 'TestNet', paradigm: 'other' },
  nodes: [{ id: 'a', label: 'Attention', type: 'attention' }],
  edges: [],
};

const RICH_GRAPH = {
  version: '2.0',
  source: 'vision',
  layout_mode: 'figure',
  nodes: [{ id: 'a', label: 'Attention', type: 'attention' }],
  edges: [],
  verification: { valid: true, confidence: 0.9 },
};

function paperWith(overrides: Record<string, unknown>): PaperData {
  return transformBackendPayloadToPaperData(overrides, undefined, 'paper.pdf');
}

describe('diagram availability — strict text-only', () => {
  it('drops the IR-derived graph when the backend reports no diagram', () => {
    const paper = paperWith({ diagram_available: false, architecture_ir: IR });

    expect(paper.diagramAvailable).toBe(false);
    expect(paper.architectureGraph).toBeUndefined();
    expect(selectDiagramGraph(paper)).toBeUndefined();
  });

  it('keeps a verified graph when the backend reports a diagram', () => {
    const paper = paperWith({ diagram_available: true, graph: RICH_GRAPH });

    expect(paper.diagramAvailable).toBe(true);
    expect(paper.architectureGraph).toEqual(RICH_GRAPH);
    expect(selectDiagramGraph(paper)).toEqual(RICH_GRAPH);
  });

  it('does not synthesize a graph from IR on a legacy paper without the flag', () => {
    const paper = paperWith({ architecture_ir: IR });

    expect(paper.diagramAvailable).toBeUndefined();
    expect(selectDiagramGraph(paper)).toBeDefined();
  });
});
