/**
 * Unit tests for the architecture_graph.ts gate functions.
 * No network, no DOM — pure TypeScript logic.
 */

import { describe, it, expect } from 'vitest';
import {
  isVerifiedVisionGraph,
  shouldRenderReconstruction,
  shouldUseAutoLayout,
  VISION_CONFIDENCE_THRESHOLD,
  GRAPH_PAYLOAD_VERSION,
} from './architecture_graph';

function makeGraph(overrides: Record<string, unknown> = {}): Parameters<typeof isVerifiedVisionGraph>[0] {
  return {
    nodes: [{ id: 'n1', label: 'Input', type: 'input' }],
    edges: [{ source: 'n1', target: 'n2' }],
    source: 'vision',
    layout_mode: 'figure',
    verification: { valid: true, confidence: 0.9 },
    ...overrides,
  };
}

describe('GRAPH_PAYLOAD_VERSION', () => {
  it('exports 2.0', () => {
    expect(GRAPH_PAYLOAD_VERSION).toBe('2.0');
  });
});

describe('VISION_CONFIDENCE_THRESHOLD', () => {
  it('exports 0.65', () => {
    expect(VISION_CONFIDENCE_THRESHOLD).toBe(0.65);
  });
});

describe('isVerifiedVisionGraph', () => {
  it('returns false for null', () => {
    expect(isVerifiedVisionGraph(null)).toBe(false);
  });

  it('returns false for undefined', () => {
    expect(isVerifiedVisionGraph(undefined)).toBe(false);
  });

  it('returns false when source is text', () => {
    expect(isVerifiedVisionGraph(makeGraph({ source: 'text' }))).toBe(false);
  });

  it('returns false when source is auto', () => {
    expect(isVerifiedVisionGraph(makeGraph({ source: 'auto' }))).toBe(false);
  });

  it('returns false when source is merged but layout_mode is auto', () => {
    expect(isVerifiedVisionGraph(makeGraph({ source: 'merged', layout_mode: 'auto' }))).toBe(false);
  });

  it('returns false when layout_mode is legacy', () => {
    expect(isVerifiedVisionGraph(makeGraph({ layout_mode: 'legacy' }))).toBe(false);
  });

  it('returns false when verification is absent', () => {
    expect(isVerifiedVisionGraph(makeGraph({ verification: undefined }))).toBe(false);
  });

  it('returns false when verification.valid is false', () => {
    expect(isVerifiedVisionGraph(makeGraph({ verification: { valid: false, confidence: 0.9 } }))).toBe(false);
  });

  it('returns false when confidence is below threshold', () => {
    const below = VISION_CONFIDENCE_THRESHOLD - 0.01;
    expect(isVerifiedVisionGraph(makeGraph({ verification: { valid: true, confidence: below } }))).toBe(false);
  });

  it('returns false when confidence equals threshold - 0.01', () => {
    // boundary: 0.64 < 0.65
    expect(isVerifiedVisionGraph(makeGraph({ verification: { valid: true, confidence: 0.64 } }))).toBe(false);
  });

  it('returns true when confidence equals threshold', () => {
    expect(isVerifiedVisionGraph(makeGraph({ verification: { valid: true, confidence: VISION_CONFIDENCE_THRESHOLD } }))).toBe(true);
  });

  it('returns false when hallucinations are present', () => {
    expect(isVerifiedVisionGraph(makeGraph({ verification: { valid: true, confidence: 0.9, hallucinations: ['fake node'] } }))).toBe(false);
  });

  it('returns false when hallucinations is empty array', () => {
    // empty array is falsy length — should still be true
    expect(isVerifiedVisionGraph(makeGraph({ verification: { valid: true, confidence: 0.9, hallucinations: [] } }))).toBe(true);
  });

  it('returns true for a fully valid verified vision graph', () => {
    expect(isVerifiedVisionGraph(makeGraph({ source: 'vision', verification: { valid: true, confidence: 0.9 } }))).toBe(true);
  });

  it('returns true for a fully valid verified merged graph', () => {
    expect(isVerifiedVisionGraph(makeGraph({ source: 'merged', verification: { valid: true, confidence: 0.9 } }))).toBe(true);
  });
});

describe('shouldRenderReconstruction', () => {
  it('returns false for null', () => {
    expect(shouldRenderReconstruction(null)).toBe(false);
  });

  it('delegates to isVerifiedVisionGraph', () => {
    const good = makeGraph({ source: 'vision', layout_mode: 'figure', verification: { valid: true, confidence: 0.9 } });
    const bad = makeGraph({ source: 'text' });
    expect(shouldRenderReconstruction(good)).toBe(true);
    expect(shouldRenderReconstruction(bad)).toBe(false);
  });
});

describe('shouldUseAutoLayout', () => {
  it('returns false for null', () => {
    expect(shouldUseAutoLayout(null)).toBe(false);
  });

  it('returns true for layout_mode=auto', () => {
    expect(shouldUseAutoLayout(makeGraph({ layout_mode: 'auto' }))).toBe(true);
  });

  it('returns true for layout_mode=legacy', () => {
    expect(shouldUseAutoLayout(makeGraph({ layout_mode: 'legacy' }))).toBe(true);
  });

  it('returns true when layout_mode is absent', () => {
    expect(shouldUseAutoLayout(makeGraph({ layout_mode: undefined }))).toBe(true);
  });

  it('returns false for layout_mode=figure', () => {
    expect(shouldUseAutoLayout(makeGraph({ layout_mode: 'figure' }))).toBe(false);
  });
});
