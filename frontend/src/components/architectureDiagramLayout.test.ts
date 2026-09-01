/**
 * Unit tests for the pure layout helpers in architectureDiagramLayout.ts.
 * No React, no DOM — pure Node.js TypeScript.
 */

import { describe, it, expect } from 'vitest';
import {
  figureNodePosition,
  computeAutoLayout,
  positionNodesFigure,
  resolveEdgeStyle,
  positionGroupsFigure,
  pathPointsToSvgPath,
  FIGURE_CANVAS_W,
  FIGURE_CANVAS_H,
  NODE_W_DEFAULT,
  NODE_H_DEFAULT,
} from './architectureDiagramLayout';
import type { GraphBoundingBox, GraphPoint, GraphGroup } from '@/types/architecture_graph';

function makeNode(id: string, bbox?: GraphBoundingBox, extra: Record<string, unknown> = {}): Parameters<typeof figureNodePosition>[0] {
  return { id, label: id, type: 'other', bbox, ...extra };
}

function makeEdge(
  source: string,
  target: string,
  overrides: Record<string, unknown> = {}
): Parameters<typeof resolveEdgeStyle>[0] {
  return { source, target, ...overrides };
}

describe('figureNodePosition', () => {
  it('returns null when bbox is absent', () => {
    expect(figureNodePosition(makeNode('n1'))).toBeNull();
  });

  it('returns null for NaN coords', () => {
    expect(figureNodePosition(makeNode('n1', { x: NaN, y: 0.1, w: 0.1, h: 0.1 }))).toBeNull();
    expect(figureNodePosition(makeNode('n1', { x: 0, y: NaN, w: 0.1, h: 0.1 }))).toBeNull();
  });

  it('returns null for Infinity coords', () => {
    expect(figureNodePosition(makeNode('n1', { x: Infinity, y: 0.1, w: 0.1, h: 0.1 }))).toBeNull();
  });

  it('returns null for negative width/height', () => {
    expect(figureNodePosition(makeNode('n1', { x: 0.1, y: 0.1, w: -0.1, h: 0.1 }))).toBeNull();
    expect(figureNodePosition(makeNode('n1', { x: 0.1, y: 0.1, w: 0.1, h: 0 }))).toBeNull();
  });

  it('returns null for coords outside [0, 1]', () => {
    expect(figureNodePosition(makeNode('n1', { x: -0.01, y: 0.1, w: 0.1, h: 0.1 }))).toBeNull();
    expect(figureNodePosition(makeNode('n1', { x: 0.1, y: 1.01, w: 0.1, h: 0.1 }))).toBeNull();
  });

  it('returns null for box extending beyond canvas', () => {
    // x=0.9, w=0.15 => x+w=1.05 > 1.0
    expect(figureNodePosition(makeNode('n1', { x: 0.9, y: 0.1, w: 0.15, h: 0.1 }))).toBeNull();
  });

  it('accepts a box at the exact canvas boundary', () => {
    const result = figureNodePosition(makeNode('n1', { x: 0, y: 0, w: 1.0, h: 1.0 }));
    expect(result).not.toBeNull();
    expect(result!.x).toBe(0);
    expect(result!.y).toBe(0);
    expect(result!.w).toBe(FIGURE_CANVAS_W);
    expect(result!.h).toBe(FIGURE_CANVAS_H);
  });

  it('scales a standard bbox to pixel coords', () => {
    const result = figureNodePosition(makeNode('n1', { x: 0.1, y: 0.2, w: 0.2, h: 0.1 }));
    expect(result).not.toBeNull();
    expect(result!.x).toBeCloseTo(100, -1);  // 0.1 * 1000
    expect(result!.y).toBeCloseTo(120, -1);  // 0.2 * 600
    expect(result!.w).toBeCloseTo(200, -1);  // 0.2 * 1000
    expect(result!.h).toBeCloseTo(60, -1);   // 0.1 * 600
  });

  it('enforces minimum width 60 and height 28', () => {
    // tiny box => min enforced
    const result = figureNodePosition(makeNode('n1', { x: 0.1, y: 0.1, w: 0.01, h: 0.01 }));
    expect(result!.w).toBe(60);
    expect(result!.h).toBe(28);
  });

  it('allows epsilon overflow (1e-6)', () => {
    // x=0.9, w=0.100001 => x+w=1.000001 > 1+1e-6 => null
    expect(figureNodePosition(makeNode('n1', { x: 0.9, y: 0.1, w: 0.100001, h: 0.1 }))).toBeNull();
    // x=0.9, w=0.1 => x+w=1.0 <= 1+1e-6 => ok
    expect(figureNodePosition(makeNode('n1', { x: 0.9, y: 0.1, w: 0.1, h: 0.1 }))).not.toBeNull();
  });
});

describe('computeAutoLayout', () => {
  it('returns empty map for empty nodes', () => {
    const result = computeAutoLayout([], []);
    expect(result.size).toBe(0);
  });

  it('places a single root node at layer 0', () => {
    const nodes = [makeNode('n1')];
    const result = computeAutoLayout(nodes, []);
    const pos = result.get('n1');
    expect(pos).toBeDefined();
    expect(pos!.layer).toBe(0);
    expect(pos!.index).toBe(0);
  });

  it('places nodes with no incoming edges at layer 0', () => {
    const nodes = [makeNode('n1'), makeNode('n2')];
    const edges = [{ source: 'n1', target: 'n2' }];
    const result = computeAutoLayout(nodes, edges);
    expect(result.get('n1')!.layer).toBe(0);
    expect(result.get('n2')!.layer).toBe(1);
  });

  it('places diamond pattern correctly', () => {
    //      n1
    //    n2   n3
    //      n4
    const nodes = ['n1', 'n2', 'n3', 'n4'].map((id) => makeNode(id));
    const edges = [
      { source: 'n1', target: 'n2' },
      { source: 'n1', target: 'n3' },
      { source: 'n2', target: 'n4' },
      { source: 'n3', target: 'n4' },
    ];
    const result = computeAutoLayout(nodes, edges);
    expect(result.get('n1')!.layer).toBe(0);
    expect(result.get('n4')!.layer).toBe(2);
  });

  it('x increases with layer', () => {
    const nodes = [makeNode('n1'), makeNode('n2')];
    const edges = [{ source: 'n1', target: 'n2' }];
    const result = computeAutoLayout(nodes, edges);
    const n1x = result.get('n1')!.x;
    const n2x = result.get('n2')!.x;
    expect(n2x).toBeGreaterThan(n1x);
  });

  it('marks source as auto', () => {
    const nodes = [makeNode('n1')];
    const result = computeAutoLayout(nodes, []);
    expect(result.get('n1')!.source).toBe('auto');
  });
});

describe('positionNodesFigure', () => {
  it('uses figure positions when isFigureMode=true and nodes have bboxes', () => {
    const nodes = [makeNode('n1', { x: 0.1, y: 0.1, w: 0.2, h: 0.1 })];
    const result = positionNodesFigure(nodes, true);
    const pos = result.get('n1');
    expect(pos!.source).toBe('figure');
    expect(pos!.x).toBeCloseTo(100, -1);
  });

  it('falls back to auto-layout when isFigureMode=false', () => {
    const nodes = [makeNode('n1', { x: 0.1, y: 0.1, w: 0.2, h: 0.1 })];
    const result = positionNodesFigure(nodes, false);
    expect(result.get('n1')!.source).toBe('auto');
  });

  it('falls back to auto-layout when no nodes have valid bboxes', () => {
    const nodes = [makeNode('n1')];
    const result = positionNodesFigure(nodes, true);
    expect(result.get('n1')!.source).toBe('auto');
  });
});

describe('resolveEdgeStyle', () => {
  const base = { source: 'n1', target: 'n2' };

  it('defaults to blue stroke when isActive=false', () => {
    const result = resolveEdgeStyle(base as Parameters<typeof resolveEdgeStyle>[0], false);
    expect(result.stroke).toBe('#d1d5db');
  });

  it('uses active blue when isActive=true and no style', () => {
    const result = resolveEdgeStyle(base as Parameters<typeof resolveEdgeStyle>[0], true);
    expect(result.stroke).toBe('#3b82f6');
    expect(result.marker).toBe('url(#arrow-active)');
  });

  it('applies residual/dashed style', () => {
    const edge = { ...base, type: 'residual_skip', style: { name: 'residual' } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.stroke).toBe('#f87171');
    expect(result.dasharray).toBe('6 4');
  });

  it('applies cross_attention type with dashed style', () => {
    const edge = { ...base, type: 'cross_attention' } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.dasharray).toBe('5 3');
  });

  it('applies beam style with default width', () => {
    const edge = { ...base, style: { name: 'beam' } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.stroke).toBe('#a78bfa');
    expect(result.width).toBe(2.4);
  });

  it('beam respects explicit positive width', () => {
    const edge = { ...base, style: { name: 'beam', width: 5 } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.width).toBe(5);
  });

  it('applies highlight style', () => {
    const edge = { ...base, style: { name: 'highlight' } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.stroke).toBe('#fbbf24');
  });

  it('applies dotted style', () => {
    const edge = { ...base, style: { name: 'dotted' } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.dasharray).toBe('1 4');
  });

  it('prefers explicit stroke over default color', () => {
    const edge = { ...base, style: { stroke: '#ff0000', name: 'default' } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.stroke).toBe('#ff0000');
  });

  it('ignores empty stroke string', () => {
    const edge = { ...base, style: { stroke: '', name: 'default' } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.stroke).toBe('#d1d5db');
  });

  it('ignores non-positive width', () => {
    const edge = { ...base, style: { width: 0 } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.width).toBe(1.5);
  });

  it('accepts positive width', () => {
    const edge = { ...base, style: { width: 3 } } as Parameters<typeof resolveEdgeStyle>[0];
    const result = resolveEdgeStyle(edge, false);
    expect(result.width).toBe(3);
  });
});

describe('positionGroupsFigure', () => {
  it('returns empty array for undefined groups', () => {
    expect(positionGroupsFigure(undefined)).toEqual([]);
  });

  it('returns empty array for empty groups', () => {
    expect(positionGroupsFigure([])).toEqual([]);
  });

  it('skips groups without valid bboxes', () => {
    const groups: GraphGroup[] = [
      { id: 'g1', label: 'Encoder' },          // no bbox
      { id: 'g2', label: 'Decoder', bbox: { x: 0, y: 0, w: 0.3, h: 0.3 } },
    ];
    const result = positionGroupsFigure(groups);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('g2');
  });

  it('uses label as fallback for missing label field', () => {
    const groups: GraphGroup[] = [
      { id: 'g1', bbox: { x: 0, y: 0, w: 0.3, h: 0.3 } },
    ];
    const result = positionGroupsFigure(groups);
    expect(result[0].label).toBe('g1');
  });

  it('defaults color to #e5e7eb', () => {
    const groups: GraphGroup[] = [
      { id: 'g1', bbox: { x: 0, y: 0, w: 0.3, h: 0.3 } },
    ];
    const result = positionGroupsFigure(groups);
    expect(result[0].color).toBe('#e5e7eb');
  });

  it('scales bbox to pixel coordinates', () => {
    const groups: GraphGroup[] = [
      { id: 'g1', label: 'Group', bbox: { x: 0.1, y: 0.2, w: 0.3, h: 0.1 }, color: '#ff0000' },
    ];
    const result = positionGroupsFigure(groups);
    expect(result[0].x).toBeCloseTo(100, -1);
    expect(result[0].y).toBeCloseTo(120, -1);
    expect(result[0].w).toBeCloseTo(300, -1);
    expect(result[0].h).toBeCloseTo(60, -1);
    expect(result[0].color).toBe('#ff0000');
  });
});

describe('pathPointsToSvgPath', () => {
  it('returns empty string for fewer than 2 points', () => {
    expect(pathPointsToSvgPath([])).toBe('');
    expect(pathPointsToSvgPath([{ x: 0.1, y: 0.2 }])).toBe('');
  });

  it('returns single M/L SVG path for 2 points', () => {
    const points: GraphPoint[] = [
      { x: 0.1, y: 0.2 },
      { x: 0.5, y: 0.5 },
    ];
    const result = pathPointsToSvgPath(points);
    expect(result).toBe(`M ${100},${120} L ${500},${300}`);
  });

  it('returns polyline for multiple points', () => {
    const points: GraphPoint[] = [
      { x: 0.0, y: 0.0 },
      { x: 0.5, y: 0.5 },
      { x: 1.0, y: 1.0 },
    ];
    const result = pathPointsToSvgPath(points);
    expect(result).toBe(`M ${0},${0} L ${500},${300} L ${1000},${600}`);
  });
});
