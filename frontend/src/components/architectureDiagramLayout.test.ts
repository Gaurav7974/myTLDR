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
  relaxNodePositions,
  generateRoundedOrthogonalPath,
  routeEdgeOrthogonal,
  computeElkLayout,
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

  it('y increases with layer (vertical stack layout)', () => {
    const nodes = [makeNode('n1'), makeNode('n2')];
    const edges = [{ source: 'n1', target: 'n2' }];
    const result = computeAutoLayout(nodes, edges);
    const n1y = result.get('n1')!.y;
    const n2y = result.get('n2')!.y;
    expect(n2y).toBeGreaterThan(n1y);
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

describe('relaxNodePositions', () => {
  it('enforces minimum vertical gap between colliding stacked nodes', () => {
    const map = new Map();
    map.set('a', { id: 'a', x: 100, y: 100, w: 160, h: 40, layer: 0, index: 0, source: 'figure' });
    // Node 'b' touches 'a' at y=140
    map.set('b', { id: 'b', x: 100, y: 140, w: 160, h: 40, layer: 0, index: 0, source: 'figure' });
    const relaxed = relaxNodePositions(map, 16);
    expect(relaxed.get('b')!.y).toBeGreaterThanOrEqual(100 + 40 + 16);
  });

  it('enforces minimum horizontal gap between colliding side-by-side nodes', () => {
    const map = new Map();
    map.set('a', { id: 'a', x: 100, y: 100, w: 60, h: 40, layer: 0, index: 0, source: 'figure' });
    // Node 'b' overlaps horizontally [140, 240] vs [100, 160]
    map.set('b', { id: 'b', x: 140, y: 100, w: 100, h: 40, layer: 0, index: 0, source: 'figure' });
    const relaxed = relaxNodePositions(map, 16, 16);
    expect(relaxed.get('b')!.x).toBeGreaterThanOrEqual(100 + 60 + 16);
  });
});

describe('generateRoundedOrthogonalPath', () => {
  it('returns empty string for < 2 points', () => {
    expect(generateRoundedOrthogonalPath([])).toBe('');
    expect(generateRoundedOrthogonalPath([{ x: 10, y: 10 }])).toBe('');
  });

  it('returns straight line for 2 points', () => {
    expect(generateRoundedOrthogonalPath([{ x: 10, y: 10 }, { x: 10, y: 100 }])).toBe('M 10 10 L 10 100');
  });

  it('inserts quadratic bezier fillets on corners for 3+ points', () => {
    const pts = [{ x: 10, y: 10 }, { x: 50, y: 10 }, { x: 50, y: 100 }];
    const path = generateRoundedOrthogonalPath(pts, 8);
    expect(path).toContain('Q 50 10');
    expect(path.startsWith('M 10 10')).toBe(true);
    expect(path.endsWith('L 50 100')).toBe(true);
  });
});

describe('routeEdgeOrthogonal', () => {
  const nodeA = { id: 'a', x: 100, y: 300, w: 160, h: 40, layer: 0, index: 0, source: 'figure' as const };
  const nodeB = { id: 'b', x: 100, y: 100, w: 160, h: 40, layer: 0, index: 0, source: 'figure' as const };

  it('routes upward sequential flow with straight vertical line', () => {
    const edge = { source: 'a', target: 'b' };
    const routed = routeEdgeOrthogonal(nodeA, nodeB, edge);
    // Node B is directly above Node A: straight line from top of A to bottom of B
    expect(routed.path).toBe('M 180 300 L 180 140');
  });

  it('routes downward sequential flow with straight vertical line', () => {
    const edge = { source: 'b', target: 'a' };
    const routed = routeEdgeOrthogonal(nodeB, nodeA, edge);
    // Node A is directly below Node B: straight line from bottom of B to top of A
    expect(routed.path).toBe('M 180 140 L 180 300');
  });

  it('routes residual edge outside via gutter lane with rounded bends', () => {
    const edge = { source: 'b', target: 'a', type: 'residual_skip' };
    const routed = routeEdgeOrthogonal(nodeB, nodeA, edge, 1000);
    // Left half gutter should step left of min(x)=100 -> gutterX=60
    expect(routed.path).toContain('L 60');
    expect(routed.path).toContain('Q 60');
  });

  it('routes cross-column edge through corridor', () => {
    const leftNode = { id: 'enc', x: 100, y: 200, w: 160, h: 40, layer: 0, index: 0, source: 'figure' as const };
    const rightNode = { id: 'dec', x: 500, y: 250, w: 160, h: 40, layer: 0, index: 0, source: 'figure' as const };
    const edge = { source: 'enc', target: 'dec', type: 'cross_attention' };
    const routed = routeEdgeOrthogonal(leftNode, rightNode, edge, 1000);
    // Corridor between 260 and 500 -> midX = 380
    expect(routed.path).toContain('380');
    expect(routed.path).toContain('Q 380');
  });
});

describe('positionGroupsFigure member-based containment', () => {
  it('encloses member nodes tightly with padding', () => {
    const groups: GraphGroup[] = [{ id: 'g1', label: 'Encoder' }];
    const nodes = [
      { id: 'n1', label: 'Layer 1', type: 'attention', group_id: 'g1' },
      { id: 'n2', label: 'Layer 2', type: 'feedforward', group_id: 'g1' },
    ];
    const posMap = new Map();
    posMap.set('n1', { id: 'n1', x: 100, y: 100, w: 160, h: 40, layer: 0, index: 0, source: 'figure' });
    posMap.set('n2', { id: 'n2', x: 100, y: 200, w: 160, h: 40, layer: 0, index: 0, source: 'figure' });

    const positioned = positionGroupsFigure(groups, posMap, nodes);
    expect(positioned).toHaveLength(1);
    const g = positioned[0];
    // x: minX(100) - 24 = 76
    expect(g.x).toBe(76);
    // y: minY(100) - 32 = 68
    expect(g.y).toBe(68);
    // w: (160) + 48 = 208
    expect(g.w).toBe(208);
    // h: (maxY(240) - minY(100)) + 32 + 20 = 192
    expect(g.h).toBe(192);
  });
});

describe('computeElkLayout', () => {
  it('returns empty result when nodes array is empty', async () => {
    const res = await computeElkLayout([], []);
    expect(res.nodes.size).toBe(0);
    expect(res.edges.size).toBe(0);
    expect(res.groups).toHaveLength(0);
  });

  it('lays out sequential nodes and computes orthogonal edge paths with bend points', async () => {
    const nodes = [
      { id: 'in', label: 'Input', type: 'input' },
      { id: 'mid', label: 'Layer', type: 'linear' },
      { id: 'out', label: 'Output', type: 'output' },
    ];
    const edges = [
      { source: 'in', target: 'mid' },
      { source: 'mid', target: 'out' },
    ];

    const res = await computeElkLayout(nodes, edges);
    expect(res.nodes.size).toBe(3);
    expect(res.nodes.get('in')).toBeDefined();
    expect(res.nodes.get('mid')).toBeDefined();
    expect(res.nodes.get('out')).toBeDefined();

    // Top to bottom flow
    const inY = res.nodes.get('in')!.y;
    const midY = res.nodes.get('mid')!.y;
    const outY = res.nodes.get('out')!.y;
    expect(midY).toBeGreaterThan(inY);
    expect(outY).toBeGreaterThan(midY);

    // Edges routed
    expect(res.edges.size).toBeGreaterThan(0);
    const edge1 = res.edges.get('in->mid');
    expect(edge1).toBeDefined();
    expect(edge1!.path.startsWith('M')).toBe(true);
    expect(typeof edge1!.midX).toBe('number');
    expect(typeof edge1!.midY).toBe('number');
  });

  it('supports compound groups and positions member nodes within group bounds', async () => {
    const groups: GraphGroup[] = [
      { id: 'enc_group', label: 'Encoder' },
    ];
    const nodes = [
      { id: 'src', label: 'Source', type: 'input' },
      { id: 'attn', label: 'Self-Attention', type: 'attention', group_id: 'enc_group' },
      { id: 'ffn', label: 'Feed Forward', type: 'feedforward', group_id: 'enc_group' },
    ];
    const edges = [
      { source: 'src', target: 'attn' },
      { source: 'attn', target: 'ffn' },
    ];

    const res = await computeElkLayout(nodes, edges, groups);
    expect(res.nodes.size).toBe(3);
    expect(res.groups).toHaveLength(1);

    const g = res.groups[0];
    expect(g.id).toBe('enc_group');
    expect(g.label).toBe('Encoder');

    const attnPos = res.nodes.get('attn')!;
    const ffnPos = res.nodes.get('ffn')!;

    // Both member nodes are inside the group rectangle
    expect(attnPos.x).toBeGreaterThanOrEqual(g.x);
    expect(attnPos.y).toBeGreaterThanOrEqual(g.y);
    expect(attnPos.x + attnPos.w).toBeLessThanOrEqual(g.x + g.w + 1);
    expect(ffnPos.y + ffnPos.h).toBeLessThanOrEqual(g.y + g.h + 1);
  });
});

