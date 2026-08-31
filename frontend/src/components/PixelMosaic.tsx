'use client';

import React, { useEffect, useRef } from 'react';

/**
 * Cursor-reactive pixel mosaic for the hero's dark band, in the spirit of
 * the AEye template: a deterministic field of blue blocks that glows and
 * swells around the pointer, with a staggered dissolve-in on load.
 *
 * All per-frame updates go through refs (no React state), so the loop
 * never re-renders. Under prefers-reduced-motion the field renders static.
 */

const SHADES = ['#2B50FF', '#1E3FCC', '#4A6BFF', '#5E7DFF', '#16329E'];
const ACCENT = '#F4F2ED';
const COLS = 30;
const ROWS = 13;
const CELL = 16;
const GLOW_SIGMA = 95; // px falloff radius around the pointer

function rand(i: number): number {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

interface CellSpec {
  color: string;
  base: number; // resting opacity
  delay: number; // entrance stagger, ms
}

const CELLS: CellSpec[] = [];
for (let i = 0; i < COLS * ROWS; i++) {
  const r = rand(i);
  const col = i % COLS;
  const density = 0.15 + (col / COLS) * 0.75; // fade toward the left edge
  if (r < density) {
    const isAccent = r > 0.94;
    CELLS.push({
      color: isAccent ? ACCENT : SHADES[Math.floor(rand(i + 999) * SHADES.length)],
      base: Number((isAccent ? 0.95 : 0.3 + rand(i + 42) * 0.6).toFixed(3)),
      delay: ((Math.floor(i / COLS) + col) % 22) * 55 + rand(i + 555) * 120,
    });
  } else {
    // Empty cells sit almost invisible but still catch the cursor glow
    CELLS.push({ color: '#2B50FF', base: 0.05, delay: 0 });
  }
}

export const PixelMosaic: React.FC<{ className?: string }> = ({ className = '' }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cellRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const els = cellRefs.current.filter((el): el is HTMLDivElement => el !== null);
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (reduced) {
      els.forEach((el, i) => {
        el.style.opacity = String(CELLS[i].base);
      });
      return;
    }

    const header = container.closest('header');
    let mx = -99999;
    let my = -99999;

    const onMove = (e: PointerEvent) => {
      if (!header) return;
      const r = header.getBoundingClientRect();
      mx = e.clientX - r.left;
      my = e.clientY - r.top;
    };
    const onLeave = () => {
      mx = -99999;
      my = -99999;
    };
    header?.addEventListener('pointermove', onMove);
    header?.addEventListener('pointerleave', onLeave);

    const start = performance.now();
    const glow = new Float32Array(CELLS.length);
    let raf = 0;

    const tick = (now: number) => {
      const t = now - start;
      let ox = 0;
      let oy = 0;
      if (header) {
        const hr = header.getBoundingClientRect();
        const cr = container.getBoundingClientRect();
        ox = cr.left - hr.left;
        oy = cr.top - hr.top;
      }
      for (let i = 0; i < CELLS.length; i++) {
        const c = CELLS[i];
        const el = els[i];
        if (!el) continue;
        const col = i % COLS;
        const row = Math.floor(i / COLS);

        // Entrance: dissolve in with per-cell stagger
        const p = Math.min(1, Math.max(0, (t - c.delay) / 450));

        // Cursor glow with exponential falloff, eased via lerp
        const cx = ox + col * CELL + CELL / 2;
        const cy = oy + row * CELL + CELL / 2;
        const dx = cx - mx;
        const dy = cy - my;
        const target = Math.exp(-(dx * dx + dy * dy) / (2 * GLOW_SIGMA * GLOW_SIGMA));
        glow[i] += (target - glow[i]) * 0.14;

        const opacity = c.base * p + glow[i] * (1 - c.base) * 0.9;
        el.style.opacity = opacity.toFixed(3);
        const s = 1 + glow[i] * 0.3;
        el.style.transform = s > 1.01 ? `scale(${s.toFixed(3)})` : '';
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      header?.removeEventListener('pointermove', onMove);
      header?.removeEventListener('pointerleave', onLeave);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      aria-hidden="true"
      className={`grid ${className}`}
      style={{
        gridTemplateColumns: `repeat(${COLS}, ${CELL}px)`,
        gridTemplateRows: `repeat(${ROWS}, ${CELL}px)`,
      }}
    >
      {CELLS.map((c, i) => (
        <div
          key={i}
          ref={(el) => {
            cellRefs.current[i] = el;
          }}
          suppressHydrationWarning
          style={{
            backgroundColor: c.color,
            opacity: String(c.base),
            willChange: 'opacity, transform',
          }}
        />
      ))}
    </div>
  );
};
