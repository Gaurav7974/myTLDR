'use client';

import React, { useEffect, useRef, useState } from 'react';

/**
 * PixelHeaderGrid matching aeye-saas.webflow.io:
 * - Real DOM pixel grid with guaranteed 100% TRUE SQUARE cells (58px x 58px)
 * - Instant hover illumination (0s delay) with smooth 600ms fadeout decay trail
 * - Randomized color accents (#0055FF, #2B50FF, #3B82F6, #00D8FF, #FFFFFF)
 * - Ambient random twinkle
 * - Retro Monospace Nav: <HOME>, ABOUT, PAPERS, DOCS, MODELS [4], DECONSTRUCT PAPER
 * - Mid Banner: ✦ PaperScope v1.0-BETA + SCROLL FOR MORE ↓
 */

interface PixelHeaderGridProps {
  onExploreClick?: () => void;
  onContactClick?: () => void;
}

const PALETTE = ['#0055FF', '#2B50FF', '#3B82F6', '#00D8FF', '#FFFFFF', '#0055FF'];
const CELL_SIZE = 58; // Locked 58px x 58px true square dimensions

export const PixelHeaderGrid: React.FC<PixelHeaderGridProps> = ({
  onExploreClick,
  onContactClick
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [activeLink, setActiveLink] = useState('home');
  const [dimensions, setDimensions] = useState({ cols: 36, rows: 5 });

  useEffect(() => {
    const updateGrid = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.offsetWidth || window.innerWidth;
      const height = containerRef.current.offsetHeight || 220;
      const cols = Math.ceil(width / CELL_SIZE) + 2;
      const rows = Math.ceil(height / CELL_SIZE) + 1;
      setDimensions({ cols, rows });
    };

    updateGrid();
    window.addEventListener('resize', updateGrid);
    return () => window.removeEventListener('resize', updateGrid);
  }, []);

  // Handle cell mouseenter to dynamically assign colors
  const handleCellEnter = (e: React.MouseEvent<HTMLDivElement>) => {
    const colorEl = e.currentTarget.querySelector<HTMLDivElement>('.pixel-box-color');
    if (colorEl) {
      const randomColor = PALETTE[Math.floor(Math.random() * PALETTE.length)];
      colorEl.style.backgroundColor = randomColor;
      colorEl.style.opacity = '1';
      colorEl.style.transition = 'opacity 0s';
    }
  };

  const handleCellLeave = (e: React.MouseEvent<HTMLDivElement>) => {
    const colorEl = e.currentTarget.querySelector<HTMLDivElement>('.pixel-box-color');
    if (colorEl) {
      colorEl.style.opacity = '0';
      colorEl.style.transition = 'opacity 0.65s cubic-bezier(0.16, 1, 0.3, 1) 0.25s';
    }
  };

  // Ambient random pixel twinkle
  useEffect(() => {
    const interval = setInterval(() => {
      if (!containerRef.current) return;
      const allColors = containerRef.current.querySelectorAll<HTMLDivElement>('.pixel-box-color');
      if (!allColors.length) return;

      const randomCell = allColors[Math.floor(Math.random() * allColors.length)];
      if (randomCell && randomCell.style.opacity !== '1') {
        const randomColor = PALETTE[Math.floor(Math.random() * PALETTE.length)];
        randomCell.style.backgroundColor = randomColor;
        randomCell.style.opacity = (0.35 + Math.random() * 0.45).toString();
        randomCell.style.transition = 'opacity 0.1s ease-in';

        setTimeout(() => {
          if (randomCell) {
            randomCell.style.opacity = '0';
            randomCell.style.transition = 'opacity 0.9s cubic-bezier(0.16, 1, 0.3, 1) 0.2s';
          }
        }, 300);
      }
    }, 220);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden bg-[#0D0E11] text-white border-b-2 border-zinc-950 select-none min-h-[200px] md:min-h-[220px] flex flex-col justify-between"
    >
      {/* ============ 100% TRUE SQUARE INTERACTIVE DOM PIXEL GRID ============ */}
      <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-auto">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `repeat(${dimensions.cols}, ${CELL_SIZE}px)`,
            gridTemplateRows: `repeat(${dimensions.rows}, ${CELL_SIZE}px)`,
          }}
        >
          {Array.from({ length: dimensions.rows * dimensions.cols }).map((_, idx) => (
            <div
              key={idx}
              onMouseEnter={handleCellEnter}
              onMouseLeave={handleCellLeave}
              className="border-r border-b border-white/[0.07] relative cursor-crosshair overflow-hidden transition-colors"
              style={{
                width: `${CELL_SIZE}px`,
                height: `${CELL_SIZE}px`,
              }}
            >
              {/* Glowing color layer */}
              <div
                className="pixel-box-color absolute inset-0 pointer-events-none opacity-0"
                style={{
                  backgroundColor: '#0055FF',
                  transition: 'opacity 0.65s cubic-bezier(0.16, 1, 0.3, 1) 0.25s'
                }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ============ TOP MONOSPACE NAVIGATION (Sits above grid) ============ */}
      <nav className="relative z-10 mx-auto flex h-12 md:h-14 w-full max-w-[1400px] items-center justify-between px-4 md:px-8 pointer-events-auto">
        {/* Nav Links Stack — only what the platform needs */}
        <div className="flex items-center gap-1 md:gap-2">
          <button
            onClick={() => {
              setActiveLink('home');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className={`flex items-center px-2.5 py-1 font-mono text-[11px] font-bold tracking-wider uppercase transition-all ${
              activeLink === 'home'
                ? 'bg-[#0055FF] text-white shadow-sm'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            <span className="text-white/70 mr-0.5">&lt;</span>
            HOME
            <span className="text-white/70 ml-0.5">&gt;</span>
          </button>

          <a
            href="#classics"
            onClick={() => setActiveLink('papers')}
            className="hidden sm:inline-block px-3 py-1 font-mono text-[11px] font-medium tracking-wider uppercase text-zinc-400 hover:text-white transition-colors"
          >
            PAPERS
          </a>
        </div>

        {/* Right Action Button */}
        <div className="flex items-center gap-3">
          <a
            href="#input-zone"
            onClick={onContactClick}
            className="flex items-center gap-2 px-3.5 py-1 bg-transparent hover:bg-white hover:text-zinc-950 text-white border border-zinc-700 hover:border-white font-mono text-[11px] font-bold tracking-wider uppercase transition-all active:translate-y-[1px]"
          >
            <svg
              className="w-3.5 h-3.5 fill-current"
              viewBox="0 0 20 20"
            >
              <path d="M4.5 14.1L5.7 15.3L12.8 8.2L14 9.4L15.1 8.2L14 7.1L15.1 5.9L14 4.7L12.8 5.9L11.6 4.7L10.4 5.9L11.6 7.1L4.5 14.1Z" />
            </svg>
            <span>DECONSTRUCT PAPER</span>
          </a>
        </div>
      </nav>

      {/* ============ HERO MID BANNER: BRANDING & SCROLL INDICATOR ============ */}
      <div className="relative z-10 mx-auto w-full max-w-[1400px] px-4 md:px-8 py-5 md:py-7 flex items-center justify-between pointer-events-none">
        {/* Left Brand — unified to myTLDR */}
        <div className="flex items-center gap-3 pointer-events-auto">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 text-[#0055FF] flex items-center justify-center font-mono font-bold text-lg">
              ✦
            </div>
            <span className="font-sans font-black text-2xl md:text-3xl tracking-tight text-white">
              myTLDR
            </span>
            <span className="hidden sm:inline-flex px-2 py-0.5 bg-zinc-900/90 border border-zinc-700/80 text-zinc-300 font-mono text-[10px] font-bold tracking-wider uppercase rounded-xs">
              Paper Decoder
            </span>
          </div>
          <span className="px-2 py-0.5 bg-zinc-900/90 border border-zinc-700/80 text-zinc-400 font-mono text-[10px] font-bold tracking-wider uppercase rounded-xs">
            v1.0-BETA
          </span>
        </div>

        {/* Right Scroll Indicator */}
        <div className="hidden sm:flex items-center gap-2 font-mono text-[11px] font-medium tracking-widest text-zinc-500 uppercase pointer-events-auto">
          <span>SCROLL FOR MORE</span>
          <svg
            className="w-3.5 h-3.5 text-zinc-400 animate-bounce"
            viewBox="0 0 16 16"
            fill="currentColor"
          >
            <path d="M8.7 2.7H7.3V10.7H6V12H7.3V13.3H8.7V12H10V10.7H8.7V2.7ZM11.3 9.3V10.7H10V9.3H11.3ZM4.7 9.3V10.7H6V9.3H4.7Z" />
          </svg>
        </div>
      </div>
    </div>
  );
};
