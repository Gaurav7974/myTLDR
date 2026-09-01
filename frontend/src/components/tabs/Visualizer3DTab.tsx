'use client';

import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Float, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { PaperData, BycroftStage } from '@/types/paper';
import ArchitectureDiagram from '@/components/ArchitectureDiagram';
import { isVerifiedVisionGraph } from '@/types/architecture_graph';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Layers,
  Info,
  Sliders,
  Grid,
  Box,
  Network,
  Maximize2,
  Minimize2,
  Type,
  Eye,
  Gauge,
  BookOpen,
  ChevronRight,
  ChevronLeft,
  Compass,
  FileText,
  Cpu
} from 'lucide-react';
import { LODLevel } from '@/types/architecture_ir';
import { MathFormula } from './MathDeconstructedTab';

interface Visualizer3DTabProps {
  paper: PaperData;
  onJumpToPage?: (page: number, sectionTitle?: string) => void;
}

// Attention Head Metadata (8 Heads)
const ATTENTION_HEADS = [
  { id: 1, name: 'Head 1', role: 'Syntactic Alignment', description: 'Tracks Subject-Verb relationships and grammatical dependency.' },
  { id: 2, name: 'Head 2', role: 'Direct Objects & Prep', description: 'Attends to prepositional targets and semantic direct objects.' },
  { id: 3, name: 'Head 3', role: 'Positional Proximity', description: 'Attends heavily to neighboring immediately adjacent tokens.' },
  { id: 4, name: 'Head 4', role: 'Pronoun & Coreference', description: 'Resolves pronouns (it, they, you) to original referent entities.' },
  { id: 5, name: 'Head 5', role: 'Long-Range Context', description: 'Captures global document-level semantic topic themes.' },
  { id: 6, name: 'Head 6', role: 'Punctuation & Delimiters', description: 'Attends to periods, commas, and sentence boundaries.' },
  { id: 7, name: 'Head 7', role: 'Morphological Agreement', description: 'Aligns plural nouns, prefixes, and verb tenses.' },
  { id: 8, name: 'Head 8', role: 'Broad Associative Field', description: 'Diffuse associative weighting across the full sequence.' },
];

// =========================================================================
// 1. 3D CAMERA CONTROLLER (Smooth Lerping to Presets & Focused Nodes)
// =========================================================================
function CameraController({
  cameraPreset,
  focusedPosition,
}: {
  cameraPreset: 'overview' | 'attention' | 'mlp' | 'logits' | 'custom';
  focusedPosition: [number, number, number] | null;
}) {
  const { camera } = useThree();
  const targetPosRef = useRef(new THREE.Vector3(2, 3, 20));
  const lookAtTargetRef = useRef(new THREE.Vector3(2, 0, 0));

  useEffect(() => {
    if (cameraPreset === 'overview') {
      targetPosRef.current.set(2, 3.2, 19);
      lookAtTargetRef.current.set(2, 0, 0);
    } else if (cameraPreset === 'attention') {
      targetPosRef.current.set(-1.8, 1.8, 8);
      lookAtTargetRef.current.set(-1.8, 0.3, 0);
    } else if (cameraPreset === 'mlp') {
      targetPosRef.current.set(6.5, 2.0, 9);
      lookAtTargetRef.current.set(6.5, 0, 0);
    } else if (cameraPreset === 'logits') {
      targetPosRef.current.set(11.2, 1.8, 8);
      lookAtTargetRef.current.set(11.2, 0, 0);
    } else if (cameraPreset === 'custom' && focusedPosition) {
      targetPosRef.current.set(
        focusedPosition[0],
        focusedPosition[1] + 1.6,
        focusedPosition[2] + 7.5
      );
      lookAtTargetRef.current.set(
        focusedPosition[0],
        focusedPosition[1],
        focusedPosition[2]
      );
    }
  }, [cameraPreset, focusedPosition]);

  useFrame(() => {
    camera.position.lerp(targetPosRef.current, 0.065);
    camera.lookAt(lookAtTargetRef.current);
  });

  return null;
}

// =========================================================================
// 2. 3D TENSOR SLAB WITH MATRIX CELL SUBDIVISIONS & GLOW
// =========================================================================
function TensorSlab3D({
  stage,
  isSelected,
  isActiveStep,
  isHovered,
  onSelect,
  onHover,
  onUnhover,
}: {
  stage: BycroftStage;
  isSelected: boolean;
  isActiveStep: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: () => void;
  onUnhover: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const size = stage.size3D || [1.8, 2.2, 0.4];

  // Subtle floating animation when active
  useFrame((state) => {
    if (meshRef.current && (isActiveStep || isSelected)) {
      meshRef.current.position.y =
        stage.position3D[1] + Math.sin(state.clock.getElapsedTime() * 3) * 0.08;
    }
  });

  const baseColor = stage.color || '#3B82F6';
  const wireColor = isSelected ? '#FFFFFF' : isActiveStep ? '#60A5FA' : isHovered ? '#38BDF8' : '#94A3B8';

  return (
    <group
      position={stage.position3D}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover();
      }}
      onPointerOut={() => onUnhover()}
    >
      <Float speed={1.2} rotationIntensity={0.06} floatIntensity={0.2}>
        {/* Main Solid Tensor Slab */}
        <mesh ref={meshRef}>
          <boxGeometry args={size} />
          <meshStandardMaterial
            color={isSelected ? '#09090B' : isActiveStep ? '#18181B' : '#27272A'}
            roughness={0.25}
            metalness={0.65}
            emissive={isActiveStep ? baseColor : isSelected ? '#2563EB' : isHovered ? '#1E293B' : '#000000'}
            emissiveIntensity={isActiveStep ? 0.45 : isSelected ? 0.3 : isHovered ? 0.2 : 0.0}
          />
        </mesh>

        {/* Matrix Cell Grid Wireframe (Bycroft Lattice Grid) */}
        <mesh position={[0, 0, size[2] / 2 + 0.01]}>
          <planeGeometry args={[size[0] * 0.94, size[1] * 0.94, 6, 8]} />
          <meshBasicMaterial
            color={wireColor}
            wireframe
            transparent
            opacity={isSelected ? 0.95 : isActiveStep ? 0.8 : isHovered ? 0.6 : 0.25}
          />
        </mesh>

        {/* Glowing Outer Edge Frame */}
        <mesh>
          <boxGeometry args={[size[0] + 0.06, size[1] + 0.06, size[2] + 0.06]} />
          <meshBasicMaterial
            color={wireColor}
            wireframe
            transparent
            opacity={isSelected ? 0.95 : isActiveStep ? 0.75 : isHovered ? 0.5 : 0.15}
          />
        </mesh>

        {/* Top Step Number Badge & Title */}
        <Text
          position={[0, size[1] / 2 + 0.48, 0]}
          fontSize={0.17}
          maxWidth={2.2}
          textAlign="center"
          color={isSelected ? '#09090B' : isActiveStep ? '#1D4ED8' : isHovered ? '#0284C7' : '#334155'}
          anchorX="center"
          anchorY="middle"
          fontWeight="bold"
        >
          {`Step ${stage.stepNumber}: ${stage.name}`}
        </Text>

        {/* Bottom Tensor Shape Subtitle */}
        <Text
          position={[0, -size[1] / 2 - 0.36, 0]}
          fontSize={0.12}
          color="#64748B"
          anchorX="center"
          anchorY="middle"
        >
          {stage.tensorOutput.split(' ')[0] || 'dim: 512'}
        </Text>

        {/* Sub-label tags if available */}
        {stage.subLabels && stage.subLabels[0] && (
          <Text
            position={[0, 0, size[2] / 2 + 0.06]}
            fontSize={0.09}
            color="#E2E8F0"
            anchorX="center"
            anchorY="middle"
            maxWidth={size[0] * 0.85}
            textAlign="center"
          >
            {stage.subLabels[0]}
          </Text>
        )}
      </Float>
    </group>
  );
}

// =========================================================================
// 3. 3D RESIDUAL BYPASS HIGHWAYS (Curved Skip Connections)
// =========================================================================
function ResidualBypassHighway3D({
  start,
  end,
  label,
  color = '#EF4444',
  isActive,
}: {
  start: [number, number, number];
  end: [number, number, number];
  label: string;
  color?: string;
  isActive: boolean;
}) {
  const curvePoints = useMemo(() => {
    const p1 = new THREE.Vector3(...start);
    const p2 = new THREE.Vector3(...end);
    const midX = (start[0] + end[0]) / 2;
    const peakY = Math.max(start[1], end[1]) + 2.4; // Arch upwards
    const controlPoint = new THREE.Vector3(midX, peakY, 0.4);

    const curve = new THREE.QuadraticBezierCurve3(p1, controlPoint, p2);
    return curve.getPoints(32).map((p) => [p.x, p.y, p.z] as [number, number, number]);
  }, [start, end]);

  return (
    <group>
      {/* Curved 3D Bypass Line */}
      <Line
        points={curvePoints}
        color={isActive ? '#DC2626' : color}
        lineWidth={isActive ? 3 : 1.5}
        dashed
        dashScale={8}
        dashSize={0.4}
        gapSize={0.2}
        transparent
        opacity={isActive ? 0.95 : 0.45}
      />

      {/* Floating Highway Badge */}
      <Text
        position={[(start[0] + end[0]) / 2, Math.max(start[1], end[1]) + 2.7, 0.4]}
        fontSize={0.13}
        color={isActive ? '#DC2626' : '#991B1B'}
        anchorX="center"
        anchorY="middle"
      >
        {`⚡ ${label}`}
      </Text>
    </group>
  );
}

// =========================================================================
// 4. MAIN VISUALIZER 3D TAB
// =========================================================================
export const Visualizer3DTab: React.FC<Visualizer3DTabProps> = ({ paper, onJumpToPage }) => {
  const stages = useMemo(() => paper.bycroftStages || [], [paper.bycroftStages]);
  const [selectedStage, setSelectedStage] = useState<BycroftStage>(stages[0]);
  const [activeStepIndex, setActiveStepIndex] = useState<number>(0);
  const [hoveredStageId, setHoveredStageId] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playSpeed, setPlaySpeed] = useState<number>(1); // 0.5x, 1x, 2x
  const [viewMode, setViewMode] = useState<'3d-bycroft' | '2d-graph' | 'heatmap'>('3d-bycroft');
  const [lodLevel, setLodLevel] = useState<LODLevel>(2); // 1=Overview, 2=Blocks, 3=Internals, 4=Math
  const [cameraPreset, setCameraPreset] = useState<'overview' | 'attention' | 'mlp' | 'logits' | 'custom'>('overview');
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [isWalkthroughOpen, setIsWalkthroughOpen] = useState<boolean>(false);
  const [selectedHeadId, setSelectedHeadId] = useState<number>(1);

  // Interactive Sentence Inference Simulation
  const sentencePresets = [
    ['Attention', 'is', 'all', 'you', 'need'],
    ['The', 'Transformer', 'attends', 'to', 'words'],
    ['Self', 'attention', 'replaces', 'recurrent', 'nets'],
  ];
  const [customTokens, setCustomTokens] = useState<string[]>(sentencePresets[0]);
  const [customInputText, setCustomInputText] = useState('Attention is all you need');
  const [temperature, setTemperature] = useState<number>(0.7);

  const handleCustomTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const words = customInputText.trim().split(/\s+/).slice(0, 6);
    while (words.length < 5) words.push(`[token_${words.length + 1}]`);
    setCustomTokens(words);
  };

  // Auto-play forward pass animation cycle
  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = 2400 / playSpeed;
    const interval = setInterval(() => {
      setActiveStepIndex((prev) => {
        const next = (prev + 1) % stages.length;
        setSelectedStage(stages[next]);
        return next;
      });
    }, intervalMs);
    return () => clearInterval(interval);
  }, [isPlaying, stages, playSpeed]);

  const handleStepPrev = useCallback(() => {
    const prev = (activeStepIndex - 1 + stages.length) % stages.length;
    setActiveStepIndex(prev);
    setSelectedStage(stages[prev]);
    setIsPlaying(false);
  }, [activeStepIndex, stages]);

  const handleStepNext = useCallback(() => {
    const next = (activeStepIndex + 1) % stages.length;
    setActiveStepIndex(next);
    setSelectedStage(stages[next]);
    setIsPlaying(false);
  }, [activeStepIndex, stages]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger when typing in inputs
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        setIsPlaying((p) => !p);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        handleStepNext();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        handleStepPrev();
      } else if (e.key === '1') {
        setCameraPreset('overview');
      } else if (e.key === '2') {
        setCameraPreset('attention');
      } else if (e.key === '3') {
        setCameraPreset('mlp');
      } else if (e.key === '4') {
        setCameraPreset('logits');
      } else if (e.key === 'w' || e.key === 'W') {
        setIsWalkthroughOpen((prev) => !prev);
      } else if (e.key === 'f' || e.key === 'F') {
        setIsFullscreen((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleStepNext, handleStepPrev]);

  // Dynamic Attention Heatmap Matrix based on head and tokens
  const dynamicHeatmapMatrix = useMemo(() => {
    const n = customTokens.length;
    const mat: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      let rowSum = 0;
      for (let j = 0; j < n; j++) {
        // Pseudo-attention weight tailored by head
        let weight = 0.1;
        if (selectedHeadId === 1 && (i === j || j === 0)) weight += 0.6; // Head 1: focus on start/self
        else if (selectedHeadId === 2 && j === i - 1) weight += 0.7; // Head 2: prev token
        else if (selectedHeadId === 3 && Math.abs(i - j) <= 1) weight += 0.5; // Head 3: adjacent
        else if (selectedHeadId === 4 && j === 1) weight += 0.65; // Head 4: key entity
        else if (selectedHeadId === 5 && j === n - 1) weight += 0.55; // Head 5: global
        else if (i === j) weight += 0.4;
        else weight += ((i * 7 + j * 13 + selectedHeadId * 5) % 10) / 25;

        row.push(weight);
        rowSum += weight;
      }
      mat.push(row.map((w) => Number((w / rowSum).toFixed(2))));
    }
    return mat;
  }, [customTokens, selectedHeadId]);

  const nextTokens = paper.nextTokenPredictions;

  return (
    <div
      className={`flex flex-col h-full bg-[#FAFAFA] relative overflow-y-auto pb-36 animate-fadeIn ${
        isFullscreen ? 'fixed inset-0 z-50 bg-[#FAFAFA]' : ''
      }`}
    >
      {/* =========================================================================
          1. TOP CONTROLS & BYCROFT WORKSPACE HEADER
          ========================================================================= */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 px-6 bg-white border-b border-zinc-200 z-10 shrink-0">
        {/* Title & Badge */}
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-zinc-900 text-white flex items-center justify-center shadow-xs">
            <Layers className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-bold text-zinc-950 text-xs font-mono uppercase">
                3D Neural Tensor Visualizer
              </h4>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-700 font-bold border border-zinc-200">
                bbycroft.net/llm
              </span>
            </div>
            <p className="text-[11px] text-zinc-500 font-medium">
              {stages.length}-stage computational matrix flow, 3D residual bypass highways & forward pass
            </p>
          </div>
        </div>

        {/* View Mode Switcher, Walkthrough Toggle & Fullscreen Button */}
        <div className="flex items-center gap-2">
          {/* Guided Walkthrough Tour Button */}
          <button
            onClick={() => setIsWalkthroughOpen(!isWalkthroughOpen)}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs font-mono font-semibold rounded-md border transition-all shadow-2xs ${
              isWalkthroughOpen
                ? 'bg-zinc-900 text-white border-zinc-900'
                : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-100'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Walkthrough Tour</span>
          </button>

          {/* Level-of-Detail (LOD 1-4) Selector */}
          <div className="hidden sm:flex items-center bg-zinc-100 rounded-md p-0.5 border border-zinc-200">
            {([
              { id: 1 as LODLevel, label: 'L1 Macro', title: 'Level 1: Macro Pipeline Overview' },
              { id: 2 as LODLevel, label: 'L2 Blocks', title: 'Level 2: Architecture Blocks & Residuals' },
              { id: 3 as LODLevel, label: 'L3 Internals', title: 'Level 3: Tensor Projections & Q/K/V' },
              { id: 4 as LODLevel, label: 'L4 Math', title: 'Level 4: Mathematical Equations' },
            ] as const).map((lod) => (
              <button
                key={lod.id}
                onClick={() => setLodLevel(lod.id)}
                className={`px-2 py-1 text-[10px] font-mono rounded transition-all ${
                  lodLevel === lod.id
                    ? 'bg-zinc-900 text-white font-bold shadow-2xs'
                    : 'text-zinc-600 hover:text-zinc-900'
                }`}
                title={lod.title}
              >
                {lod.label}
              </button>
            ))}
          </div>

          <div className="flex items-center bg-zinc-100 rounded-md p-0.5 border border-zinc-200">
            <button
              onClick={() => setViewMode('3d-bycroft')}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-all ${
                viewMode === '3d-bycroft'
                  ? 'bg-white text-zinc-950 font-semibold shadow-2xs'
                  : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              <Box className="w-3.5 h-3.5" />
              <span>3D Spatial Flow</span>
            </button>

            <button
              onClick={() => setViewMode('2d-graph')}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-all ${
                viewMode === '2d-graph'
                  ? 'bg-white text-zinc-950 font-semibold shadow-2xs'
                  : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              <Network className="w-3.5 h-3.5" />
              <span>2D DAG Graph</span>
            </button>

            <button
              onClick={() => setViewMode('heatmap')}
              className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-all ${
                viewMode === 'heatmap'
                  ? 'bg-white text-zinc-950 font-semibold shadow-2xs'
                  : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              <Grid className="w-3.5 h-3.5" />
              <span>Attention Heatmap</span>
            </button>
          </div>

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 rounded-md bg-zinc-100 hover:bg-zinc-200 border border-zinc-200 text-zinc-700 transition-all shadow-2xs"
            title={isFullscreen ? 'Exit Fullscreen (F)' : 'Fullscreen Canvas (F)'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* =========================================================================
          2. FORWARD-PASS SCRUBBER & CAMERA PRESETS BAR
          ========================================================================= */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-2 bg-zinc-50 border-b border-zinc-200/80 text-xs font-mono shrink-0">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
            Forward Pass:
          </span>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-1 px-3 py-1 rounded bg-zinc-900 text-white font-semibold hover:bg-zinc-800 transition-all shadow-2xs"
          >
            {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3 fill-current" />}
            <span>{isPlaying ? 'Pause' : 'Play Flow'}</span>
          </button>

          <div className="flex items-center gap-1 ml-1">
            <button
              onClick={handleStepPrev}
              className="p-1 rounded bg-white hover:bg-zinc-200 border border-zinc-200 text-zinc-700 transition-all shadow-2xs"
              title="Step Back (←)"
            >
              <SkipBack className="w-3 h-3" />
            </button>
            <button
              onClick={handleStepNext}
              className="p-1 rounded bg-white hover:bg-zinc-200 border border-zinc-200 text-zinc-700 transition-all shadow-2xs"
              title="Step Forward (→)"
            >
              <SkipForward className="w-3 h-3" />
            </button>
          </div>

          {/* Speed Selector */}
          <div className="flex items-center bg-white rounded border border-zinc-200 p-0.5 ml-2">
            {[0.5, 1, 2].map((spd) => (
              <button
                key={spd}
                onClick={() => setPlaySpeed(spd)}
                className={`px-1.5 py-0.5 text-[10px] rounded ${
                  playSpeed === spd ? 'bg-zinc-900 text-white font-bold' : 'text-zinc-600 hover:text-zinc-900'
                }`}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>

        {/* 3D Camera Presets */}
        {viewMode === '3d-bycroft' && (
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-zinc-400 font-semibold uppercase flex items-center gap-1">
              <Eye className="w-3 h-3 text-zinc-500" />
              <span>Camera:</span>
            </span>
            <button
              onClick={() => setCameraPreset('overview')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
                cameraPreset === 'overview'
                  ? 'bg-zinc-900 text-white border-zinc-900 font-bold'
                  : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-100'
              }`}
              title="Overview (1)"
            >
              🔭 Overview
            </button>
            <button
              onClick={() => setCameraPreset('attention')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
                cameraPreset === 'attention'
                  ? 'bg-zinc-900 text-white border-zinc-900 font-bold'
                  : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-100'
              }`}
              title="Attention (2)"
            >
              ⚡ Attention
            </button>
            <button
              onClick={() => setCameraPreset('mlp')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
                cameraPreset === 'mlp'
                  ? 'bg-zinc-900 text-white border-zinc-900 font-bold'
                  : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-100'
              }`}
              title="MLP 4x (3)"
            >
              🧠 MLP 4x
            </button>
            <button
              onClick={() => setCameraPreset('logits')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
                cameraPreset === 'logits'
                  ? 'bg-zinc-900 text-white border-zinc-900 font-bold'
                  : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-100'
              }`}
              title="Logits (4)"
            >
              🎯 Logits
            </button>
          </div>
        )}
      </div>

      {/* =========================================================================
          3. MAIN INTERACTIVE VISUALIZER CANVAS (3D THREE.JS / 2D FLOW / HEATMAP)
          ========================================================================= */}
      <div
        className={`${
          isFullscreen ? 'h-[75vh]' : 'h-[400px] md:h-[460px]'
        } w-full relative bg-[#09090B] border-b border-zinc-200 shrink-0 overflow-hidden`}
      >
        {viewMode === '3d-bycroft' && (
          <>
            <Canvas
              camera={{ position: [2, 3, 20], fov: 40 }}
              className="w-full h-full cursor-grab active:cursor-grabbing bg-[#09090B]"
            >
              <ambientLight intensity={0.9} />
              <pointLight position={[10, 15, 10]} intensity={1.4} />
              <pointLight position={[-10, -5, -5]} intensity={0.6} />
              <directionalLight position={[-12, 16, 8]} intensity={1.1} />
              <directionalLight position={[12, -8, -6]} intensity={0.5} />

              {/* Smooth Camera Controller */}
              <CameraController
                cameraPreset={cameraPreset}
                focusedPosition={selectedStage?.position3D || null}
              />

              {/* Render All 11 3D Spatial Tensor Slabs */}
              {stages.map((stage, idx) => (
                <TensorSlab3D
                  key={stage.id}
                  stage={stage}
                  isSelected={selectedStage?.id === stage.id}
                  isActiveStep={activeStepIndex === idx}
                  isHovered={hoveredStageId === stage.id}
                  onSelect={() => {
                    setSelectedStage(stage);
                    setActiveStepIndex(idx);
                    setCameraPreset('custom');
                  }}
                  onHover={() => setHoveredStageId(stage.id)}
                  onUnhover={() => setHoveredStageId(null)}
                />
              ))}

              {/* Dynamic 3D Residual / Skip Connection Highways */}
              {(paper.residualHighways || [
                { id: 'res-1', startStep: 1, endStep: 6, label: 'Residual Highway 1', color: '#EF4444' },
                { id: 'res-2', startStep: 6, endStep: 10, label: 'Residual Highway 2', color: '#EF4444' }
              ]).map((hw) => {
                const startStage = stages.find(s => s.stepNumber === hw.startStep) || stages[0];
                const endStage = stages.find(s => s.stepNumber === hw.endStep) || stages[stages.length - 1];
                if (!startStage || !endStage) return null;
                const startHeight = startStage.size3D ? startStage.size3D[1] : 2.2;
                const endHeight = endStage.size3D ? endStage.size3D[1] : 2.2;
                const startPos: [number, number, number] = [
                  startStage.position3D[0],
                  startStage.position3D[1] + (startHeight / 2),
                  startStage.position3D[2]
                ];
                const endPos: [number, number, number] = [
                  endStage.position3D[0],
                  endStage.position3D[1] + (endHeight / 2),
                  endStage.position3D[2]
                ];
                const isActive = activeStepIndex === (hw.endStep - 1);
                return (
                  <ResidualBypassHighway3D
                    key={hw.id}
                    start={startPos}
                    end={endPos}
                    label={hw.label}
                    color={hw.color || '#EF4444'}
                    isActive={isActive}
                  />
                );
              })}

              <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                minDistance={3}
                maxDistance={35}
                target={[2, 0, 0]}
              />
            </Canvas>

            {/* GUIDED WALKTHROUGH FLOATING STORY CARD (bbycroft.net/llm style) */}
            {isWalkthroughOpen && selectedStage && (
              <div className="absolute top-4 right-4 max-w-sm w-full bg-white/95 backdrop-blur-md rounded-lg border border-zinc-300 shadow-xl p-4 space-y-3 z-20 animate-fadeIn text-zinc-900 font-sans">
                {/* Header & Step Counter */}
                <div className="flex items-center justify-between border-b border-zinc-200/80 pb-2">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-zinc-900 text-white font-mono text-[10px] font-bold">
                      {selectedStage.stepNumber} / {stages.length}
                    </span>
                    <span className="text-xs font-mono font-bold text-zinc-900 truncate max-w-[170px]">
                      {selectedStage.name}
                    </span>
                  </div>
                  <button
                    onClick={() => setIsWalkthroughOpen(false)}
                    className="text-zinc-400 hover:text-zinc-900 text-xs font-mono"
                  >
                    ✕
                  </button>
                </div>

                {/* Mathematical Equation Formula */}
                <div className="py-2 px-3 rounded bg-zinc-50 border border-zinc-200 text-center font-mono overflow-x-auto min-h-[44px] flex items-center justify-center">
                  <MathFormula latex={selectedStage.mathFormula} displayMode={true} className="text-xs" />
                </div>

                {/* Pedagogical Narrative */}
                <div className="space-y-1.5">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 font-bold flex items-center gap-1">
                    <Compass className="w-3 h-3 text-zinc-700" />
                    <span>How it works:</span>
                  </span>
                  <p className="text-xs text-zinc-700 leading-relaxed font-normal">
                    {selectedStage.detailedWalkthrough || selectedStage.description}
                  </p>
                </div>

                {/* Concrete Vector Numbers Example */}
                {selectedStage.concreteExample && (
                  <div className="p-2.5 rounded bg-zinc-100/90 border border-zinc-200 font-mono text-[10px] text-zinc-800 space-y-1">
                    <span className="text-[9px] uppercase tracking-wider text-zinc-500 font-bold block">
                      Concrete Vector Calculation:
                    </span>
                    <p className="text-zinc-900 leading-snug">{selectedStage.concreteExample}</p>
                  </div>
                )}

                {/* Provenance & Bi-directional PDF Source Linking */}
                <div className="p-2.5 rounded bg-zinc-50 border border-zinc-200/90 flex items-center justify-between text-[11px] font-mono">
                  <div className="flex items-center gap-1.5 text-zinc-700 truncate mr-2">
                    <FileText className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
                    <span className="truncate">
                      Page {selectedStage.stepNumber <= 2 ? 3 : selectedStage.stepNumber <= 6 ? 4 : 5} · {selectedStage.stepNumber <= 2 ? 'Sec 3.1' : selectedStage.stepNumber <= 6 ? 'Sec 3.2' : 'Sec 3.3'}
                    </span>
                  </div>
                  {onJumpToPage && (
                    <button
                      onClick={() => onJumpToPage(selectedStage.stepNumber <= 2 ? 3 : selectedStage.stepNumber <= 6 ? 4 : 5, selectedStage.name)}
                      className="px-2 py-0.5 rounded bg-zinc-900 hover:bg-zinc-800 text-white text-[10px] font-semibold transition-all shadow-2xs shrink-0 active:scale-95"
                      title="Jump directly to source page in PDF"
                    >
                      View Source ↗
                    </button>
                  )}
                </div>

                {/* Navigation Buttons */}
                <div className="flex items-center justify-between pt-2 border-t border-zinc-200/80">
                  <button
                    onClick={handleStepPrev}
                    className="flex items-center gap-1 px-2.5 py-1 rounded bg-zinc-100 hover:bg-zinc-200 text-zinc-800 text-xs font-mono font-semibold transition-all shadow-2xs"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Previous</span>
                  </button>

                  <button
                    onClick={handleStepNext}
                    className="flex items-center gap-1 px-3 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-mono font-semibold transition-all shadow-2xs"
                  >
                    <span>Next Step</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {viewMode === '2d-graph' && (
          <div className="w-full h-full bg-white">
            {paper.architectureGraph && (paper.architectureGraph.nodes?.length ?? 0) > 0 ? (
              <ArchitectureDiagram
                graph={paper.architectureGraph}
                height={720}
              />
            ) : (
              // Trust-first fallback: no static Transformer mock. The
              // diagram tab shows an honest unavailable state and points
              // the user to the parsed text / 3D scene instead.
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
                    Other tabs (3D scene, parsed text) still work; we deliberately
                    do not render a placeholder diagram that could misrepresent
                    the paper.
                  </p>
                  {isVerifiedVisionGraph(paper.architectureGraph) === false &&
                   paper.architectureGraph &&
                   paper.architectureGraph.verification &&
                   !paper.architectureGraph.verification.valid && (
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
        )}

        {viewMode === 'heatmap' && (
          <div className="w-full h-full bg-white p-6 overflow-y-auto flex flex-col justify-start gap-6 animate-fadeIn">
            {/* Interactive Sentence Prompt Input */}
            <div className="space-y-2 max-w-2xl mx-auto w-full">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-zinc-900 uppercase tracking-wider flex items-center gap-1.5">
                  <Type className="w-3.5 h-3.5 text-zinc-700" />
                  <span>Input Token Sequence</span>
                </span>
                <div className="flex items-center gap-1 text-[11px] font-mono text-zinc-500">
                  <span>Presets:</span>
                  {sentencePresets.map((preset, pIdx) => (
                    <button
                      key={pIdx}
                      onClick={() => {
                        setCustomTokens(preset);
                        setCustomInputText(preset.join(' '));
                      }}
                      className="px-2 py-0.5 rounded bg-zinc-100 hover:bg-zinc-200 text-zinc-800 text-[10px] font-mono transition-colors"
                    >
                      P{pIdx + 1}
                    </button>
                  ))}
                </div>
              </div>

              <form onSubmit={handleCustomTextSubmit} className="flex gap-2">
                <input
                  type="text"
                  value={customInputText}
                  onChange={(e) => setCustomInputText(e.target.value)}
                  placeholder="Enter words to compute attention matrix..."
                  className="flex-1 px-3 py-1.5 bg-zinc-50 border border-zinc-300 rounded-md text-xs font-mono text-zinc-900 focus:bg-white focus:outline-none focus:ring-1 focus:ring-zinc-900 transition-all"
                />
                <button
                  type="submit"
                  className="px-3 py-1.5 bg-zinc-900 text-white rounded-md text-xs font-mono font-semibold hover:bg-zinc-800 transition-all"
                >
                  Recalculate
                </button>
              </form>
            </div>

            {/* 8 Attention Heads Selector Bar */}
            <div className="space-y-2 max-w-2xl mx-auto w-full border-t border-zinc-100 pt-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono font-bold text-zinc-800 uppercase tracking-wider flex items-center gap-1">
                  <Cpu className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Multi-Head Attention Heads (h = 8 parallel subspaces)</span>
                </span>
                <span className="text-[10px] font-mono text-zinc-500">d_k = 64 each</span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5">
                {ATTENTION_HEADS.map((head) => (
                  <button
                    key={head.id}
                    onClick={() => setSelectedHeadId(head.id)}
                    className={`p-2 rounded-md border text-left transition-all ${
                      selectedHeadId === head.id
                        ? 'bg-zinc-900 text-white border-zinc-900 shadow-2xs'
                        : 'bg-zinc-50 hover:bg-zinc-100 text-zinc-800 border-zinc-200'
                    }`}
                  >
                    <div className="text-[10px] font-mono font-bold">{head.name}</div>
                    <div className="text-[9px] font-mono truncate text-zinc-400">{head.role}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col lg:flex-row items-center justify-center gap-10">
              {/* Scaled Dot-Product Attention Heatmap Matrix */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-mono text-zinc-900 text-xs uppercase tracking-wider font-bold">
                    Scaled Dot-Product Attention Heatmap ({ATTENTION_HEADS[selectedHeadId - 1]?.role})
                  </h4>
                </div>

                <div className="inline-block p-4 rounded-lg bg-zinc-50 border border-zinc-200 shadow-2xs">
                  {/* Top Key Token Header */}
                  <div className="flex gap-1.5 ml-14 mb-2">
                    {customTokens.map((t, idx) => (
                      <span
                        key={idx}
                        className="w-8 text-center font-mono text-[10px] text-zinc-500 font-semibold truncate"
                        title={t}
                      >
                        {t.slice(0, 4)}
                      </span>
                    ))}
                  </div>

                  {/* Matrix Rows */}
                  {dynamicHeatmapMatrix.map((row, rIdx) => (
                    <div key={rIdx} className="flex items-center gap-1.5 mb-1.5">
                      <span
                        className="w-12 text-right font-mono text-[10px] text-zinc-600 font-semibold truncate pr-2"
                        title={customTokens[rIdx] || ''}
                      >
                        Q: {customTokens[rIdx]?.slice(0, 5) || `T${rIdx}`}
                      </span>

                      {row.map((val, cIdx) => {
                        const opacity = Math.max(0.12, val);
                        return (
                          <div
                            key={cIdx}
                            title={`Query: "${customTokens[rIdx]}" → Key: "${customTokens[cIdx]}": ${(
                              val * 100
                            ).toFixed(1)}%`}
                            className="w-8 h-8 rounded flex items-center justify-center text-[10px] font-mono font-bold transition-transform hover:scale-110 hover:ring-2 hover:ring-zinc-900 cursor-pointer shadow-2xs"
                            style={{
                              backgroundColor: `rgba(24, 24, 27, ${opacity})`,
                              color: val > 0.35 ? '#FFFFFF' : '#18181B',
                            }}
                          >
                            {val.toFixed(2).replace('0.', '.')}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>

              {/* Next-Token Top-5 Probability Distribution */}
              {nextTokens && (
                <div className="space-y-3 w-full max-w-xs">
                  <div className="flex items-center justify-between">
                    <h4 className="font-mono text-zinc-900 text-xs uppercase tracking-wider font-bold">
                      Next-Token Logits
                    </h4>
                    <div className="flex items-center gap-1 bg-white border border-zinc-200 rounded p-0.5 text-[10px] font-mono">
                      <span className="text-zinc-400 pl-1">T=</span>
                      {[0.2, 0.7, 1.0].map((tVal) => (
                        <button
                          key={tVal}
                          onClick={() => setTemperature(tVal)}
                          className={`px-1.5 py-0.2 rounded ${
                            temperature === tVal
                              ? 'bg-zinc-900 text-white font-bold'
                              : 'text-zinc-600 hover:text-zinc-900'
                          }`}
                        >
                          {tVal}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2.5 p-4 rounded-lg bg-zinc-50 border border-zinc-200 shadow-2xs">
                    {nextTokens.map((item, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="font-bold text-zinc-900">
                            {idx + 1}. &ldquo;{item.token}&rdquo;
                          </span>
                          <span className="text-zinc-600">{(item.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-zinc-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-zinc-900 rounded-full transition-all duration-500"
                            style={{ width: `${item.probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* =========================================================================
          4. KEYBOARD SHORTCUTS BAR
          ========================================================================= */}
      <div className="flex items-center justify-center gap-4 px-6 py-2 bg-zinc-100/70 border-b border-zinc-200 text-[10px] font-mono text-zinc-500">
        <span><kbd className="px-1.5 py-0.5 rounded bg-white border border-zinc-300 font-bold text-zinc-700 shadow-2xs">Space</kbd> Play/Pause</span>
        <span><kbd className="px-1.5 py-0.5 rounded bg-white border border-zinc-300 font-bold text-zinc-700 shadow-2xs">← / →</kbd> Step Pipeline</span>
        <span><kbd className="px-1.5 py-0.5 rounded bg-white border border-zinc-300 font-bold text-zinc-700 shadow-2xs">1-4</kbd> Cameras</span>
        <span><kbd className="px-1.5 py-0.5 rounded bg-white border border-zinc-300 font-bold text-zinc-700 shadow-2xs">W</kbd> Guided Tour</span>
        <span><kbd className="px-1.5 py-0.5 rounded bg-white border border-zinc-300 font-bold text-zinc-700 shadow-2xs">F</kbd> Fullscreen</span>
      </div>

      {/* =========================================================================
          5. DEEP-DIVE TENSOR & MATHEMATICS INSPECTOR CARD
          ========================================================================= */}
      {selectedStage && (
        <div className="p-5 md:p-6 space-y-4 bg-white border-t border-zinc-200">
          {/* Header Bar */}
          <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
            <div className="flex items-center gap-2.5">
              <span className="w-7 h-7 rounded-md bg-zinc-900 text-white flex items-center justify-center text-xs font-mono font-bold shadow-xs">
                {selectedStage.stepNumber}
              </span>
              <div>
                <h3 className="font-bold text-zinc-950 text-sm md:text-base">{selectedStage.name}</h3>
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  {selectedStage.category}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Provenance Badge & PDF Jump Button */}
              {onJumpToPage && (
                <button
                  onClick={() => onJumpToPage(selectedStage.stepNumber <= 2 ? 3 : selectedStage.stepNumber <= 6 ? 4 : 5, selectedStage.name)}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-white text-xs font-mono font-semibold transition-all shadow-2xs active:scale-95"
                  title="Navigate PDF viewer directly to source page"
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>Page {selectedStage.stepNumber <= 2 ? 3 : selectedStage.stepNumber <= 6 ? 4 : 5} in PDF ↗</span>
                </button>
              )}

              {selectedStage.flops && (
                <div className="hidden sm:flex items-center gap-1 text-[11px] font-mono px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-200 text-zinc-700">
                  <Gauge className="w-3.5 h-3.5 text-zinc-500" />
                  <span>{selectedStage.flops}</span>
                </div>
              )}
              <div className="flex items-center gap-1 text-[11px] font-mono px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-200 text-zinc-700">
                <Sliders className="w-3.5 h-3.5 text-zinc-500" />
                <span>{selectedStage.parameters}</span>
              </div>
            </div>
          </div>

          {/* Mathematical Formulation with KaTeX */}
          <div className="p-4 rounded-md bg-zinc-50 border border-zinc-200/90 text-center font-mono text-sm text-zinc-900 py-4 flex items-center justify-center overflow-x-auto min-h-[64px]">
            <MathFormula latex={selectedStage.mathFormula} displayMode={true} className="text-sm md:text-base font-serif" />
          </div>

          {/* Tensor Dimension Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3.5 rounded-md bg-white border border-zinc-200 space-y-1 shadow-2xs">
              <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 font-semibold">
                Input Tensor Dimensions
              </span>
              <p className="text-xs font-mono font-bold text-zinc-900">{selectedStage.tensorInput}</p>
            </div>

            <div className="p-3.5 rounded-md bg-white border border-zinc-200 space-y-1 shadow-2xs">
              <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 font-semibold">
                Output Tensor Dimensions
              </span>
              <p className="text-xs font-mono font-bold text-zinc-900">{selectedStage.tensorOutput}</p>
            </div>
          </div>

          {/* Detailed Walkthrough Narrative */}
          {selectedStage.detailedWalkthrough && (
            <div className="p-4 rounded-md bg-zinc-50/90 border border-zinc-200 space-y-2">
              <div className="flex items-center gap-1.5 text-zinc-900 font-mono text-[11px] uppercase tracking-wider font-bold">
                <BookOpen className="w-3.5 h-3.5 text-zinc-700" />
                <span>Granular Operation Walkthrough</span>
              </div>
              <p className="text-xs text-zinc-700 leading-relaxed font-sans">
                {selectedStage.detailedWalkthrough}
              </p>
            </div>
          )}

          {/* Concrete Numbers Example */}
          {selectedStage.concreteExample && (
            <div className="p-3.5 rounded-md bg-zinc-100/80 border border-zinc-200 font-mono text-xs text-zinc-900 space-y-1">
              <span className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold block">
                Concrete Vector Calculation:
              </span>
              <p className="text-zinc-800 leading-relaxed">{selectedStage.concreteExample}</p>
            </div>
          )}

          {/* Sub-Operations / Matrix Slices */}
          {selectedStage.subLabels && selectedStage.subLabels.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider font-semibold">
                Sub-Operations:
              </span>
              {selectedStage.subLabels.map((lbl, idx) => (
                <span
                  key={idx}
                  className="px-2 py-0.5 rounded-md bg-zinc-100 border border-zinc-200 text-zinc-800 text-[11px] font-mono font-medium"
                >
                  {lbl}
                </span>
              ))}
            </div>
          )}

          {/* Mechanical Intuition & Explanation */}
          <div className="p-4 rounded-md bg-zinc-50/70 border border-zinc-200/80 space-y-2">
            <div className="flex items-center gap-1.5 text-zinc-900 font-mono text-[11px] uppercase tracking-wider font-bold">
              <Info className="w-3.5 h-3.5 text-zinc-700" />
              <span>Intuition: What happens to the vectors here?</span>
            </div>
            <p className="text-xs text-zinc-700 leading-relaxed font-sans">
              {selectedStage.description}
            </p>
            <p className="text-xs text-zinc-600 leading-relaxed italic font-serif pt-1.5 border-t border-zinc-200/60">
              &ldquo;{selectedStage.intuition}&rdquo;
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
