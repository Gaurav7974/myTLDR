'use client';

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { SceneNode3D } from '@/types/architecture_ir';

interface TensorBlock3DProps {
  node: SceneNode3D;
  isSelected: boolean;
  isActiveStep: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: () => void;
  onUnhover: () => void;
}

export const TensorBlock3D: React.FC<TensorBlock3DProps> = ({
  node,
  isSelected,
  isActiveStep,
  isHovered,
  onSelect,
  onHover,
  onUnhover
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [width, height, depth] = node.size_3d;
  const { rows, cols } = node.grid_subdivisions;

  // Lattice cell grid line segments
  const gridLines = useMemo(() => {
    const lines: [number, number, number][][] = [];
    const zOffset = depth / 2 + 0.005;

    // Horizontal lattice dividers
    for (let r = 1; r < rows; r++) {
      const y = -height / 2 + (r * height) / rows;
      lines.push([
        [-width / 2, y, zOffset],
        [width / 2, y, zOffset]
      ]);
    }

    // Vertical lattice dividers
    for (let c = 1; c < cols; c++) {
      const x = -width / 2 + (c * width) / cols;
      lines.push([
        [x, -height / 2, zOffset],
        [x, height / 2, zOffset]
      ]);
    }

    return lines;
  }, [width, height, depth, rows, cols]);

  // Subtle breathing animation when active
  useFrame((state) => {
    if (!meshRef.current) return;
    if (isActiveStep) {
      const t = state.clock.getElapsedTime();
      const scale = 1 + Math.sin(t * 5) * 0.04;
      meshRef.current.scale.set(scale, scale, scale);
    } else if (isHovered || isSelected) {
      meshRef.current.scale.set(1.05, 1.05, 1.05);
    } else {
      meshRef.current.scale.set(1.0, 1.0, 1.0);
    }
  });

  return (
    <Float
      speed={isSelected || isActiveStep ? 3.0 : 1.2}
      rotationIntensity={0.06}
      floatIntensity={isActiveStep ? 0.35 : 0.15}
    >
      <group
        position={node.position_3d}
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
        {/* Main 3D Matrix Slab */}
        <mesh ref={meshRef}>
          <boxGeometry args={[width, height, depth]} />
          <meshStandardMaterial
            color={node.color_hex}
            metalness={0.4}
            roughness={0.25}
            transparent={true}
            opacity={isSelected || isActiveStep ? 0.95 : isHovered ? 0.85 : 0.65}
            emissive={isSelected || isActiveStep ? node.color_hex : '#000000'}
            emissiveIntensity={isSelected ? 0.45 : isActiveStep ? 0.65 : isHovered ? 0.2 : 0}
          />
        </mesh>

        {/* Outer Bounding Wireframe Outline */}
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(width, height, depth)]} />
          <lineBasicMaterial
            color={isSelected ? '#FFFFFF' : isActiveStep ? '#6EE7B7' : isHovered ? '#93C5FD' : '#475569'}
            linewidth={isSelected || isActiveStep ? 2.5 : 1}
          />
        </lineSegments>

        {/* 2D Cell Grid Lattice Lines on Front Face */}
        {gridLines.map((line, idx) => (
          <Line
            key={idx}
            points={line}
            color={isSelected || isActiveStep ? '#FFFFFF' : '#334155'}
            lineWidth={0.7}
            transparent
            opacity={isSelected || isActiveStep ? 0.6 : 0.3}
          />
        ))}

        {/* Top Node Label */}
        <Text
          position={[0, height / 2 + 0.45, 0]}
          fontSize={0.24}
          color={isSelected ? '#60A5FA' : isActiveStep ? '#34D399' : '#F1F5F9'}
          anchorX="center"
          anchorY="middle"
          maxWidth={width * 1.5}
          textAlign="center"
        >
          {node.label}
        </Text>

        {/* Bottom Dimension Indicator */}
        {node.sub_labels && node.sub_labels.length > 0 && (
          <Text
            position={[0, -height / 2 - 0.3, 0]}
            fontSize={0.16}
            color="#94A3B8"
            anchorX="center"
            anchorY="middle"
          >
            {node.sub_labels.join(' · ')}
          </Text>
        )}
      </group>
    </Float>
  );
};
