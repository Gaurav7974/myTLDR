'use client';

import React from 'react';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface HierarchicalGroup3DProps {
  id: string;
  label: string;
  repetitions?: number;
  center: [number, number, number];
  size: [number, number, number];
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export const HierarchicalGroup3D: React.FC<HierarchicalGroup3DProps> = ({
  label,
  repetitions = 1,
  center,
  size,
  isCollapsed,
  onToggleCollapse
}) => {
  const [w, h, d] = size;

  return (
    <group position={center}>
      {/* Translucent Bounding Box Container */}
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onToggleCollapse();
        }}
      >
        <boxGeometry args={[w, h, d]} />
        <meshStandardMaterial
          color="#38BDF8"
          transparent={true}
          opacity={0.06}
          roughness={0.8}
        />
      </mesh>

      {/* Dashed Bounding Edges */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(w, h, d)]} />
        <lineDashedMaterial
          color="#60A5FA"
          dashSize={0.2}
          gapSize={0.1}
          transparent={true}
          opacity={0.4}
        />
      </lineSegments>

      {/* Group Header Tag */}
      <Text
        position={[0, h / 2 + 0.35, 0]}
        fontSize={0.2}
        color="#38BDF8"
        anchorX="center"
        anchorY="middle"
      >
        {`${label} ${repetitions > 1 ? `(× ${repetitions} Stacks)` : ''} [${isCollapsed ? 'Click to Expand +' : 'Click to Collapse -'}]`}
      </Text>
    </group>
  );
};
