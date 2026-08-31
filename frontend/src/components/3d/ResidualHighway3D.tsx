'use client';

import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface ResidualHighway3DProps {
  start: [number, number, number];
  end: [number, number, number];
  label?: string;
  color?: string;
  isActive?: boolean;
}

export const ResidualHighway3D: React.FC<ResidualHighway3DProps> = ({
  start,
  end,
  label = 'Residual Highway',
  color = '#EF4444',
  isActive = false
}) => {
  const pulseRef = useRef<THREE.Mesh>(null);

  // Compute Quadratic 3D Bezier Curve
  const { curve, midPoint } = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const midX = (start[0] + end[0]) / 2;
    const distX = Math.abs(end[0] - start[0]);
    const arcHeight = Math.min(4.8, 1.4 + distX * 0.22);
    const controlVec = new THREE.Vector3(midX, arcHeight, 0.4);

    const quadCurve = new THREE.QuadraticBezierCurve3(startVec, controlVec, endVec);
    const mid = quadCurve.getPoint(0.5);

    return { curve: quadCurve, midPoint: mid };
  }, [start, end]);

  // Create smooth 3D tube geometry along curve
  const tubeGeometry = useMemo(() => {
    return new THREE.TubeGeometry(curve, 48, isActive ? 0.05 : 0.03, 8, false);
  }, [curve, isActive]);

  // Animated pulse along the curve
  useFrame((state) => {
    if (!pulseRef.current) return;
    const t = (state.clock.getElapsedTime() * 0.6) % 1;
    const pos = curve.getPoint(t);
    pulseRef.current.position.copy(pos);
  });

  return (
    <group>
      {/* Main 3D Curved Tube */}
      <mesh geometry={tubeGeometry}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.8 : 0.3}
          transparent={true}
          opacity={isActive ? 0.95 : 0.7}
          roughness={0.2}
          metalness={0.5}
        />
      </mesh>

      {/* Animated Glowing Pulse Particle */}
      <mesh ref={pulseRef}>
        <sphereGeometry args={[isActive ? 0.12 : 0.08, 16, 16]} />
        <meshStandardMaterial
          color="#FFFFFF"
          emissive={color}
          emissiveIntensity={1.5}
          roughness={0.1}
        />
      </mesh>

      {/* Floating Highway Label */}
      {label && (
        <Text
          position={[midPoint.x, midPoint.y + 0.35, midPoint.z]}
          fontSize={0.18}
          color={isActive ? '#FCA5A5' : '#EF4444'}
          anchorX="center"
          anchorY="middle"
        >
          {label}
        </Text>
      )}
    </group>
  );
};
