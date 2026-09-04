'use client';

import React from 'react';

export interface UglyShrimpLoaderProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'default';
  color?: string;
  className?: string;
}

const SIZE_MAP = {
  xs: { box: 'w-3 h-3', scale: 'scale-[0.24]' },
  sm: { box: 'w-4 h-4', scale: 'scale-[0.32]' },
  md: { box: 'w-6 h-6', scale: 'scale-[0.48]' },
  lg: { box: 'w-10 h-10', scale: 'scale-[0.80]' },
  default: { box: 'w-[50px] h-[50px]', scale: 'scale-100' },
};

export const UglyShrimpLoader: React.FC<UglyShrimpLoaderProps> = ({
  size = 'sm',
  color,
  className = '',
}) => {
  const config = SIZE_MAP[size] || SIZE_MAP.sm;

  return (
    <div
      className={`relative inline-flex items-center justify-center shrink-0 ${config.box} ${className}`}
      style={color ? ({ ['--loader-color' as string]: color } as React.CSSProperties) : undefined}
      aria-label="Loading..."
      role="status"
    >
      <div className={`ugly-shrimp-loader transform ${config.scale} origin-center pointer-events-none`}>
        <div className="ugly-shrimp-square ugly-shrimp-sq1" />
        <div className="ugly-shrimp-square ugly-shrimp-sq2" />
        <div className="ugly-shrimp-square ugly-shrimp-sq3" />
        <div className="ugly-shrimp-square ugly-shrimp-sq4" />
        <div className="ugly-shrimp-square ugly-shrimp-sq5" />
        <div className="ugly-shrimp-square ugly-shrimp-sq6" />
        <div className="ugly-shrimp-square ugly-shrimp-sq7" />
        <div className="ugly-shrimp-square ugly-shrimp-sq8" />
        <div className="ugly-shrimp-square ugly-shrimp-sq9" />
      </div>
    </div>
  );
};
