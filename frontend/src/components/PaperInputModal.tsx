'use client';

import React, { useState } from 'react';
import { X, Link as LinkIcon, FileUp, Sparkles } from 'lucide-react';

interface PaperInputModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSample: () => void;
}

export const PaperInputModal: React.FC<PaperInputModalProps> = ({
  isOpen,
  onClose,
  onSelectSample,
}) => {
  const [url, setUrl] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setIsProcessing(true);
    setTimeout(() => {
      setIsProcessing(false);
      onSelectSample();
      onClose();
    }, 1000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs animate-fadeIn">
      <div className="bg-white rounded-lg border border-zinc-200 shadow-2xl max-w-lg w-full overflow-hidden transition-all text-zinc-900">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-100 bg-[#FAFAFA]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-zinc-900 text-white flex items-center justify-center shadow-xs">
              <Sparkles className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="font-bold text-zinc-950 text-sm">Deconstruct Research Paper</h3>
              <p className="text-[11px] text-zinc-500 font-mono">Extract 3D structure, math & code</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-zinc-700 p-1.5 rounded-md hover:bg-zinc-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-5 bg-[#FFFFFF]">
          
          {/* Link Form */}
          <form onSubmit={handleSubmit} className="space-y-2.5">
            <label className="block text-xs font-semibold text-zinc-800">
              Paste arXiv or PDF URL
            </label>
            <div className="relative">
              <LinkIcon className="absolute left-3 top-3 w-4 h-4 text-zinc-400" />
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://arxiv.org/abs/1706.03762 or https://.../paper.pdf"
                className="w-full pl-9 pr-4 py-2 bg-zinc-50 border border-zinc-200 rounded-md text-xs text-zinc-900 focus:bg-white focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-all placeholder:text-zinc-400"
              />
            </div>
            <button
              type="submit"
              disabled={!url.trim() || isProcessing}
              className="w-full py-2 text-xs font-semibold text-white bg-zinc-900 hover:bg-zinc-800 disabled:opacity-40 rounded-md shadow-xs transition-all active:scale-[0.99]"
            >
              {isProcessing ? 'Deconstructing Paper...' : 'Deconstruct via URL'}
            </button>
          </form>

          {/* Divider */}
          <div className="relative flex items-center justify-center">
            <div className="border-t border-zinc-200 w-full"></div>
            <span className="absolute bg-white px-3 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
              or upload document
            </span>
          </div>

          {/* Drag & Drop Box */}
          <div
            className="border border-dashed border-zinc-300 hover:border-zinc-500 bg-zinc-50/50 hover:bg-zinc-100/50 rounded-lg p-5 text-center transition-all cursor-pointer group"
            onClick={() => {
              onSelectSample();
              onClose();
            }}
          >
            <div className="w-9 h-9 rounded-md bg-white border border-zinc-200 text-zinc-800 flex items-center justify-center mx-auto mb-2 shadow-2xs group-hover:scale-105 transition-transform">
              <FileUp className="w-4 h-4" />
            </div>
            <p className="text-xs font-semibold text-zinc-800">
              Drag & drop your PDF or Markdown file here
            </p>
            <p className="text-[11px] text-zinc-400 font-mono mt-0.5">
              Supports .pdf and .md up to 30 pages (max 25MB)
            </p>
          </div>

          {/* Benchmark Preset Button */}
          <div className="p-3 bg-zinc-50 border border-zinc-200 rounded-md flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-5 h-5 rounded-md bg-zinc-900 text-white flex items-center justify-center text-[10px] font-bold">
                ✓
              </div>
              <div>
                <p className="text-xs font-semibold text-zinc-900">Try Benchmark Preset</p>
                <p className="text-[11px] text-zinc-500">Attention Is All You Need (Vaswani et al.)</p>
              </div>
            </div>
            <button
              onClick={() => {
                onSelectSample();
                onClose();
              }}
              className="px-3 py-1.5 text-xs font-semibold text-zinc-900 bg-white hover:bg-zinc-100 border border-zinc-300 rounded-md transition-all shadow-2xs active:scale-95"
            >
              Load Sample
            </button>
          </div>

        </div>

      </div>
    </div>
  );
};
