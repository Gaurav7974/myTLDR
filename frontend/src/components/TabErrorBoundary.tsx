'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
  tabName?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class TabErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`[TabErrorBoundary] Error caught in ${this.props.tabName || 'tab'}:`, error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 md:p-8 animate-fadeIn max-w-2xl mx-auto pb-36 font-sans">
          <div className="rounded-lg border border-zinc-200 bg-white p-8 text-center space-y-3 shadow-2xs">
            <div className="w-10 h-10 rounded-md bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center mx-auto">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <h4 className="font-bold text-zinc-950 text-sm font-mono uppercase">
              {this.props.tabName ? `${this.props.tabName} Encountered an Issue` : 'Content Unavailable'}
            </h4>
            <p className="text-xs text-zinc-600 leading-relaxed max-w-md mx-auto">
              An unexpected error occurred while rendering this tab. Other tabs and the PDF viewer remain fully operational.
            </p>
            <div className="pt-2">
              <button
                type="button"
                onClick={this.handleReset}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-900 text-white hover:bg-zinc-800 text-xs font-mono font-medium transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Retry Tab</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
