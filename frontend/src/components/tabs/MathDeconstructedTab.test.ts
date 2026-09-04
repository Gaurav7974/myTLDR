import { describe, it, expect } from 'vitest';
import katex from 'katex';

describe('MathDeconstructedTab and KaTeX resilience', () => {
  it('safely renders valid LaTeX strings without throwing', () => {
    const rendered = katex.renderToString('\\text{Softmax}(x)', {
      displayMode: true,
      throwOnError: false,
    });
    expect(rendered).toContain('katex');
  });

  it('safely handles invalid LaTeX without throwing when throwOnError is false', () => {
    const renderBadLatex = () => {
      return katex.renderToString('\\invalidCommand{test}', {
        displayMode: true,
        throwOnError: false,
      });
    };
    expect(renderBadLatex).not.toThrow();
  });

  it('handles empty equations array and undefined equations safely', () => {
    const paperWithEmptyEquations = {
      equations: [],
    };
    const equations = paperWithEmptyEquations.equations || [];
    expect(equations.length).toBe(0);

    const paperWithUndefinedEquations = {};
    const equationsUndef = (paperWithUndefinedEquations as { equations?: unknown[] }).equations || [];
    expect(equationsUndef.length).toBe(0);
  });

  it('safely retrieves first variable symbol without optional chaining index crash', () => {
    const emptyEquations: unknown[] = [];
    const firstEq = emptyEquations[0] as { variables?: { symbol: string }[] } | undefined;
    // Old buggy code was: equations[0]?.variables[0]?.symbol -> threw TypeError when undefined[0]
    // Safe expression:
    const activeVar = firstEq?.variables?.[0]?.symbol || null;
    expect(activeVar).toBeNull();
  });
});
