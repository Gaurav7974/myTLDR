import { describe, it, expect } from 'vitest';
import { findCitationPage, ChatCitation } from './AIChatTab';
import { PaperData } from '@/types/paper';

function paperWithPages(pages: { pageNumber: number; markdown: string }[]): PaperData {
  return {
    metadata: { id: 't', title: 'T', authors: [], publishedDate: '', tags: [], citationsCount: 0 },
    eli5Summary: { oneSentenceSummary: '', analogy: '', coreProblem: '', keyInnovations: [], prerequisites: [] },
    markdownContent: '',
    architectureNodes: [],
    equations: [],
    codeSnippets: [],
    parsedDocument: { pageCount: pages.length, pages: pages.map((p) => ({ ...p, source: 'pymupdf' as const })) },
  };
}

const cit = (content: string): ChatCitation => ({ section_title: 'S', content, similarity: 0.9 });

describe('findCitationPage', () => {
  it('returns the page holding the citation topic', () => {
    const paper = paperWithPages([
      { pageNumber: 1, markdown: 'Introduction to language modeling' },
      { pageNumber: 2, markdown: 'Multi-head self-attention computes scaled dot-product attention over queries' },
    ]);
    expect(findCitationPage(cit('Scaled dot-product attention over queries explained'), paper)).toBe(2);
  });

  it('returns null when fewer than 2 distinctive words match', () => {
    const paper = paperWithPages([{ pageNumber: 1, markdown: 'Introduction to language modeling' }]);
    expect(findCitationPage(cit('Quantum chromodynamics lattice gauge ensembles'), paper)).toBe(null);
  });

  it('returns null without a parsed document', () => {
    const paper = paperWithPages([]);
    delete (paper as Partial<PaperData>).parsedDocument;
    expect(findCitationPage(cit('Scaled dot-product attention over queries'), paper)).toBe(null);
  });
});
