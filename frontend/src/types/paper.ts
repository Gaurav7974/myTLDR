import { ArchitectureIR, SceneGraph3D } from './architecture_ir';
import { ArchitectureGraph } from './architecture_graph';

export interface SubComponent {
  id: string;
  name: string;
  role: string;
  description: string;
  tensorShape?: string;
  formula?: string;
  codeSnippet?: string;
}

export interface ArchitectureNode3D {
  id: string;
  name: string;
  category: 'encoder' | 'decoder' | 'attention' | 'norm' | 'embedding' | 'output';
  description: string;
  position: [number, number, number]; // [x, y, z] for 3D layout
  color: string;
  tensorShapeInput?: string;
  tensorShapeOutput?: string;
  subComponents?: SubComponent[];
  latexFormula?: string;
}

export interface EquationBreakdown {
  id: string;
  title: string;
  latex: string;
  explanation: string;
  variables: {
    symbol: string;
    name: string;
    meaning: string;
    dimension?: string;
  }[];
}

export interface KeywordDefinition {
  id: string;
  term: string;
  category: 'core' | 'architecture' | 'mechanism' | 'math' | 'optimization';
  definition: string;
  simpleExplanation: string;
  formula?: string;
  relatedSection?: string;
}

export interface PaperMetadata {
  id: string;
  title: string;
  authors: string[];
  publishedDate: string;
  arxivId?: string;
  tags: string[];
  keywords?: KeywordDefinition[];
  pdfUrl?: string;
  fileBlobUrl?: string;
  citationsCount: number;
  contentHash?: string;
}

export interface BycroftStage {
  id: string;
  name: string;
  stepNumber: number;
  category: 'embedding' | 'projections' | 'attention_matrix' | 'multihead_norm' | 'mlp_ffn' | 'residual' | 'convolution' | 'ssm_state' | 'diffusion_denoise' | 'logits' | 'output';
  description: string;
  mathFormula: string;
  tensorInput: string;
  tensorOutput: string;
  parameters: string;
  flops?: string;
  intuition: string;
  detailedWalkthrough?: string;
  concreteExample?: string;
  position3D: [number, number, number];
  size3D?: [number, number, number];
  gridDimensions?: { rows: number; cols: number };
  subLabels?: string[];
  color: string;
}

export interface AttentionHeatmapData {
  tokens: string[];
  matrix: number[][]; // [queryIdx][keyIdx]
}

export interface NextTokenCandidate {
  token: string;
  probability: number;
  logit: number;
}

export interface ProvenanceSource {
  page?: number;
  section?: string;
  figure?: string;
}

export interface SummaryOverview {
  one_sentence: string;
  problem: string;
  solution: string;
  why_it_matters: string;
}

export interface KeySpecification {
  label: string;
  value: string;
  source?: ProvenanceSource;
}

export interface PaperContribution {
  title: string;
  summary: string;
  category: 'novel' | 'architectural' | 'methodological' | 'experimental' | 'supporting';
  importance?: string;
  source?: ProvenanceSource;
}

export interface KeyFinding {
  title: string;
  summary: string;
  metric_value?: string;
  source?: ProvenanceSource;
}

export interface PaperLimitation {
  title: string;
  summary: string;
  details?: string;
  source?: ProvenanceSource;
}

export interface StructuredPaperSummary {
  overview: SummaryOverview;
  key_specs: KeySpecification[];
  contributions: PaperContribution[];
  key_findings: KeyFinding[];
  limitations: PaperLimitation[];
}

export interface StepMechanism {
  step: number;
  title: string;
  description: string;
  whyItMatters: string;
}

export interface TradeoffItem {
  advantage: string;
  limitation: string;
  details?: string;
}

export interface IntuitionFAQ {
  id: string;
  question: string;
  shortAnswer: string;
  deepExplanation: string;
  conceptTag?: string;
}

export interface ParadigmShift {
  theOldWay: string;
  theNewWay: string;
  whyItMatters: string;
}

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export interface ParsedPage {
  pageNumber: number;
  markdown: string;
  source: 'pymupdf' | 'mistral_ocr';
}

export interface ParsedFigure {
  id: string;
  page: number;
  caption?: string | null;
}

export interface ParsedDocument {
  pageCount: number;
  pages: ParsedPage[];
  figures?: ParsedFigure[];
}

export interface PaperData {
  metadata: PaperMetadata;
  structuredSummary?: StructuredPaperSummary;
  eli5Summary: {
    oneSentenceSummary: string;
    analogy: string;
    coreProblem: string;
    paradigmShift?: ParadigmShift;
    keyInnovations: string[];
    prerequisites: string[];
    stepByStepMechanism?: StepMechanism[];
    tradeoffsAndLimitations?: TradeoffItem[];
    commonQuestions?: IntuitionFAQ[];
    knowledgeCheck?: QuizQuestion[];
  };
  markdownContent: string;
  parsedDocument?: ParsedDocument;
  architectureNodes: ArchitectureNode3D[];
  equations: EquationBreakdown[];
  codeSnippets: {
    title: string;
    language: string;
    code: string;
  }[];
  /** Honest refusal reason when the paper has no implementable architecture
   *  (backend needs_code gate). Shown instead of invented code. */
  codeNote?: string | null;
  architectureIR?: ArchitectureIR;
  sceneGraph3D?: SceneGraph3D;
  /** Figure-faithful graph payload from the backend (vision/merged/text/legacy).
   *  When present, the visualizer uses it for the 2D diagram; when absent,
   *  the visualizer shows an honest unavailable state (never a static mock). */
  architectureGraph?: ArchitectureGraph;
  bycroftStages?: BycroftStage[];
  residualHighways?: { id: string; startStep: number; endStep: number; label: string; color?: string }[];
  attentionHeatmap?: AttentionHeatmapData;
  nextTokenPredictions?: NextTokenCandidate[];
}
