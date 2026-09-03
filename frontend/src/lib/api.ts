/**
 * Client API services for paper ingestion, SSE streaming, and LLM chat.
 */

import { PaperData, ParsedDocument } from '@/types/paper';
import { ArchitectureGraph } from '@/types/architecture_graph';

export interface StreamProgressEvent {
  stage: 'extracting' | 'explaining' | 'diagramming' | 'done' | 'error';
  message?: string;
  data?: unknown;
}

export async function processPaperStream(
  params: {
    source: 'upload' | 'arxiv' | 'url';
    url?: string;
    file?: File;
    difficulty?: 'beginner' | 'intermediate' | 'expert';
  },
  onProgress?: (event: StreamProgressEvent) => void
): Promise<PaperData> {
  const { source, url, file, difficulty = 'intermediate' } = params;
  const fileBlobUrl = file && typeof URL !== 'undefined' ? URL.createObjectURL(file) : undefined;

  try {
    const formData = new FormData();
    formData.append('source', source);
    formData.append('difficulty_level', difficulty);
    if (url) formData.append('url', url);
    if (file) formData.append('file', file);

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

    const response = await fetch(`${backendUrl}/api/stream`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(600000),
    });

    if (!response.ok) {
      throw new Error(`Backend streaming returned ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Streaming response body is unavailable');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalPayload: Record<string, unknown> | null = null;
    let backendError: string | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let isDone = false;
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        const dataStr = trimmed.slice(5).trim();

        if (dataStr === '[DONE]') {
          isDone = true;
          break;
        }

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.stage === 'done') {
            finalPayload = parsed as Record<string, unknown>;
          } else if (parsed.stage === 'error') {
            backendError = parsed.message || 'Processing failed. Please retry.';
          }
          onProgress?.({
            stage: parsed.stage,
            message: parsed.message || (parsed.stage === 'done' ? 'Completed!' : undefined),
            data: parsed,
          });
        } catch {
          // Ignore partial chunk parse errors
        }
      }
      if (isDone) {
        reader.cancel().catch(() => {});
        break;
      }
    }

    if (backendError) {
      throw new Error(backendError);
    }
    if (finalPayload) {
      const result = transformBackendPayloadToPaperData(finalPayload, url, file?.name);
      if (fileBlobUrl) { result.metadata.fileBlobUrl = fileBlobUrl; result.metadata.pdfUrl = fileBlobUrl; }
      return result;
    }
    // Stream ended without a done event (backend crashed mid-pipeline) —
    // surface a real error instead of silently substituting mock data.
    throw new Error('The processing stream ended before results were ready. Please retry.');
  } catch (err) {
    console.warn('[Pipeline] Backend stream error:', err);
    throw err instanceof Error ? err : new Error('Could not reach the processing backend.');
  }
}

function transformBackendPayloadToPaperData(
  payload: Record<string, unknown>,
  url?: string,
  filename?: string
): PaperData {
  const explanation = (payload.explanation as Record<string, unknown>) || {};
  const metadata = (payload.metadata as Record<string, unknown>) || {};

  const structuredSummary = (explanation.overview ? explanation : (explanation.structuredSummary || explanation)) as PaperData['structuredSummary'];
  const overview = (explanation.overview as Record<string, string>) || {};

  const rawTitle = (metadata.title as string) || filename?.replace(/\.pdf$/i, '') || url?.replace(/.*\/abs\//, 'arXiv:') || 'Uploaded Research Paper';

    const parsedDocument = transformParsedText(payload.parsed_text);

  // Prefer backend pdf_url; otherwise construct direct arXiv PDF URL from arxivId/url
  const arxivMatch = url?.match(/\d{4}\.\d{4,5}/)?.[0];
  const directPdfUrl =
    (metadata.pdf_url as string) ||
    (metadata.arxiv_id ? `https://arxiv.org/pdf/${metadata.arxiv_id}.pdf` : undefined) ||
    (url && url.includes('arxiv.org/pdf') ? url : undefined) ||
    (arxivMatch ? `https://arxiv.org/pdf/${arxivMatch}.pdf` : undefined) ||
    url;

  return {
    metadata: {
      id: (metadata.arxiv_id as string) || 'paper-' + Date.now(),
      title: rawTitle,
      authors: (metadata.authors as string[]) || ['Research Author et al.'],
      publishedDate: (metadata.published as string) || 'Recent',
      arxivId: (metadata.arxiv_id as string) || url?.replace(/.*\/abs\//, '').replace(/\.pdf.*$/, '') || url?.match(/\d{4}\.\d{4,5}/)?.[0],
      tags: (metadata.categories as string[]) || (explanation.tags as string[]) || ['Machine Learning', 'Deep Learning', 'Neural Architecture'],
      keywords: (explanation.metadata as Record<string, unknown>)?.keywords as PaperData['metadata']['keywords'] || [],
      pdfUrl: directPdfUrl,
      citationsCount: (metadata.citations as number) || 1200,
    },
    structuredSummary,
    eli5Summary: {
      oneSentenceSummary: overview.one_sentence || (explanation.summary as string) || '',
      analogy: (explanation.analogy as string) || 'A modular computation engine that processes representations concurrently.',
      coreProblem: overview.problem || '',
      paradigmShift: explanation.paradigmShift as PaperData['eli5Summary']['paradigmShift'],
      keyInnovations: (explanation.key_innovations as string[]) || [],
      prerequisites: (explanation.prerequisites as string[]) || ['Linear Algebra', 'PyTorch Basics'],
      stepByStepMechanism: explanation.step_by_step_mechanism as PaperData['eli5Summary']['stepByStepMechanism'],
      tradeoffsAndLimitations: explanation.tradeoffs as PaperData['eli5Summary']['tradeoffsAndLimitations'],
      commonQuestions: explanation.commonQuestions as PaperData['eli5Summary']['commonQuestions'],
      knowledgeCheck: explanation.knowledgeCheck as PaperData['eli5Summary']['knowledgeCheck'],
    },
    markdownContent: (explanation.markdown as string) || '',
    parsedDocument,
    architectureNodes: (payload.architecture_nodes as PaperData['architectureNodes']) || [],
    equations: (payload.equations as PaperData['equations']) || [],
    codeSnippets: (payload.code_snippets as PaperData['codeSnippets']) || [],
    codeNote: (payload.code_note as string) || null,
    architectureIR: payload.architecture_ir as PaperData['architectureIR'],
    sceneGraph3D: payload.scene_graph_3d as PaperData['sceneGraph3D'],
    architectureGraph: transformArchitectureGraph(payload.graph, payload.architecture_ir),
    bycroftStages: payload.bycroft_stages as PaperData['bycroftStages'],
    residualHighways: payload.bycroft_highways as PaperData['residualHighways'],
    attentionHeatmap: payload.attention_heatmap as PaperData['attentionHeatmap'],
    nextTokenPredictions: payload.next_token_predictions as PaperData['nextTokenPredictions'],
  };
}

/**
 * Convert canonical ArchitectureIR into an ArchitectureGraph for 2D rendering.
 * Used when a paper has no extracted figure bboxes or visual graph, but has
 * structured semantic architecture nodes and data flows.
 */
export function irToArchitectureGraph(rawIr: unknown): ArchitectureGraph | undefined {
  if (!rawIr || typeof rawIr !== 'object') return undefined;
  const ir = rawIr as Record<string, unknown>;
  const irNodes = Array.isArray(ir.nodes) ? ir.nodes : [];
  if (irNodes.length === 0) return undefined;

  const outNodes = irNodes.map((n) => {
    const node = n as Record<string, unknown>;
    return {
      id: String(node.id),
      label: String(node.label ?? node.id),
      type: String(node.type ?? 'other'),
      description: typeof node.description === 'string' ? node.description : undefined,
      math_formula: typeof node.math_formula === 'string' ? node.math_formula : undefined,
      parameters: typeof node.parameters === 'string' ? node.parameters : undefined,
      flops: typeof node.flops === 'string' ? node.flops : undefined,
      paper_term: typeof node.paper_term === 'string' ? node.paper_term : undefined,
    };
  });

  const irEdges = Array.isArray(ir.edges) ? ir.edges : [];
  const outEdges = irEdges.map((e) => {
    const edge = e as Record<string, unknown>;
    const edgeType = String(edge.type ?? 'data_flow');
    return {
      source: String(edge.source),
      target: String(edge.target),
      type: edgeType as any,
      label: typeof edge.label === 'string' ? edge.label : null,
      style: edgeType === 'residual_skip' ? { name: 'residual' as const } : undefined,
    };
  });

  const archMeta = (ir.architecture as Record<string, unknown>) || {};
  return {
    nodes: outNodes,
    edges: outEdges,
    version: String(ir.version || '1.0'),
    source: 'auto',
    layout_mode: 'auto',
    paper_title: typeof archMeta.name === 'string' ? archMeta.name : undefined,
  };
}

/**
 * Map the backend `graph` payload (rich ArchitectureGraph, or the legacy
 * react-flow/dagre shape on older deployments) to the frontend shape.
 * Falls back to synthesizing from ArchitectureIR if the graph payload is empty.
 */
export function transformArchitectureGraph(raw: unknown, rawIr?: unknown): ArchitectureGraph | undefined {
  if (raw && typeof raw === 'object') {
    const r = raw as Record<string, unknown>;
    const nodes = Array.isArray(r.nodes) ? r.nodes : [];
    if (nodes.length > 0) {
      const firstNode = nodes[0] as Record<string, unknown> | undefined;
      const isReactFlowShape = !!firstNode && 'position' in firstNode;

      if (!isReactFlowShape) {
        return r as unknown as ArchitectureGraph;
      }

      // Legacy react-flow shape: convert to ArchitectureGraph nodes/edges
      // with labels from the react-flow `data.label` field. layout_mode=auto.
      const outNodes = nodes.map((n) => {
        const node = n as Record<string, unknown>;
        const data = (node.data as Record<string, unknown>) || {};
        return {
          id: String(node.id),
          label: String(data.label ?? node.id),
          type: String(data.node_type ?? 'other'),
        };
      });
      const outEdges = Array.isArray(r.edges)
        ? (r.edges as Array<Record<string, unknown>>).map((e) => ({
            source: String(e.source),
            target: String(e.target),
            label: (e.label as string) ?? null,
          }))
        : [];
      return {
        nodes: outNodes,
        edges: outEdges,
        version: 'legacy',
        source: 'legacy',
        layout_mode: 'auto',
      };
    }
  }

  // Fallback to synthesizing ArchitectureGraph from ArchitectureIR if present
  if (rawIr) {
    return irToArchitectureGraph(rawIr);
  }

  return undefined;
}

function transformParsedText(raw: unknown): ParsedDocument | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const parsed = raw as Record<string, unknown>;
  const rawPages = Array.isArray(parsed.pages) ? parsed.pages : [];
  const pages = rawPages
    .filter((p): p is Record<string, unknown> => !!p && typeof p === 'object')
    .map((p) => ({
      pageNumber: Number(p.page_number) || 0,
      markdown: typeof p.markdown === 'string' ? p.markdown : '',
      source: p.source === 'mistral_ocr' ? ('mistral_ocr' as const) : ('pymupdf' as const),
    }))
    .filter((p) => p.markdown.trim().length > 0);

  if (pages.length === 0) return undefined;

  return {
    pageCount: Number(parsed.page_count) || pages.length,
    pages,
    figures: Array.isArray(parsed.figures)
      ? (parsed.figures as ParsedDocument['figures'])
      : undefined,
  };
}
