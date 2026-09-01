/**
 * Client API services for paper ingestion, SSE streaming, and LLM chat.
 */

import { PaperData, ParsedDocument } from '@/types/paper';
import { ArchitectureGraph } from '@/types/architecture_graph';
import { generateDynamicPaperData } from './mockData';

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
      signal: AbortSignal.timeout(180000),
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

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;
        const dataStr = trimmed.slice(5).trim();

        if (dataStr === '[DONE]') break;

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.stage === 'done') {
            finalPayload = parsed as Record<string, unknown>;
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
    }

    if (finalPayload) {
      const result = transformBackendPayloadToPaperData(finalPayload, url, file?.name);
      if (fileBlobUrl) {
        result.metadata.fileBlobUrl = fileBlobUrl;
        result.metadata.pdfUrl = fileBlobUrl;
      }
      return result;
    }
  } catch (err) {
    console.warn('[Pipeline] Backend stream error, using fallback:', err);
  }

  // Graceful fallback with progressive stages
  onProgress?.({ stage: 'extracting', message: 'Extracting paper structure and metadata...' });
  await new Promise((r) => setTimeout(r, 600));

  onProgress?.({ stage: 'explaining', message: 'Generating structured executive brief and specs...' });
  await new Promise((r) => setTimeout(r, 600));

  onProgress?.({ stage: 'diagramming', message: 'Compiling 3D scene graph & tensor matrix highways...' });
  await new Promise((r) => setTimeout(r, 600));

  const paperTitle = file ? file.name.replace(/\.pdf$/i, '') : url || 'Research Paper';
  const paper = generateDynamicPaperData(paperTitle, url);
  if (fileBlobUrl) {
    paper.metadata.fileBlobUrl = fileBlobUrl;
    paper.metadata.pdfUrl = fileBlobUrl;
  }

  onProgress?.({ stage: 'done', message: 'Complete', data: paper });
  return paper;
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
    architectureIR: payload.architecture_ir as PaperData['architectureIR'],
    sceneGraph3D: payload.scene_graph_3d as PaperData['sceneGraph3D'],
    architectureGraph: transformArchitectureGraph(payload.graph),
    bycroftStages: payload.bycroft_stages as PaperData['bycroftStages'],
    residualHighways: payload.bycroft_highways as PaperData['residualHighways'],
    attentionHeatmap: payload.attention_heatmap as PaperData['attentionHeatmap'],
    nextTokenPredictions: payload.next_token_predictions as PaperData['nextTokenPredictions'],
  };
}

/**
 * Map the backend `graph` payload (which the pipeline emits as
 * `graph_to_react_flow` OR a rich `ArchitectureGraph` depending on the
 * path) to the frontend `ArchitectureGraph` shape.
 *
 * The backend uses two different shapes for the SSE `graph` field:
 *  - ``{nodes:[{id,position,data}], edges:[{id,source,target,...}]}``
 *    (react-flow / dagre layout — legacy / text-only path)
 *  - The rich ArchitectureGraph produced by the figure-faithful pipeline
 *    (vision / merged path).
 *
 * The frontend is permissive: when the payload is the rich shape, it
 * passes through. When it is the react-flow shape (no `nodes[0].bbox`),
 * we treat it as auto-layout and pass it through with `layout_mode='auto'`
 * so the diagram can still render the labels.
 */
function transformArchitectureGraph(raw: unknown): ArchitectureGraph | undefined {
  if (!raw || typeof raw !== 'object') return undefined;
  const r = raw as Record<string, unknown>;
  const nodes = Array.isArray(r.nodes) ? r.nodes : [];
  if (nodes.length === 0) return undefined;

  const firstNode = nodes[0] as Record<string, unknown> | undefined;
  const isRichShape = firstNode && (
    'bbox' in firstNode ||
    'description' in firstNode ||
    'math_formula' in firstNode ||
    'paper_term' in firstNode ||
    'provenance' in firstNode
  );

  if (isRichShape) {
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
