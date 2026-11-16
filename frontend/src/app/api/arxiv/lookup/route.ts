import type { NextRequest } from 'next/server';
import { backendHeaders, backendUrl } from '@/lib/backendProxy';

export async function GET(req: NextRequest) {
  const arxivId = req.nextUrl.searchParams.get('arxiv_id') || '';
  try {
    const response = await fetch(
      `${backendUrl()}/api/arxiv/lookup?arxiv_id=${encodeURIComponent(arxivId)}`,
      {
        headers: backendHeaders(req),
        signal: AbortSignal.timeout(30000),
      }
    );
    return new Response(await response.text(), {
      status: response.status,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Lookup failed';
    return Response.json({ error: message }, { status: 503 });
  }
}
