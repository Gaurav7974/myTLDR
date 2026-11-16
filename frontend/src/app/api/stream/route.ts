import type { NextRequest } from 'next/server';
import { backendHeaders, backendUrl } from '@/lib/backendProxy';

/** Stream the backend SSE response without buffering it in the function. */
export async function POST(req: NextRequest) {
  try {
    const response = await fetch(`${backendUrl()}/api/stream`, {
      method: 'POST',
      body: req.body,
      headers: {
        ...backendHeaders(req),
        'Content-Type': req.headers.get('content-type') || '',
      },
      duplex: 'half',
      signal: AbortSignal.timeout(600000),
    } as RequestInit & { duplex: 'half' });

    if (!response.ok) {
      return new Response(await response.text(), {
        status: response.status,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return new Response(response.body, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Connection failed';
    return Response.json({ error: message }, { status: 503 });
  }
}
