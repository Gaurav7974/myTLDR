import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { NextRequest } from 'next/server';

function fakeRequest(body: unknown, headers: Record<string, string> = {}): NextRequest {
  return {
    json: async () => body,
    body: new ReadableStream(),
    nextUrl: new URL('http://localhost/api/arxiv/lookup?arxiv_id=1706.03762'),
    headers: new Headers(headers),
  } as unknown as NextRequest;
}

function fetchCall() {
  const fetchMock = vi.mocked(globalThis.fetch);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  return fetchMock.mock.calls[0];
}

describe('server-side backend proxies', () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.BACKEND_API_KEY;

  beforeEach(() => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('data: {"stage":"done"}\n\n', {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    ) as typeof fetch;
    process.env.BACKEND_API_KEY = 'server-secret';
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.BACKEND_API_KEY;
    else process.env.BACKEND_API_KEY = originalKey;
    vi.restoreAllMocks();
  });

  it('chat proxy attaches the server-only key and forwards the client IP', async () => {
    const { POST } = await import('../app/api/chat/route');

    await POST(fakeRequest({ query: 'hello' }, { 'x-forwarded-for': '1.2.3.4' }));

    const [url, init] = fetchCall();
    expect(String(url)).toContain('/api/chat');
    expect((init?.headers as Record<string, string>)['X-API-Key']).toBe('server-secret');
    expect((init?.headers as Record<string, string>)['X-Forwarded-For']).toBe('1.2.3.4');
  });

  it('arxiv proxy attaches the key and preserves the query string', async () => {
    const { GET } = await import('../app/api/arxiv/lookup/route');

    await GET(fakeRequest(null, { 'x-forwarded-for': '5.6.7.8' }));

    const [url, init] = fetchCall();
    expect(String(url)).toContain('/api/arxiv/lookup?arxiv_id=1706.03762');
    expect((init?.headers as Record<string, string>)['X-API-Key']).toBe('server-secret');
    expect((init?.headers as Record<string, string>)['X-Forwarded-For']).toBe('5.6.7.8');
  });

  it('stream proxy forwards the body without buffering and keeps SSE open', async () => {
    const { POST } = await import('../app/api/stream/route');

    const response = await POST(fakeRequest(null, {
      'x-forwarded-for': '9.10.11.12',
      'content-type': 'multipart/form-data; boundary=abc',
    }));

    const [url, init] = fetchCall();
    expect(String(url)).toContain('/api/stream');
    expect((init?.headers as Record<string, string>)['X-API-Key']).toBe('server-secret');
    expect((init?.headers as Record<string, string>)['Content-Type']).toBe(
      'multipart/form-data; boundary=abc'
    );
    expect((init as RequestInit & { duplex?: string }).duplex).toBe('half');
    expect(response.headers.get('Content-Type')).toContain('text/event-stream');
    expect(await response.text()).toContain('stage');
  });
});
