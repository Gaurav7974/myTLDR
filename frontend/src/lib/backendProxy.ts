import type { NextRequest } from 'next/server';

/** Server-only helpers for the three routes that talk to the paid backend. */
export function backendUrl(): string {
  return process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
}

export function backendHeaders(req: NextRequest): Record<string, string> {
  const headers: Record<string, string> = {};
  const key = process.env.BACKEND_API_KEY;
  if (key) headers['X-API-Key'] = key;
  const ip =
    req.headers.get('x-forwarded-for')?.split(',')[0].trim() ||
    req.headers.get('x-real-ip');
  if (ip) headers['X-Forwarded-For'] = ip;
  return headers;
}
