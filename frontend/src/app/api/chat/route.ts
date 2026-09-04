import { NextRequest, NextResponse } from 'next/server';

// Simple in-memory rate limiter: 30 requests per minute per IP
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT = 30;
const RATE_WINDOW_MS = 60_000;

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const entry = rateLimitMap.get(ip);

  if (!entry || now > entry.resetAt) {
    rateLimitMap.set(ip, { count: 1, resetAt: now + RATE_WINDOW_MS });
    return false;
  }

  entry.count++;
  return entry.count > RATE_LIMIT;
}

interface ChatRequestBody {
  paper_hash?: string;
  paper_title?: string;
  query: string;
  history?: Array<{ sender: string; text: string }>;
  difficulty?: string;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      paper_hash,
      paper_id,
      paper_title = 'Research Paper',
      query,
      difficulty = 'intermediate'
    } = body;

    const effectiveHash = paper_hash || paper_id || '';
    const effectiveTitle = paper_title || 'Research Paper';

    if (!query || typeof query !== 'string' || !query.trim()) {
      return NextResponse.json(
        { error: 'A query string is required' },
        { status: 400 }
      );
    }

    const backendUrl = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002';

    // Forward real client IP so backend rate limiter works per-user
    const forwarded = req.headers.get('x-forwarded-for');
    const clientIp = forwarded ? forwarded.split(',')[0].trim() : req.headers.get('x-real-ip');

    if (clientIp && isRateLimited(clientIp)) {
      return NextResponse.json(
        { error: 'Rate limit exceeded. Please wait a minute before trying again.' },
        { status: 429 }
      );
    }

    // 1. Forward request to Python FastAPI RAG / LLM endpoint
    try {
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(clientIp ? { 'X-Forwarded-For': clientIp } : {}),
        },
        body: JSON.stringify({
          paper_hash: effectiveHash,
          paper_id: effectiveHash,
          paper_title: effectiveTitle,
          query: query.trim(),
          history: body.history || [],
          difficulty,
        }),
        signal: AbortSignal.timeout(25000), // 25s timeout for deep LLM synthesis
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json({
          answer: data.answer,
          citations: data.citations || [],
        });
      }

      // If backend responded with an error, extract detail
      const errData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errData.detail || `AI backend responded with status ${response.status}` },
        { status: response.status }
      );
    } catch (fetchErr: unknown) {
      const errMsg = fetchErr instanceof Error ? fetchErr.message : 'Connection failed';
      return NextResponse.json(
        {
          error: `Unable to connect to AI Assistant backend at ${backendUrl} (${errMsg}). Please verify that the backend server is running.`,
        },
        { status: 503 }
      );
    }

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to process chat message';
    return NextResponse.json(
      { error: message },
      { status: 500 }
    );
  }
}
