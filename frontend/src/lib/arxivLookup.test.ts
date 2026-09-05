/**
 * Unit tests for lookupArxiv (issue: typo'd ID silently decodes wrong paper).
 * fetch is stubbed — no network.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { lookupArxiv } from './api';

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockFetch(ok: boolean, body: unknown, status = 200) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      status,
      json: async () => body,
    }))
  );
}

describe('lookupArxiv', () => {
  it('returns canonical metadata on success', async () => {
    mockFetch(true, {
      arxiv_id: '1512.03385',
      title: 'Deep Residual Learning',
      authors: ['Kaiming He'],
      abstract: '...',
      categories: ['cs.CV'],
    });
    const meta = await lookupArxiv('1512.03385');
    expect(meta.arxiv_id).toBe('1512.03385');
    expect(meta.title).toBe('Deep Residual Learning');
    const calledUrl = String((fetch as ReturnType<typeof vi.fn>).mock.calls[0][0]);
    expect(calledUrl).toContain('/api/arxiv/lookup?arxiv_id=1512.03385');
  });

  it('throws the backend detail on unknown IDs', async () => {
    mockFetch(false, { detail: "arXiv ID '9999.99999' does not match any known paper." }, 400);
    await expect(lookupArxiv('9999.99999')).rejects.toThrow('does not match any known paper');
  });

  it('falls back to status text on non-JSON errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('no json');
        },
      }))
    );
    await expect(lookupArxiv('1512.03385')).rejects.toThrow('arXiv lookup failed (500)');
  });
});
