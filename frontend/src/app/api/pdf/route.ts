import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const arxivId = searchParams.get('arxivId');
  const customUrl = searchParams.get('url');

  let targetUrl = '';
  if (arxivId) {
    // Extract clean 4-5 digit numeric ID (e.g. 2205.14135 from https://arxiv.org/abs/2205.14135 or 2205.14135v2)
    const match = arxivId.match(/(\d{4}\.\d{4,5})/);
    if (match) {
      targetUrl = `https://arxiv.org/pdf/${match[1]}.pdf`;
    } else if (arxivId.startsWith('http://') || arxivId.startsWith('https://')) {
      targetUrl = arxivId;
    } else {
      targetUrl = 'https://arxiv.org/pdf/1706.03762.pdf';
    }
  } else if (customUrl) {
    const match = customUrl.match(/(\d{4}\.\d{4,5})/);
    if (match) {
      targetUrl = `https://arxiv.org/pdf/${match[1]}.pdf`;
    } else {
      targetUrl = customUrl;
    }
  } else {
    targetUrl = 'https://arxiv.org/pdf/1706.03762.pdf';
  }

  try {
    const response = await fetch(targetUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Failed to fetch PDF: ${response.statusText}` },
        { status: response.status }
      );
    }

    const pdfBuffer = await response.arrayBuffer();

    return new NextResponse(pdfBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'inline; filename="paper.pdf"',
        'Cache-Control': 'public, max-age=86400, s-maxage=86400',
        'X-Frame-Options': 'ALLOWALL',
      },
    });
  } catch (error) {
    console.error('Error proxying arXiv PDF:', error);
    return NextResponse.json(
      { error: 'Internal server error fetching PDF' },
      { status: 500 }
    );
  }
}
