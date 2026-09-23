"""
Downloader utility for test papers.
Downloads PDFs from arXiv with rate limiting and caches them in testing/papers/.
"""

import time
import requests
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf
from testing.harness import PAPERS_DIR

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

TEST_PAPERS = [
    {"id": "1706.03762", "tier": "Baseline", "expected_pages": 15, "name": "Attention Is All You Need"},
    {"id": "2511.21631", "tier": "Tier 1 (~40p)", "expected_pages": 42, "name": "Qwen3-VL Technical Report"},
    {"id": "2307.06435", "tier": "Tier 2 (~47p)", "expected_pages": 47, "name": "A Comprehensive Overview of Large Language Models"},
    {"id": "2412.19437", "tier": "Tier 3 (~53p)", "expected_pages": 53, "name": "DeepSeek-V3 Technical Report"},
    {"id": "2309.16609", "tier": "Tier 4 (~59p)", "expected_pages": 59, "name": "Qwen Technical Report"},
]


def download_paper(arxiv_id: str) -> Path:
    dest = PAPERS_DIR / f"{arxiv_id}.pdf"
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"[CACHE] {arxiv_id} already exists ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        return dest

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"[DOWNLOAD] Fetching {arxiv_id} from {url} ...")
    headers = {"User-Agent": USER_AGENT}

    # Respect arXiv rate limit
    time.sleep(3.5)

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    print(f"[DOWNLOAD] Saved to {dest} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
    return dest


def inspect_paper(pdf_path: Path) -> dict:
    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    first_page_text = doc[0].get_text()[:500] if page_count > 0 else ""
    doc.close()
    return {
        "path": str(pdf_path),
        "size_mb": round(pdf_path.stat().st_size / 1024 / 1024, 2),
        "page_count": page_count,
        "sample": first_page_text.replace("\n", " ").strip()[:100]
    }


if __name__ == "__main__":
    print("[DOWNLOADER] Starting download and inspection of test suite...")
    for paper in TEST_PAPERS:
        try:
            path = download_paper(paper["id"])
            info = inspect_paper(path)
            print(f" -> {paper['id']} ({paper['name']}): {info['page_count']} pages, {info['size_mb']} MB")
        except Exception as e:
            print(f" -> ERROR downloading {paper['id']}: {e}")
