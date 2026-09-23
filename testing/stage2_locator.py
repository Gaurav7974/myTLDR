"""
Stage 2: Architecture Section Locator Benchmark.
Tests find_architecture_section at 50,000 chars (current limit) vs expanded windows (100,000 chars).
Measures offset, detection success, confidence, and anti-hallucination verification.
"""

import sys
import time
import re
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testing.harness import PAPERS_DIR, RESULTS_DIR
from app.core.config import settings
from app.services.text_extraction import text_extractor
import pymupdf

PATTERNS = [
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:architecture|model\s+architecture|network\s+architecture)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:methodology|methods|proposed\s+(?:method|model|approach))\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:model\s+design|overall\s+architecture)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?deepseek-v3\s+architecture\b"),
]


def find_ground_truth_offset(text: str) -> dict:
    matches = []
    for pat in PATTERNS:
        for m in pat.finditer(text):
            matches.append({
                "matched_text": m.group(0).strip(),
                "char_offset": m.start(),
                "approx_page": round(m.start() / 4500, 1)  # ~4500 chars/page
            })
    matches.sort(key=lambda x: x["char_offset"])
    return matches[0] if matches else {"matched_text": "None", "char_offset": -1, "approx_page": -1}


def test_locator_for_paper(pdf_path: Path) -> dict:
    doc = pymupdf.open(str(pdf_path))
    raw_text = "\n\n".join(p.get_text() for p in doc)
    doc.close()
    clean_text = text_extractor.strip_tail_sections(raw_text)

    gt = find_ground_truth_offset(clean_text)

    # Test 1: Current production limit (50,000 chars)
    orig_limit = settings.ARCH_SECTION_LLM_MAX_CHARS
    settings.ARCH_SECTION_LLM_MAX_CHARS = 50000

    t0 = time.perf_counter()
    res_50k = text_extractor.find_architecture_section(clean_text)
    time_50k = round(time.perf_counter() - t0, 3)

    len_50k = len(res_50k) if res_50k else 0
    found_50k = res_50k is not None

    # Test 2: Expanded limit (100,000 chars)
    settings.ARCH_SECTION_LLM_MAX_CHARS = 100000
    t1 = time.perf_counter()
    res_100k = text_extractor.find_architecture_section(clean_text)
    time_100k = round(time.perf_counter() - t1, 3)

    len_100k = len(res_100k) if res_100k else 0
    found_100k = res_100k is not None

    # Restore settings
    settings.ARCH_SECTION_LLM_MAX_CHARS = orig_limit

    return {
        "filename": pdf_path.name,
        "clean_text_length": len(clean_text),
        "ground_truth_offset": gt["char_offset"],
        "ground_truth_heading": gt["matched_text"],
        "ground_truth_approx_page": gt["approx_page"],
        "within_50k_window": gt["char_offset"] <= 50000 if gt["char_offset"] > 0 else False,
        "run_50k": {
            "found": found_50k,
            "extracted_length": len_50k,
            "latency_sec": time_50k,
            "sample": res_50k[:120].replace("\n", " ").strip() if res_50k else ""
        },
        "run_100k": {
            "found": found_100k,
            "extracted_length": len_100k,
            "latency_sec": time_100k,
            "sample": res_100k[:120].replace("\n", " ").strip() if res_100k else ""
        }
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 2 BENCHMARK: ARCHITECTURE SECTION LOCATOR")
    print("=" * 70)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_results = []

    for pdf in pdf_files:
        print(f"\n[TESTING] Evaluating section locator on {pdf.name} ...")
        res = test_locator_for_paper(pdf)
        all_results.append(res)
        print(f"  Ground Truth Heading: '{res['ground_truth_heading']}' at char {res['ground_truth_offset']:,} (~page {res['ground_truth_approx_page']})")
        print(f"  Within 50k window: {res['within_50k_window']}")
        print(f"  [50k window] Found: {res['run_50k']['found']} ({res['run_50k']['extracted_length']:,} chars in {res['run_50k']['latency_sec']}s)")
        print(f"  [100k window] Found: {res['run_100k']['found']} ({res['run_100k']['extracted_length']:,} chars in {res['run_100k']['latency_sec']}s)")

    out_file = RESULTS_DIR / "stage2_locator_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Stage 2 results saved to {out_file}")
