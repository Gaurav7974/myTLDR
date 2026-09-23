"""
Stage 1: PDF Ingestion, Text Extraction, Reference Stripping, and Academic Paper Validation.
Tests performance, memory, and character volume across all test papers.
"""

import sys
import time
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testing.harness import PAPERS_DIR, RESULTS_DIR
from app.services.text_extraction import text_extractor
from app.services.paper_validator import paper_validator
import pymupdf


def test_ingestion(pdf_path: Path) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    size_mb = round(len(pdf_bytes) / 1024 / 1024, 2)

    # 1. Raw PyMuPDF parsing
    t0 = time.perf_counter()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    raw_pages = [page.get_text() for page in doc]
    raw_text = "\n\n".join(raw_pages)
    doc.close()
    pymupdf_time = round(time.perf_counter() - t0, 3)

    raw_char_count = len(raw_text)

    # 2. Reference stripping test
    t1 = time.perf_counter()
    clean_text = text_extractor.strip_tail_sections(raw_text)
    strip_time = round(time.perf_counter() - t1, 3)
    clean_char_count = len(clean_text)
    chars_saved = raw_char_count - clean_char_count
    pct_saved = round((chars_saved / raw_char_count * 100), 1) if raw_char_count > 0 else 0

    # 3. Figure extraction test
    t2 = time.perf_counter()
    doc2 = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    extracted_figures = 0
    total_images_in_doc = 0
    for page in doc2:
        imgs = page.get_images()
        total_images_in_doc += len(imgs)
        if extracted_figures < 60:
            extracted_figures += len(imgs)
    extracted_figures = min(extracted_figures, 60)
    doc2.close()
    figure_time = round(time.perf_counter() - t2, 3)

    # 4. Academic paper validator test
    t3 = time.perf_counter()
    is_academic, validator_reason = paper_validator.validate_research_paper(clean_text)
    validator_time = round(time.perf_counter() - t3, 3)

    result = {
        "filename": pdf_path.name,
        "size_mb": size_mb,
        "page_count": page_count,
        "pymupdf_time_sec": pymupdf_time,
        "raw_characters": raw_char_count,
        "clean_characters": clean_char_count,
        "tail_stripped_chars": chars_saved,
        "tail_stripped_pct": pct_saved,
        "total_images_in_doc": total_images_in_doc,
        "capped_figures_extracted": extracted_figures,
        "figure_harvest_time_sec": figure_time,
        "is_academic": is_academic,
        "validator_reason": validator_reason,
        "validator_time_sec": validator_time
    }
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 1 BENCHMARK: INGESTION & EXTRACTION")
    print("=" * 70)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_results = []

    for pdf in pdf_files:
        print(f"\n[TESTING] Processing {pdf.name} ...")
        res = test_ingestion(pdf)
        all_results.append(res)
        print(f"  Pages: {res['page_count']} | Size: {res['size_mb']} MB")
        print(f"  Extraction: {res['pymupdf_time_sec']}s | Raw Chars: {res['raw_characters']:,} -> Clean: {res['clean_characters']:,} (-{res['tail_stripped_pct']}%)")
        print(f"  Figures Found: {res['total_images_in_doc']} (Capped to {res['capped_figures_extracted']}) in {res['figure_harvest_time_sec']}s")
        print(f"  Academic Gate: {'PASS' if res['is_academic'] else 'FAIL'} ({res['validator_time_sec']}s) - {res['validator_reason'][:60]}")

    out_file = RESULTS_DIR / "stage1_ingestion_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Stage 1 results saved to {out_file}")
