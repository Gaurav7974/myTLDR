"""
Stage 3: Explanation and Pedagogical Summary Benchmark.
Tests explanation_service.generate across all test papers.
Measures latency, schema completeness, and citation provenance.
"""

import sys
import time
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testing.harness import PAPERS_DIR, RESULTS_DIR
from app.services.text_extraction import text_extractor
from app.services.explanation import explanation_service
import pymupdf


def test_explanation_for_paper(pdf_path: Path) -> dict:
    doc = pymupdf.open(str(pdf_path))
    raw_text = "\n\n".join(p.get_text() for p in doc)
    page_count = len(doc)
    doc.close()
    clean_text = text_extractor.strip_tail_sections(raw_text)

    t0 = time.perf_counter()
    try:
        explanation = explanation_service.generate(clean_text, difficulty="intermediate")
        latency = round(time.perf_counter() - t0, 3)
        success = True
        error_msg = None
    except Exception as e:
        latency = round(time.perf_counter() - t0, 3)
        success = False
        explanation = None
        error_msg = str(e)

    # Inspect schema completeness if successful
    overview = explanation.get("overview") if explanation else {}
    key_specs = explanation.get("key_specs") if explanation else []
    contributions = explanation.get("contributions") if explanation else []
    key_findings = explanation.get("key_findings") if explanation else []
    limitations = explanation.get("limitations") if explanation else []

    # Check provenance citations: what pages are cited?
    cited_pages = set()
    for item in (key_specs or []) + (contributions or []) + (key_findings or []) + (limitations or []):
        if isinstance(item, dict) and "source" in item and isinstance(item["source"], dict):
            p = item["source"].get("page")
            if p:
                cited_pages.add(p)

    return {
        "filename": pdf_path.name,
        "page_count": page_count,
        "clean_text_length": len(clean_text),
        "success": success,
        "latency_sec": latency,
        "error": error_msg,
        "has_overview": bool(overview.get("one_sentence")),
        "key_specs_count": len(key_specs) if isinstance(key_specs, list) else 0,
        "contributions_count": len(contributions) if isinstance(contributions, list) else 0,
        "key_findings_count": len(key_findings) if isinstance(key_findings, list) else 0,
        "limitations_count": len(limitations) if isinstance(limitations, list) else 0,
        "cited_pages": sorted(list(cited_pages)),
        "max_cited_page": max(cited_pages) if cited_pages else None,
        "sample_overview": overview.get("one_sentence", "") if overview else ""
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 3 BENCHMARK: EXPLANATION GENERATION")
    print("=" * 70)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_results = []

    for pdf in pdf_files:
        print(f"\n[TESTING] Generating explanation for {pdf.name} ...")
        res = test_explanation_for_paper(pdf)
        all_results.append(res)
        print(f"  Success: {res['success']} ({res['latency_sec']}s)")
        if res["success"]:
            print(f"  Overview: {res['sample_overview'][:100]}...")
            print(f"  Counts: Specs={res['key_specs_count']}, Contribs={res['contributions_count']}, Findings={res['key_findings_count']}, Limits={res['limitations_count']}")
            print(f"  Cited Pages: {res['cited_pages']} (Max cited page: {res['max_cited_page']} of {res['page_count']})")
        else:
            print(f"  Error: {res['error']}")

    out_file = RESULTS_DIR / "stage3_explanation_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Stage 3 results saved to {out_file}")
