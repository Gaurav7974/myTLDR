"""
End-to-End Stress Test Runner.
Executes the full production `process_paper` pipeline on papers ranging from 15 to 59 pages.
Tests pipeline robustness, latency, diagram generation, code generation, and pedagogical summaries.
"""

import sys
import asyncio
import time
from pathlib import Path
import json

# Setup paths
TESTING_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTING_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

from testing.harness import PAPERS_DIR, RESULTS_DIR
from app.core.config import settings
from app.services.pipeline import process_paper
import pymupdf


async def run_pipeline_for_paper(pdf_path: Path, max_pages_setting: int = 65, arch_chars_setting: int = 50000) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    doc.close()

    # Tweak settings dynamically in-memory for this test run
    orig_max_pages = settings.MAX_PDF_PAGES
    orig_arch_chars = settings.ARCH_SECTION_LLM_MAX_CHARS

    settings.MAX_PDF_PAGES = max_pages_setting
    settings.ARCH_SECTION_LLM_MAX_CHARS = arch_chars_setting

    print(f"\n" + "=" * 70)
    print(f"RUNNING E2E PIPELINE: {pdf_path.name}")
    print(f"Pages: {page_count} | Size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
    print(f"Settings: MAX_PDF_PAGES={settings.MAX_PDF_PAGES}, ARCH_SECTION_LLM_MAX_CHARS={settings.ARCH_SECTION_LLM_MAX_CHARS}")
    print("=" * 70)

    stages_seen = []
    final_event = None
    t0 = time.perf_counter()

    try:
        async for event in process_paper(
            source_type="upload",
            file_bytes=pdf_bytes,
            url=None,
            difficulty="intermediate",
            db=None
        ):
            stage = event.get("stage")
            msg = event.get("message", "")
            stages_seen.append(stage)
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"[{elapsed}s] -> Stage: {stage} | {msg}")

            if stage == "done":
                final_event = event
            elif stage == "error":
                final_event = event
                break

    except Exception as e:
        print(f"[ERROR] Pipeline raised exception: {e}")
        final_event = {"stage": "exception", "error": str(e)}

    finally:
        total_time = round(time.perf_counter() - t0, 2)
        settings.MAX_PDF_PAGES = orig_max_pages
        settings.ARCH_SECTION_LLM_MAX_CHARS = orig_arch_chars

    # Parse and analyze results
    if final_event and final_event.get("stage") == "done":
        explanation = final_event.get("explanation") or {}
        graph = final_event.get("graph") or {}
        arch_ir = final_event.get("architecture_ir") or {}
        equations = final_event.get("equations") or []
        code_snippets = final_event.get("code_snippets") or []
        diagram_available = final_event.get("diagram_available", False)

        graph_nodes = len(graph.get("nodes", [])) if graph else 0
        graph_edges = len(graph.get("edges", [])) if graph else 0
        ir_nodes = len(arch_ir.get("nodes", [])) if arch_ir else 0
        paradigm = arch_ir.get("architecture", {}).get("paradigm", "unknown") if arch_ir else "unknown"

        summary_available = final_event.get("summary_available", False)
        overview_text = explanation.get("overview", {}).get("one_sentence", "")

        report_item = {
            "filename": pdf_path.name,
            "page_count": page_count,
            "size_mb": round(len(pdf_bytes) / 1024 / 1024, 2),
            "status": "SUCCESS",
            "total_time_sec": total_time,
            "stages_traversed": stages_seen,
            "diagram_available": diagram_available,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_source": graph.get("source") if graph else None,
            "ir_nodes": ir_nodes,
            "ir_paradigm": paradigm,
            "equations_count": len(equations),
            "code_snippets_count": len(code_snippets),
            "summary_available": summary_available,
            "overview_sample": overview_text[:120]
        }
    else:
        report_item = {
            "filename": pdf_path.name,
            "page_count": page_count,
            "size_mb": round(len(pdf_bytes) / 1024 / 1024, 2),
            "status": "FAILED",
            "total_time_sec": total_time,
            "stages_traversed": stages_seen,
            "error": final_event.get("message") or final_event.get("error") if final_event else "Unknown failure"
        }

    print("\nRESULT SUMMARY:")
    print(json.dumps(report_item, indent=2))
    return report_item


async def main():
    test_files = [
        PAPERS_DIR / "1706.03762.pdf",  # 15 pages
        PAPERS_DIR / "2511.21631.pdf",  # 42 pages
        PAPERS_DIR / "2307.06435.pdf",  # 47 pages
        PAPERS_DIR / "2412.19437.pdf",  # 53 pages
        PAPERS_DIR / "2309.16609.pdf",  # 59 pages
    ]

    all_results = []
    for pdf in test_files:
        if pdf.exists():
            res = await run_pipeline_for_paper(pdf, max_pages_setting=65, arch_chars_setting=60000)
            all_results.append(res)
            # Brief pause between papers to allow rate-limits to cool
            await asyncio.sleep(4)

    out_file = RESULTS_DIR / "e2e_stress_test_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[DONE] All E2E stress test results saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
