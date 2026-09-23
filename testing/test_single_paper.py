"""
Single Paper End-to-End Test Runner.
Executes the full production pipeline on one paper at a time.
Provides detailed per-step diagnostic logging and saves the full result payload.
"""

import sys
import asyncio
import time
import argparse
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


async def run_single_paper(paper_id: str, max_pages: int = 65, arch_chars: int = 50000, section_ceiling: int = 40000):
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        print(f"[ERROR] Paper file {pdf_path} not found!")
        return

    pdf_bytes = pdf_path.read_bytes()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    doc.close()

    # Apply test settings
    orig_max_pages = settings.MAX_PDF_PAGES
    orig_arch_chars = settings.ARCH_SECTION_LLM_MAX_CHARS
    orig_ceiling = settings.SECTION_CHAR_CEILING

    settings.MAX_PDF_PAGES = max_pages
    settings.ARCH_SECTION_LLM_MAX_CHARS = arch_chars
    settings.SECTION_CHAR_CEILING = section_ceiling

    print("=" * 80)
    print(f"TESTING PAPER: {paper_id} ({pdf_path.name})")
    print(f"Page Count: {page_count} | Size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
    print(f"Active Settings: MAX_PDF_PAGES={settings.MAX_PDF_PAGES}, "
          f"ARCH_SECTION_LLM_MAX_CHARS={settings.ARCH_SECTION_LLM_MAX_CHARS}, "
          f"SECTION_CHAR_CEILING={settings.SECTION_CHAR_CEILING}")
    print("=" * 80)

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
            print(f"[{elapsed:6.2f}s] Stage: {stage:<12} | {msg}")

            if stage == "done":
                final_event = event
            elif stage == "error":
                final_event = event
                print(f"[ERROR EVENT] {event.get('message')}")
                break

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()
        final_event = {"stage": "exception", "error": str(e)}

    finally:
        total_time = round(time.perf_counter() - t0, 2)
        settings.MAX_PDF_PAGES = orig_max_pages
        settings.ARCH_SECTION_LLM_MAX_CHARS = orig_arch_chars
        settings.SECTION_CHAR_CEILING = orig_ceiling

    print("-" * 80)
    print(f"Total Execution Time: {total_time}s")
    print(f"Stages Traversed: {' -> '.join(stages_seen)}")
    print("-" * 80)

    if final_event and final_event.get("stage") == "done":
        explanation = final_event.get("explanation") or {}
        graph = final_event.get("graph") or {}
        arch_ir = final_event.get("architecture_ir") or {}
        equations = final_event.get("equations") or []
        code_snippets = final_event.get("code_snippets") or []
        diagram_available = final_event.get("diagram_available", False)

        graph_nodes = graph.get("nodes", []) if graph else []
        graph_edges = graph.get("edges", []) if graph else []
        ir_nodes = arch_ir.get("nodes", []) if arch_ir else []
        ir_edges = arch_ir.get("edges", []) if arch_ir else []
        paradigm = arch_ir.get("architecture", {}).get("paradigm", "unknown") if arch_ir else "unknown"

        overview = explanation.get("overview", {})
        key_specs = explanation.get("key_specs", [])
        contributions = explanation.get("contributions", [])
        key_findings = explanation.get("key_findings", [])
        limitations = explanation.get("limitations", [])

        # Collect cited pages
        cited_pages = set()
        for item in (key_specs or []) + (contributions or []) + (key_findings or []) + (limitations or []):
            if isinstance(item, dict) and "source" in item and isinstance(item["source"], dict):
                p = item["source"].get("page")
                if p:
                    cited_pages.add(p)

        sample_eq = ""
        if equations and isinstance(equations[0], dict):
            sample_eq = str(equations[0].get("latex", ""))[:60]
        elif equations:
            sample_eq = str(equations[0])[:60]

        sample_code = None
        if code_snippets and isinstance(code_snippets[0], dict):
            sample_code = str(code_snippets[0].get("title", ""))
        elif code_snippets:
            sample_code = str(code_snippets[0])[:60]

        summary_metrics = {
            "paper_id": paper_id,
            "page_count": page_count,
            "total_time_sec": total_time,
            "diagram": {
                "available": diagram_available,
                "source": graph.get("source") if graph else None,
                "layout_mode": graph.get("layout_mode") if graph else None,
                "nodes_count": len(graph_nodes),
                "edges_count": len(graph_edges),
                "sample_nodes": [n.get("label") for n in graph_nodes[:5]]
            },
            "architecture_ir": {
                "paradigm": paradigm,
                "nodes_count": len(ir_nodes),
                "edges_count": len(ir_edges),
                "sample_nodes": [n.get("label") for n in ir_nodes[:5]]
            },
            "math_and_code": {
                "equations_extracted": len(equations),
                "code_snippets_extracted": len(code_snippets),
                "sample_equation": sample_eq,
                "sample_code_title": sample_code
            },
            "pedagogical_summary": {
                "one_sentence": overview.get("one_sentence"),
                "problem": overview.get("problem"),
                "solution": overview.get("solution"),
                "specs_count": len(key_specs),
                "contributions_count": len(contributions),
                "findings_count": len(key_findings),
                "limitations_count": len(limitations),
                "cited_pages": sorted(list(cited_pages)),
                "max_cited_page": max(cited_pages) if cited_pages else None
            }
        }

        print("\nDETAILED ANALYSIS RESULT:")
        print(json.dumps(summary_metrics, indent=2))

        # Save summary and full payload
        res_file = RESULTS_DIR / f"{paper_id}_summary.json"
        with open(res_file, "w") as f:
            json.dump(summary_metrics, f, indent=2)

        full_payload_file = RESULTS_DIR / f"{paper_id}_full_event.json"
        with open(full_payload_file, "w") as f:
            # Strip huge parsed_text markdown for json serialization if needed
            clean_event = dict(final_event)
            if "parsed_text" in clean_event and isinstance(clean_event["parsed_text"], dict):
                clean_event["parsed_text"] = {"page_count": clean_event["parsed_text"].get("page_count")}
            json.dump(clean_event, f, indent=2)

        print(f"\n[SAVED] Summary saved to {res_file}")
        return summary_metrics

    else:
        print("\n[RESULT] Pipeline failed to produce 'done' event.")
        return {"paper_id": paper_id, "status": "FAILED", "event": final_event}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single paper E2E")
    parser.add_argument("paper_id", type=str, help="arXiv ID of the paper (e.g. 2511.21631)")
    parser.add_argument("--max-pages", type=int, default=65, help="Override MAX_PDF_PAGES")
    parser.add_argument("--arch-chars", type=int, default=50000, help="Override ARCH_SECTION_LLM_MAX_CHARS")
    parser.add_argument("--ceiling", type=int, default=40000, help="Override SECTION_CHAR_CEILING")

    args = parser.parse_args()
    asyncio.run(run_single_paper(args.paper_id, args.max_pages, args.arch_chars, args.ceiling))
