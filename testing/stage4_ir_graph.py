"""
Stage 4: Architecture IR and Graph Extraction Benchmark.
Tests ir_extractor and graph_extractor on test papers.
Measures node/edge count, taxonomy adherence, 3D scene graph generation, and verification.
"""

import sys
import time
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testing.harness import PAPERS_DIR, RESULTS_DIR
from app.services.text_extraction import text_extractor
from app.services.ir_extraction import ir_extractor
from app.services.graph_extraction import graph_extractor
import pymupdf


def test_ir_and_graph(pdf_path: Path) -> dict:
    doc = pymupdf.open(str(pdf_path))
    raw_text = "\n\n".join(p.get_text() for p in doc)
    doc.close()
    clean_text = text_extractor.strip_tail_sections(raw_text)

    # Use first 20k chars as focused text (mimicking pipeline fallback when arch_text not isolated)
    focused_text = clean_text[:20000]

    paper_title = pdf_path.stem

    # 1. Test canonical ArchitectureIR + 3D SceneGraph
    t0 = time.perf_counter()
    try:
        canonical_ir, scene_graph = ir_extractor.extract(focused_text, paper_title)
        ir_time = round(time.perf_counter() - t0, 3)
        ir_success = canonical_ir is not None
        ir_nodes_count = len(canonical_ir.nodes) if canonical_ir else 0
        ir_edges_count = len(canonical_ir.edges) if canonical_ir else 0
        paradigm = canonical_ir.architecture.paradigm if canonical_ir else "unknown"
        scene_nodes_count = len(scene_graph.nodes_3d) if scene_graph else 0
    except Exception as e:
        ir_time = round(time.perf_counter() - t0, 3)
        ir_success = False
        ir_nodes_count = 0
        ir_edges_count = 0
        paradigm = "error"
        scene_nodes_count = 0

    # 2. Test ArchitectureGraph (ReactFlow / Dagre model)
    t1 = time.perf_counter()
    try:
        tgraph = graph_extractor.extract_graph(focused_text, paper_title)
        graph_time = round(time.perf_counter() - t1, 3)
        tgraph_success = tgraph is not None
        graph_nodes = len(tgraph.nodes) if tgraph else 0
        graph_edges = len(tgraph.edges) if tgraph else 0
    except Exception as e:
        graph_time = round(time.perf_counter() - t1, 3)
        tgraph_success = False
        tgraph = None
        graph_nodes = 0
        graph_edges = 0

    # 3. Test Verification Gate
    t2 = time.perf_counter()
    verified = False
    v_reason = "not attempted"
    if tgraph and graph_nodes > 0:
        try:
            v_res = graph_extractor.verify_graph(focused_text, tgraph, paper_title)
            verified = bool(v_res.get("valid", False))
            v_reason = v_res.get("reason", "")
        except Exception as e:
            v_reason = f"Verification error: {e}"
    v_time = round(time.perf_counter() - t2, 3)

    return {
        "filename": pdf_path.name,
        "ir_extraction": {
            "success": ir_success,
            "latency_sec": ir_time,
            "paradigm": str(paradigm),
            "nodes_count": ir_nodes_count,
            "edges_count": ir_edges_count,
            "scene_3d_nodes": scene_nodes_count
        },
        "graph_extraction": {
            "success": tgraph_success,
            "latency_sec": graph_time,
            "nodes_count": graph_nodes,
            "edges_count": graph_edges,
            "verified": verified,
            "verification_reason": v_reason[:100],
            "verification_time_sec": v_time
        }
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 4 BENCHMARK: ARCHITECTURE IR & GRAPH EXTRACTION")
    print("=" * 70)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_results = []

    for pdf in pdf_files:
        print(f"\n[TESTING] Extracting architecture IR and graph for {pdf.name} ...")
        res = test_ir_and_graph(pdf)
        all_results.append(res)
        print(f"  ArchitectureIR: Success={res['ir_extraction']['success']} ({res['ir_extraction']['latency_sec']}s) | Paradigm={res['ir_extraction']['paradigm']} | Nodes={res['ir_extraction']['nodes_count']}, Edges={res['ir_extraction']['edges_count']}, 3D={res['ir_extraction']['scene_3d_nodes']}")
        print(f"  Graph Extraction: Success={res['graph_extraction']['success']} ({res['graph_extraction']['latency_sec']}s) | Nodes={res['graph_extraction']['nodes_count']}, Edges={res['graph_extraction']['edges_count']}")
        print(f"  Verification Gate: Verified={res['graph_extraction']['verified']} ({res['graph_extraction']['verification_time_sec']}s) - {res['graph_extraction']['verification_reason']}")

    out_file = RESULTS_DIR / "stage4_ir_graph_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Stage 4 results saved to {out_file}")
