"""Fast live RAG contract smoke test (no /api/stream wait).

Extracts the transformer PDF with the same code path the app uses, computes the
content_hash the server would use, then calls the real /api/chat with full_text.
Exercises: retrieval, citation format, abstention, numbers rule, per-question
latency. Writes results to tests/rag_eval/e2e_results.json (gitignored).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

from app.services.pipeline import compute_content_hash  # noqa: E402
from app.services.text_extraction import text_extractor  # noqa: E402

BASE = "http://127.0.0.1:8000"
PDF = REPO / "testing" / "papers" / "1706.03762.pdf"
FIXTURES = Path(__file__).parent / "fixtures"


def main():
    extraction = text_extractor.extract_from_pdf_bytes(PDF.read_bytes())
    full_text = extraction.text
    content_hash = (
        compute_content_hash(full_text)
        if callable(compute_content_hash)
        else hashlib.sha256(full_text.encode()).hexdigest()
    )
    print(
        f"pages={extraction.page_count} chars={len(full_text)} hash={content_hash[:12]}",
        flush=True,
    )

    cases = [
        c
        for c in json.loads(
            (FIXTURES / "eval_dataset.json").read_text(encoding="utf-8")
        )["cases"]
        if c.get("paper_key") == "transformer" and c.get("origin") == "proper_cases"
    ]
    results = []
    for case in cases:
        t0 = time.time()
        with httpx.Client(timeout=180) as client:
            resp = client.post(
                f"{BASE}/api/chat",
                json={
                    "paper_hash": content_hash,
                    "paper_title": "Attention Is All You Need",
                    "query": case["question"],
                    "full_text": full_text,
                },
            )
            out = resp.json()
        secs = round(time.time() - t0, 1)
        hay = " ".join(
            str(c.get("section_title", "")) + " " + str(c.get("content", ""))
            for c in out.get("citations", [])
        ).lower()
        answer = out.get("answer", "")
        hit = any(e.lower() in hay for e in case.get("expect_substrings", []))
        row = {
            "question": case["question"],
            "expect": case["expect_substrings"],
            "citations": len(out.get("citations", [])),
            "citation_hit": hit,
            "answer_chars": len(answer),
            "latency_s": secs,
            "llm_failed": answer.startswith("The AI service is temporarily"),
            "answer_sample": answer[:200],
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    hits = sum(1 for r in results if r["citation_hit"])
    fails = sum(1 for r in results if r["llm_failed"])
    out_path = FIXTURES.parent / "e2e_results.json"
    out_path.write_text(json.dumps({"rows": results}, indent=2), encoding="utf-8")
    print(
        f"\ncitation_hit {hits}/{len(results)}  llm_failures {fails}  wrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
