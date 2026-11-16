"""Live end-to-end (docker-free fallback). Verifies the full RAG contract against
the *running app* on :8000 with the real embedding + LLM providers.

Writes results to tests/rag_eval/e2e_results.json (gitignored).
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402
import pymupdf  # noqa: E402

BASE = "http://127.0.0.1:8000"


def _post_stream(pdf_path: Path) -> str:
    boundary = "----rag-e2e-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="source"\r\n\r\nupload\r\n'
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="difficulty_level"\r\n\r\nintermediate\r\n'
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="paper.pdf"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + pdf_path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{BASE}/api/stream",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    content_hash = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            text = line.decode("utf-8", "replace")
            if "content_hash" in text and '"stage": "done"' in text:
                try:
                    payload = json.loads(text[text.index("{") :])
                    content_hash = payload.get("content_hash")
                except Exception:
                    pass
    if not content_hash:
        raise RuntimeError(f"no content_hash from stream for {pdf_path.name}")
    return content_hash


def _post_chat(content_hash: str, title: str, query: str) -> dict:
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            f"{BASE}/api/chat",
            json={"paper_hash": content_hash, "paper_title": title, "query": query},
        )
        return resp.json()


def main():
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "eval_dataset.json").read_text(
            encoding="utf-8"
        )
    )
    papers = cases["papers"]
    results = []

    for paper_key, paper in papers.items():
        pdf = REPO / (paper.get("pdf") or "")
        if not pdf.exists():
            print(f"SKIP {paper_key}: pdf not found", file=sys.stderr)
            continue
        gold = [
            c
            for c in cases["cases"]
            if c.get("paper_key") == paper_key and c.get("origin") == "proper_cases"
        ]
        if not gold:
            continue
        print(f"== {paper_key} ==", flush=True)
        content_hash = _post_stream(pdf)
        time.sleep(20)  # background indexing
        for case in gold:
            t0 = time.time()
            try:
                out = _post_chat(content_hash, paper.get("title", ""), case["question"])
            except Exception as e:
                out = {"answer": f"ERROR {e}", "citations": []}
            secs = round(time.time() - t0, 1)
            hay = " ".join(
                str(c.get("section_title", "")) + " " + str(c.get("content", ""))
                for c in out.get("citations", [])
            ).lower()
            hit = any(e.lower() in hay for e in case.get("expect_substrings", []))
            abstain = case.get("abstain")
            results.append(
                {
                    "paper": paper_key,
                    "question": case["question"][:80],
                    "status": "hit" if hit else "miss",
                    "expected": "abstain" if abstain else "answer",
                    "hit_expected": (hit and not abstain)
                    or ((not out.get("citations")) and abstain),
                    "citations": len(out.get("citations", [])),
                    "latency_s": secs,
                    "answer_chars": len(out.get("answer", "")),
                    "llm_failed": out.get("answer", "").startswith(
                        "The AI service is temporarily"
                    ),
                }
            )
            print(json.dumps(results[-1]), flush=True)

    out_path = Path(__file__).parent / "e2e_results.json"
    out_path.write_text(json.dumps({"rows": results}, indent=2), encoding="utf-8")
    hits = sum(1 for r in results if r["hit_expected"])
    print(f"\nwrote {out_path}  hit_expected {hits}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
