"""Deterministic offline RAG eval against the production code.

Usage (from repo root):
    python backend/tests/rag_eval/run_benchmark.py            # BM25-only, all gold cases
    python backend/tests/rag_eval/run_benchmark.py --paper transformer
    python backend/tests/rag_eval/run_benchmark.py --embed    # live Gemini path (needs GOOGLE_EMBEDDING_KEY)

Metrics: recall@5 (any gold substring in top-5), MRR, nDCG@5 (binary), precision@5.
Only cases with `origin == "proper_cases"` are scored — drafted_for_review rows are
never used as a gate. No LLM calls. Skips papers whose PDFs are missing.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import pymupdf  # noqa: E402

from app.services import rag as rag_module  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def metrics(retrieved: list[dict], expects: list[str]):
    found = None
    for rank, c in enumerate(retrieved, 1):
        hay = (
            str(c.get("section_title", "")) + " " + str(c.get("content", ""))
        ).lower()
        if any(e.lower() in hay for e in expects):
            found = rank
            break
    recall = 1.0 if found else 0.0
    mrr = 1.0 / found if found else 0.0
    dcg = sum(
        1.0 / math.log2(r + 1)
        for r, c in enumerate(retrieved, 1)
        if any(
            e.lower()
            in (
                str(c.get("section_title", "")) + " " + str(c.get("content", ""))
            ).lower()
            for e in expects
        )
    )
    prec = sum(
        1
        for c in retrieved
        if any(
            e.lower()
            in (
                str(c.get("section_title", "")) + " " + str(c.get("content", ""))
            ).lower()
            for e in expects
        )
    ) / max(1, len(retrieved))
    return recall, mrr, dcg, prec


def _find_pdf(paper: dict) -> Path | None:
    for candidate in (
        REPO / paper.get("pdf", ""),
        REPO / "papers" / Path(paper.get("pdf", "")).name,
        REPO / "testing/papers" / Path(paper.get("pdf", "")).name,
    ):
        if candidate.exists():
            return candidate
    return None


async def _retrieve(chunks, embeddings, paper_hash: str, query: str, top_k: int = 5):
    rag_module._MEMORY_CHUNK_STORE[paper_hash] = [
        {"section_title": c["section_title"], "content": c["content"], "embedding": e}
        for c, e in zip(chunks, embeddings)
    ]
    rag_module._MEMORY_VECTOR_STORE[paper_hash] = rag_module._MEMORY_CHUNK_STORE[
        paper_hash
    ]
    return await rag_module.rag_service.retrieve_chunks(
        None, paper_hash, query, top_k=top_k
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=None, help="Only evaluate this paper key")
    parser.add_argument(
        "--embed", action="store_true", help="Use live Gemini embeddings"
    )
    args = parser.parse_args()

    cases_file = json.loads(
        (FIXTURES / "eval_dataset.json").read_text(encoding="utf-8")
    )
    papers = cases_file["papers"]
    cases = [c for c in cases_file["cases"] if c.get("origin") == "proper_cases"]

    rows = []
    for paper_key, paper in papers.items():
        if args.paper and paper_key != args.paper:
            continue
        pdf = _find_pdf(paper)
        if pdf is None:
            print(
                f"SKIP {paper_key}: pdf not found ({paper.get('pdf')})", file=sys.stderr
            )
            continue
        full_text = "\n".join(p.get_text() for p in pymupdf.open(str(pdf)))
        chunks = rag_module.chunk_paper_text(full_text)
        print(
            f"{paper_key}: pages={pymupdf.open(str(pdf)).page_count} chars={len(full_text)} chunks={len(chunks)}"
        )
        paper_cases = [c for c in cases if c.get("paper_key") == paper_key]
        if not paper_cases:
            continue

        embeddings: list = [None] * len(chunks)
        if args.embed:

            async def _embed_all():
                return await asyncio.gather(
                    *[rag_module.compute_dense_embedding(c["content"]) for c in chunks]
                )

            try:
                embeddings = list(asyncio.run(_embed_all()))
                if any(e is None for e in embeddings):
                    print(
                        f"WARN {paper_key}: some embeddings failed — dense path unavailable, BM25 only",
                        file=sys.stderr,
                    )
                    embeddings = [None] * len(chunks)
            except Exception as e:
                print(
                    f"WARN {paper_key}: embedding failed ({e}) — BM25 only",
                    file=sys.stderr,
                )

        for case in paper_cases:
            t0 = time.perf_counter()
            if args.embed and any(embeddings):
                retrieved = asyncio.run(
                    _retrieve(chunks, embeddings, f"eval-{paper_key}", case["question"])
                )
            else:
                retrieved = rag_module.bm25_keyword_search(
                    chunks, case["question"], top_k=5
                )
            ms = (time.perf_counter() - t0) * 1000
            recall, mrr, ndcg, prec = metrics(
                retrieved, case.get("expect_substrings", [])
            )
            rows.append(
                {
                    "id": case["id"],
                    "paper": paper_key,
                    "type": case.get("type"),
                    "recall": recall,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "prec": prec,
                    "hits": len(retrieved),
                    "ms": round(ms, 2),
                }
            )
            print(json.dumps(rows[-1]))

        rag_module._MEMORY_CHUNK_STORE.pop(f"eval-{paper_key}", None)
        rag_module._MEMORY_VECTOR_STORE.pop(f"eval-{paper_key}", None)

    if not rows:
        print("no cases evaluated", file=sys.stderr)
        return 1

    def agg(k):
        return round(sum(r[k] for r in rows) / len(rows), 4)

    summary = {m: agg(m) for m in ("recall", "mrr", "ndcg", "prec")}
    print("\nsummary", json.dumps(summary), "n=", len(rows))
    out = Path(__file__).parent / "last_run.json"
    out.write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8"
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
