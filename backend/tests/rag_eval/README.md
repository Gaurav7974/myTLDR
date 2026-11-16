# RAG eval harness and audit evidence

Evaluation assets retained from the LlamaIndex experiment (2026-11-05). The POC
implementation was discarded; the grounded decision record says: keep the custom
RAG in `backend/app/services/rag.py`, adopt only the small fixes.

## Files

| Path | What it is |
|---|---|
| `RAG_AUDIT.md` / `REPORT.md` | Two audit write-ups of the LlamaIndex POC. Evidence for keeping custom RAG. |
| `EDGE_CASES.md` | Edge-case checklist (empty/short/long docs, scanned PDFs, injection, indexing failure modes). |
| `fixtures/eval_dataset.json` | 60 questions. Only `origin: "proper_cases"` rows are gold. `drafted_for_review` rows are unverified and must not be used as a gate. |
| `fixtures/eval_dataset_audit.json` | The audit's verified subset (6 rows, `source: proper_cases.json`) plus unverified template slots. |
| `fixtures/proper_cases.json` | 6-paper gold substring expectations used by the audit benchmark. |
| `fixtures/benchmark_results.json` | Audit's BM25-only A/B artifact (backend vs POC). Reference only — superseded by `run_benchmark.py`. |
| `run_benchmark.py` | Deterministic offline eval of the **production** `backend` RAG. No LLM, no DB. |
| `last_run.json` | Generated on each run (gitignored). |

## Usage

```bash
cd backend && python tests/rag_eval/run_benchmark.py                 # all gold cases, BM25-only
cd backend && python tests/rag_eval/run_benchmark.py --paper transformer
cd backend && python tests/rag_eval/run_benchmark.py --embed         # live Gemini, needs GOOGLE_EMBEDDING_KEY
```

Both `pytest` and this script run against the same `app.services.rag` code path; the
benchmark must stay green whenever the retrieval tests pass.

## Ground rule

`drafted_for_review` rows are never scored. Any number in a report drawn from those
rows would be invented ground truth.
