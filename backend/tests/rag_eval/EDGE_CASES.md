# Edge cases — investigated (offline, code-level + script llamaindex/audit/run_edge_cases.py)
See run_edge_cases.py output + report section 11. Each case: expected vs actual from code paths:
empty/short/long doc, scanned (OCR fallback uncommitted diff), equations/tables, duplicate upload, failed embedding, partial index, DB down, provider outage, concurrent upload, multi-worker, restart, stale cache, missing metadata, wrong dim, other-model vectors, re-index after chunker change, prompt injection, unanswerable, long history.
Unverified live items are marked UNVERIFIED in the report.
