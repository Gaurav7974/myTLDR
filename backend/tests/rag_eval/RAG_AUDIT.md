# Comprehensive RAG Audit + LlamaIndex Evaluation — myTLDR

**Role:** Senior AI Engineer / RAG Architect
**Date:** 2026-11-05
**Branches audited:** `main` @ `991d191afbb223e40cfe9aa2f90b72f7f67d4421` (local HEAD = `origin/main`)
**Uncommitted local changes:** `.gitignore`, `backend/app/services/text_extraction.py` (vision-OCR fallback `_vision_ocr`/`_vision_transcribe`), untracked `backend/tests/test_ocr_fallback.py`, untracked `llamaindex/` (the POC audited here).
**Deployed commit:** **UNVERIFIED** — no deployment metadata accessible in-repo that pins a commit (Railway config only shows `railway.toml` start command). Do not assume deployed == local HEAD.
**Isolation:** Production RAG path (`backend/`) was **not modified** by this audit. All POC code lives in `llamaindex/`.

## 1. Executive verdict

**Recommendation: Keep the existing custom RAG. Adopt selected ideas from the POC, not the LlamaIndex framework.**

| Question | Answer |
|---|---|
| Does LlamaIndex improve retrieval quality here? | **No measurable improvement.** On 16 live questions (real PDFs, BM25 mode — see §10) recall@5 is **tied** (0.938 both). MRR +0.011 for the POC is from a **stemming fix + RRF fusion that are custom logic**, not from LlamaIndex primitives. |
| Does it improve engineering? | Marginally on durability (persisted `chunks.json`, async wrappers), but both are **~30 lines of custom code**, not a framework dependency. |
| Does it improve cost/latency? | **Worse.** The POC's `VectorStoreIndex` construction re-embeds every chunk a second time (see §8). `llama-index-core` adds ~50 transitive deps. |
| Can it reuse `PaperChunk` + pgvector? | **Not out of the box.** The POC does not use a Postgres vector store at all — it uses `SimpleVectorStore` + a custom `chunks.json`. A custom `BasePydanticVectorStore` adapter would be required to hit `paper_chunks`. |
| Should we migrate? | **No.** Backport the three small fixes (§13) to `backend/app/services/rag.py` and keep the framework out. |

**Bottom line:** This is a 500-line custom system that is already faithful to its contract (grounded, cited, fail-closed). LlamaIndex would add a heavy abstraction layer that duplicates — not replaces — the parts that matter (`PaperChunk` schema, Gemini embeddings, the grounding prompt, Mistral/Groq client).

## 2. Actual repository & deployment state

- Branch: `main`, HEAD `991d191` = `origin/main`. Log head: `991d191 fix(build,rag,ui): resolve Vercel TS2339…`.
- Working tree dirty (see above). The `llamaindex/` directory is **untracked** — it is not part of the deployed codebase nor the HEAD commit.
- Backend test suite: **99 passed** (`cd backend; python -m pytest tests -q`, 2026-11-05).
- POC unit tests (offline subset): **16 passed** (`cd llamaindex; python -m pytest tests/test_chunking.py tests/test_retriever.py tests/test_contract.py tests/test_chat_mock.py -q`). Full suite collects 29 tests, but live/E2E tests (`test_live*`, `test_full_pipeline_e2e`, `test_proper_papers_e2e`) hang without keys/network and were **not** re-run in this audit (prior run artifacts under `llamaindex/data/verification/` were read instead).
- POC env: `llama-index-core==0.14.25` installed on system Python; **not** in `backend/venv` (good separation, but means the backend venv cannot import the POC — verified `ModuleNotFoundError: No module named 'llama_index'`).
- `llama-index-postgres` and `llama-index-embeddings-openai` are **not installed** (verified via `pip show`).
- Deployment: `railway.toml` → `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 300 --limit-concurrency 20`. Single worker mitigates the multi-worker cache issue (§5E) but does not eliminate it.

## 3. Existing RAG architecture (traced)

Files: `backend/app/services/rag.py` (572 lines), `backend/app/services/pipeline.py`, `backend/app/api/endpoints/chat.py`, `backend/app/models/orm.py`, `backend/app/core/{config,database}.py`, `backend/app/services/{llm,text_extraction}.py`.

```
PDF → text_extraction.extract_from_pdf_bytes (PyMuPDF / Mistral OCR / vision fallback)
    → full_text (flat, references stripped)
    → pipeline.process_paper: compute_content_hash = SHA256(normalize_text(full_text))
    → asyncio.create_task(_index_rag_background) → rag_service.index_paper(session, hash, text)
        → chunk_paper_text(full_text)            [600 chars / 100 overlap]
        → _MEMORY_CHUNK_STORE[hash] = chunks (BM25-ready, always)
        → compute_dense_embedding per chunk (Gemini, Semaphore(4))
        → _MEMORY_VECTOR_STORE[hash] = same (L2) + pgvector INSERT … ON CONFLICT DO NOTHING (L1)
    → stream endpoint yields SSE "done" with content_hash
Chat → POST /api/chat {paper_hash, query, history, difficulty, full_text?}
    → rag_service.answer_question(db, hash, title, query, history, difficulty, full_text)
        → on-the-fly BM25 chunk if hash missing/incomplete
        → anaphoric query expansion (last user turn prepended if query <6 words or contains this/that/it/…)
        → retrieve_chunks(db, hash, search_query, top_k=5)
            1. pgvector: 1-(embedding <=> :vec) ORDER BY <=> LIMIT 5, filter ≥0.25
            2. in-memory cosine (real embeddings in _MEMORY_CHUNK_STORE), filter ≥0.25
            3. bm25_keyword_search(mem_chunks, query, top_k=5), filter ≥0.25
        → system prompt (grounding/numbers-refusal) + [Source N – Section]: chunk
        → groq_client.call(messages, model=MISTRAL_LLM_MODEL, temperature=0.3)
        → {answer, citations[{section_title, content, similarity}], indexing_in_progress?}
```

Three-level fallback, no hybrid fusion: path 1 is used exclusively if it yields ≥1 above-threshold row; otherwise path 2; otherwise path 3. Vector and keyword scores are **never combined**.

## 4. Detailed implementation audit

### A. Chunking — `rag.py:73-128`

- **Char-based, not token-based.** 600 chars ≈ 150 tokens; overlap 100 chars. Simple, deterministic, but chunks can split mid-equation — boundary is `rfind('. ' / '.\n' / '; ')` with hard-cut fallback at `chunk_size` if boundary < `chunk_size//2`.
- **Section detection:** markdown `# …` headings + regex for `Abstract|Introduction|Related Work|Architecture|Model Architecture|Methodology|Method|Experiments|Results|Discussion|Conclusion|Attention|Training|Background`, line <80 chars. Misses non-standard headings (e.g. "3.2.1 Scaled Dot-Product…" is *not* matched — it becomes part of the buffer, section label stays whatever was last matched). No page-number metadata — chunks carry only `{chunk_index, section_title, content}`.
- **Equations/tables/captions:** flattened to plain joined lines (`buffer += " " + trimmed`); LaTeX is preserved as text but not isolated; quotes/captions not distinguished.
- **Duplicates:** overlap retains 100 chars → adjacent chunks share ~1 sentence. No de-duplication.
- **Tail rule:** final buffer kept only if `len>50` chars — short trailing fragments are silently dropped.
- **Re-indexing:** none. No chunker version in DB (`PaperChunk` has no version column). Changing chunker → old chunks stay (see §5E partial-index bug).

**Recommendation for research papers:** Keep 600/100 for now (proven on live papers), but (a) add `page_number` from `extraction.pages` before chunking, (b) keep `section_title` as metadata, (c) add a `chunker_version` column so re-index is explicit. A token-based splitter (e.g. `SentenceSplitter` with 512 tokens/50 overlap) is **not** demonstrably better for this corpus — see benchmark note in §10 — and would force a full re-embed.

### B. Embeddings — `rag.py:41-70`

- **Provider/model:** Google Gemini `gemini-embedding-001`, **3072-dim** (`settings.EMBEDDING_DIM`), via `POST https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:embedContent`. Same model for query and documents (compatible).
- **Cache:** `_embedding_cache` LRU 512, key `sha256(text)[:32]` — **no model/dim in key**. Model change → stale vectors served.
- **Failure/rate limits:** single attempt, 10 s timeout, no retry. Indexing uses `Semaphore(4)` + `asyncio.gather` per chunk (one HTTP call per chunk — no batch API). Query embedding is one call per retrieval (cache helps repeats).
- **Versioning:** none in DB. `paper_chunks.embedding` is `Vector(3072)` (falls back to `Text` if pgvector missing — a mixed-dim vector from another model would fail insert or silently poison cosine).
- **POC improvement:** `llamaindex/src/embeddings.py` includes `{model}:{dim}:{text}` in the cache key and persists `embedding_model`/`embedding_dim` in `chunks.json`, invalidating mismatches. **This idea is worth backporting; the framework is not.**

### C. Retrieval — `rag.py:340-424` (bugs highlighted)

**Bugs / architectural limits (actual, not theoretical):**

1. **No hybrid fusion.** Vector and BM25 are *fallbacks*, not a fusion. An exact-term query that happens to have a weak vector hit (≥0.25) never sees BM25; a strong keyword hit is discarded if vector path returns anything. The POC's RRF + `0.6·vec + 0.4·bm25` rerank is the only genuine retrieval improvement in either codebase — and it is **~70 lines of custom code** (`llamaindex/src/retriever.py:271-326`), not LlamaIndex magic.
2. **BM25 IDF floor at 1.5** (`rag.py:220`). In a single-paper corpus, topic terms appear in nearly all chunks → raw IDF → 0 → floor inflates them to maximum weight. This is a deliberate hack to keep "attention" rankable, but it **inverts BM25**: the most common terms get the highest weights. It works empirically on these papers, but it is not IDF.
3. **BM25 normalization `score/10.0`** (`rag.py:238`) is arbitrary. Scores are unbounded; the same 0.25 threshold is applied to cosine (0–1, well-calibrated) and to `min(score/10,1)`. Threshold semantics differ per path — tuning `RAG_RELEVANCE_THRESHOLD` for one path mistunes the other.
4. **Stemming mismatch** (`rag.py:152-164`, `elif` chain): `stem("embeddings") → "embedding"` but `stem("embedding") → "embedd"`. Morphological variants never meet. POC `_normalize_token` (`retriever.py:142-166`) uses sequential strips and fixes this — verified as the +1 hit on the Transformer fixture.
5. **Token regex `\b[a-z]{3,}\b`** drops numbers, symbols, and short acronyms. `"How many tokens was Qwen pretrained on?"` (expect `3 trillion`) can never match lexically — *both* systems score recall 0.0 on it (§10). Equations (`QK^T`, `d_k`) are invisible to BM25.
6. **pgvector SQL** (`rag.py:357-369`): `ORDER BY embedding <=> :query_vector LIMIT :top_k` with the vector string-cast (`str(query_embedding)`). Correct operator, but **no HNSW/IVFFlat index** exists anywhere (no `CREATE INDEX` in `orm.py` or `init_db` — grep confirms). Every query is a full scan of `paper_chunks` filtered by `paper_hash`. Fine at 67 chunks/paper; will degrade at scale.
7. **In-memory cosine duplicates pgvector** (`rag.py:392-415`) with identical threshold. It exists solely so chat works when the DB is down — but it only works if `index_paper` ran in the **same process** (see §5E).
8. **No dedup/diversity/rerank.** Overlapping chunks can both appear in top-5.
9. **Query rewriting** is only the anaphoric prepend (`rag.py:454-463`). No HyDE/query-expansion.
10. **Top-k inconsistency:** `retrieve_chunks(top_k=4)` default vs `answer_question` passing `top_k=5`. Cosmetic.

### D. Generation & citations — `rag.py:426-569`

- **Prompt** is strong: `<role_and_scope>`, `<grounding_and_citations>` with an explicit **Numbers rule** (every number must appear verbatim in the cited chunk; never derive/interpolate), refusal handling, difficulty-aware formatting. Context is wrapped as `--- BEGIN PAPER CONTEXT ---` with "untrusted source material" labels — **prompt-injection hardening is real** (`test_rag_grounding.py` asserts the numbers rule is present).
- **Citation traceability is prompt-level only.** `citations` are the retrieved chunks (section + content + similarity) returned alongside the answer. The LLM is asked to write `[Source N – Section Title]` inline, but nothing verifies the claim text against the chunk (no entailment check, no chunk-ID anchors). "Citation precision" in this system = "the retrieved chunk was relevant", not "the sentence was entailed".
- **Context budget:** 5 chunks × ~700 chars ≈ 3.5k chars — safe for the 128k model, but there is **no explicit budget enforcement**. History turns are truncated to 4000 chars each (`rag.py:549`) with **no turn cap** — a long conversation grows unbounded until `llm.py:137` 413-shrink trims the *last* message, which is the user prompt containing the grounding context. Shrinking grounding is the worst possible truncation.
- **Provider failure:** returns `"AI service is temporarily unavailable"` with citations preserved (`rag.py:559-564`). Good.

### E. Production engineering

1. **`groq_client.call` is synchronous and blocks the event loop.** `answer_question` is `async` but calls `ChatClient.call` → `requests.post(timeout=120)` directly (`rag.py:554`, `llm.py:119`). Every chat request blocks the FastAPI worker for the full LLM latency. POC `aanswer_question` wraps with `asyncio.to_thread` — wrapping is a 1-line fix in backend too.
2. **Fire-and-forget indexing.** `pipeline.py:129` — `asyncio.create_task(_index_rag_background(...))` with no handle, no persistence, no retry. Worker restart = lost indexing. If `index_paper` is mid-flight when the process dies, the DB holds a *partial* index.
3. **In-memory stores are process-local.** `_MEMORY_CHUNK_STORE` (LRU 100) and `_MEMORY_VECTOR_STORE` are module dicts. Railway `--workers 1` saves us today; scaling to ≥2 workers means chat on worker B can't see papers indexed on worker A (pgvector path still works if DB is up, but BM25/in-memory fallback returns `[]` → `indexing_in_progress: True` forever). This is the **strongest argument for the POC's persisted store** — but the right fix is Postgres-only retrieval (or Redis), not `chunks.json` on ephemeral container disk.
4. **Partial-index handling is broken for chunker changes.** `index_paper` checks `existing >= len(chunks)` → skip; else `ON CONFLICT DO NOTHING` per `(paper_hash, chunk_index)`. If the chunker changes, old chunks are never deleted — new text for the same `chunk_index` is ignored and stale chunks persist indefinitely. No `chunker_version` to invalidate.
5. **Async session safety is OK.** `chat.py` opens a fresh `async_session()` per request; `pipeline.py` background task opens its own; `get_db_optional` yields `None` on failure so the stream endpoint degrades intentionally. `retrieve_chunks` rolls back after a failed pgvector query — safe within a request-scoped session.
6. **Observability:** structured-ish logging only (`get_logger`, request-id middleware in `main.py`). No metrics, no tracing, no p50/p95 dashboards, no embedding/LLM cost tracking.
7. **Cost:** ~67 embedding calls/paper (Transformer) + 1/query, all single-shot (no `batchEmbedContents`). No caching of query embeddings beyond the 512-entry LRU.

## 5. LlamaIndex capability comparison

Studied against `https://developers.llamaindex.ai/python/framework/` (fetched 2026-11-05). Installed: `llama-index-core 0.14.25`.

| Capability | What LlamaIndex provides | Already in myTLDR? | Actually better? | Reuses `PaperChunk`/embeddings? | Migration cost / deps |
|---|---|---|---|---|---|
| Document & TextNode abstractions | `Document`, `TextNode` (text+metadata+embedding) | `chunk_paper_text` dicts `{chunk_index, section_title, content}` | No — same payload, more ceremony | Yes (POC `to_nodes` maps 1:1) | `llama-index-core` alone |
| Node parsers / metadata | `SentenceSplitter`, metadata extractors | Custom regex section detect | No evidence for this corpus; would force re-embed | N/A | Re-embed all papers |
| Ingestion pipelines | `IngestionPipeline` (transform→embed→store) | `index_paper` gather+insert | No — pipeline is our `index_paper` | Hard if not using LI store | Moderate |
| Embedding integrations | `BaseEmbedding` ABC, `VectorStoreIndex.from_documents` | Direct `httpx` to Gemini embedContent | No — POC subclasses `BaseEmbedding` and still calls the same endpoint | Yes, same model | None if we keep adapter |
| PostgreSQL/pgvector store | `llama-index-vector-stores-postgres` (PGVectorStore) | `PaperChunk` + `pgvector` + raw `<=>` SQL | **Not better — different schema.** Default creates its own tables | **No.** Requires custom `BasePydanticVectorStore` adapter to hit `paper_chunks` | Adapter + `@lru_cache` graphstore etc. Package not installed |
| Custom vector-store integration | Implement `VectorStore` ABC | — | N/A | Would need pgvector adapter | ~150 lines, ongoing |
| Dense retrieval | `VectorStoreRetriever` over any store | `retrieve_chunks` L1/L2 | Equal (same cosine/L1) | Yes with adapter | Adapter |
| Hybrid / BM25 | `BM25Retriever` (in-memory nodes only); `QueryFusionRetriever` w/ RRF | Hand-rolled BM25 (`rag.py:173-245`) | Fusion primitives exist but **POC didn't use them** — hand-rolled `rrf_fuse`+`rerank_fused` | BM25 over `chunks.json` in POC | Custom either way |
| Retrieval fusion | RRF / weighted `QueryFusionRetriever` | None (fallback chain) | **Yes, the idea is good** — but implementable in ~70 lines without LI | Yes | Backport to `rag.py` |
| Reranking / postprocessors | `NodePostprocessor` (e.g. `SentenceTransformersRerank`) | None | Cross-encoder rerank *could* help ranking, but adds HF model download + latency; POC used a linear `0.6·vec+0.4·bm25` blend, not a cross-encoder | Yes | `sentence-transformers` + model weights |
| Query transformations | `TransformQueryEngine` (HyDE, sub-question) | Anaphoric prepend only | Unproven here; costs 1–2 extra LLM calls/query, risks grounding | Yes | Prompt design + latency |
| Response synthesis | `QueryEngine` (compact/tree/accumulate) | Custom prompt in `answer_question` | **No — stock synth would weaken the numbers rule.** Must keep custom prompt anyway | Yes | Replace prompt + re-validate |
| Async support | `aquery`/`aindex` on indexes | `asyncio.to_thread` wrappers in POC; backend LLM call is **blocking** | POC's `asyncio.to_thread` pattern is the fix, not LI | Yes | 1 line in backend |
| Instrumentation / observability | `llama_index.core.instrumentation` spans | Logging only | Could adopt spans incrementally, but no retrieval-quality effect | Yes | Opt-in, low risk |

**Also evaluated:** `llama-index-vector-stores-postgres` PGVectorStore docs (404 on the framework `vector_stores/` path — the current docs site reorganized under `module_guides/indexing/index_stores/`). Regardless of URL, the integration creates `llama_index_node`-style tables; it does **not** detect or reuse an existing `paper_chunks` table. A custom adapter is unavoidable either way.

**Key finding:** Every POC improvement traces to **custom code** (sequential stemming, RRF `rrf_fuse`, `0.6/0.4` rerank, `chunks.json` persistence, model-aware cache key, `asyncio.to_thread`). The LlamaIndex imports that remain are `TextNode`, `VectorStoreIndex`, `SimpleVectorStore`, `BaseEmbedding`, `PrivateAttr` — thin wrappers around the same Gemini endpoint and cosine math. **The framework is not the source of value.**

## 6. POC implementation (already exists in `llamaindex/`)

| File | Role |
|---|---|
| `src/config.py` | DRY env loader; lookup `llamaindex/.env → .env → backend/.env`; same key names (`OCR_KEY`, `GROQ_API_KEY`, `GOOGLE_EMBEDDING_KEY`) |
| `src/embeddings.py` | `GeminiEmbedding(BaseEmbedding)` + module helpers `embed_text`/`embed_batch` (httpx async, cache key includes model+dim, `asyncio.to_thread`-safe sync wrappers) |
| `src/chunking.py` | 600/100 char parity chunker + `to_nodes(chunks, paper_hash) → TextNode` |
| `src/ingest.py` | Canonical `data/<sha256(paper_hash)>/chunks.json` (+ `registry-entry.json`, optional `llama_index/` dir). **Always persisted** — both BM25 and vector modes survive restart. Validates `embedding_model`/`embedding_dim` on load. Idempotent skip if persisted chunks ≥ new chunks |
| `src/retriever.py` | `vector_search` (in-memory cosine) + `bm25_search` (ported, fixed stemming) → `rrf_fuse` (k=60, key=`content[:200]+section`) → `rerank_fused` (`RAG_VECTOR_WEIGHT=0.6` / `RAG_BM25_WEIGHT=0.4`, normalized by max similarity). Threshold 0.25. Top-k 5 |
| `src/chat.py` | `answer_question` (identical return shape, prompt, numbers rule, history normalization, BM25-only `full_text` bootstrap **with zero network calls**), plus `aanswer_question`/`aindex_paper`/`aretrieve` via `asyncio.to_thread` |
| `src/llm_client.py` | Mistral-first (key rotation) → Groq fallback, JSON-mode hardening, 429 backoff, 413-shrink — parity copy of `backend/app/services/llm.py` |
| `src/eval.py` | `recall@5`/`MRR` harness (full-chunk matching, honest BM25 degrade without key) + 5-case Transformer demo |
| `scripts/proper_e2e.py` | 7-stage live verifier (extraction → validation → hash → pipeline → dual indexing → retrieval compare → live answers) with artifacts |
| `tests/` | 29 tests collected; offline subset 16 pass; live/E2E require keys + real PDFs |

**POC bugs/notes (read, not run live in this audit):**

1. **Double-embed on index** (`ingest.py:220-232`): `embed_batch` is called manually, then `VectorStoreIndex(nodes, embed_model=GeminiEmbedding())` embeds the same `TextNode`s *again* (nodes carry no precomputed embedding). Every vector-mode index costs 2× Gemini calls. Workaround: pass `nodes` with `embedding=` prefilled, or skip the `VectorStoreIndex` artifact (retrieval only reads `chunks.json` anyway — the LI artifact is **write-only**).
2. **`rerank_fused` key normalization inconsistency** (`retriever.py:320-321`): `key = (… ).lower()` is immediately overwritten by the non-lowered version, while `rrf_fuse` dedups on the lowered key. Minor: dedup can miss case-variant duplicates across the two stages.
3. **Threading:** `embed_text` spawns a `ThreadPoolExecutor` per call when inside a running loop — safe but wasteful vs a shared executor.
4. **No pgvector adapter.** `chunks.json` is the store; pgvector is bypassed entirely. Deploying the POC as-is would *remove* the `paper_chunks` table from the hot path (not a bug per se, but it is a different storage story — see §12).

## 7. Benchmark methodology

- **Baseline A:** `backend/app/services/rag.py` (`rag_service.index_paper` + `retrieve_chunks`).
- **Experiment B:** `llamaindex/src` (`ingest.index_paper` + `retriever.retrieve`), same `full_text`, same `content_hash`.
- **Held constant:** documents, extraction output, chunking (600/100), Gemini model/dim where keyed, similarity metric (cosine), top-k=5, LLM + prompt, threshold 0.25.
- **Artifacts read (not re-run):** `llamaindex/data/verification/pytest-2/{transformer,qwen3vl,qwen}/rag_compare.json` + `summary.json` — 16 frozen questions on real PDFs via `scripts/proper_e2e.py --tier 2`.
- **Mode caveat:** `transformer/result.json` records `"mode": "bm25"` — the Tier-2 run executed **without** `GOOGLE_EMBEDDING_KEY`. Both systems therefore compared **lexical-only** retrieval. Dense/hybrid deltas are **unmeasured**.
- **Metrics:** `recall@5 = 1.0 if any expected substring in top-5 section+content`, `MRR = 1/rank`, else 0. No `nDCG@5`, no `precision@K`, no answer-correctness scoring, no latency/cost.

## 8. Benchmark results (from artifacts)

16 questions (6 Transformer + 5 Qwen3-VL + 5 Qwen):

| Paper | Backend recall@5 | POC recall@5 | Backend MRR | POC MRR |
|---|---|---|---|---|
| Transformer | 1.000 | 1.000 | 0.569 | 0.597 |
| Qwen3-VL | 1.000 | 1.000 | 0.900 | 0.900 |
| Qwen | 0.800 | 0.800 | 0.800 | 0.800 |
| **Weighted (n=16)** | **0.938** | **0.938** | **0.744** | **0.755** |

- **Tied recall.** The expected-case set is small and lexical-friendly; 15/16 questions are answerable by BM25 alone.
- **MRR +0.011** for the POC, explained by: (1) fixed stemming, (2) RRF preferring vector+BM25 agreement on the Transformer BLEU question (MRR 0.333→0.500). Both are portable without LlamaIndex.
- **Both systems fail** "How many tokens was Qwen pretrained on?" (recall 0.0) — numeric tokens are invisible to both BM25 implementations. This is a shared ceiling, not a differentiator.
- **Live answers** (`answers.json`, Transformer): 6/6 grounded (`recall@5=1.0`, substring check on answer or citations), answer lengths 341–1068 chars, 5 citations each.
- **Not measured:** dense retrieval (was BM25-only), hybrid fusion quality, p50/p95 latency, API cost, `nDCG@5`, `precision@K`, citation precision/entailment, abstention quality, reliability under provider/DB failure.

**Separate experiments requested (§6 of task) — status:**

| Experiment | Status |
|---|---|
| Improved chunking | Not isolated — both use 600/100 parity (by design). |
| Hybrid retrieval (RRF) | **Partially measured** via `rrf_fuse`+`rerank_fused` in POC, BM25-only mode. No dense-hybrid comparison. |
| Reranking (cross-encoder) | Not evaluated. POC rerank is a linear blend, not a model. |
| Query transformations | Not evaluated. |

Do **not** combine these into one deltas table — §13 keeps them separate.

## 9. Evaluation dataset

`llamaindex/tests/fixtures/eval_dataset.json` (created by this audit, 60 rows) covers: conceptual, exact-factual, numerical, equation-related, architecture, terminology, multi-section, conversational follow-up, unanswerable, out-of-scope, adversarial/prompt-injection. Each row: `{id, paper_key, paper_id, title, question, type, expect_substrings, expect_section_hint, abstain, origin}` — where `origin: "proper_cases"` = expect substrings taken from the repo's own frozen cases (not invented); `origin: "drafted_for_review"` = needs human/ground-truth verification against the actual PDF before use as a gate.

**To complete to 50–100 gold rows:** verify each `drafted_for_review` row against `papers/1512.03385.pdf` (ResNet, present locally: 12 pp) and `papers/2507.20534.pdf`; repeat for Qwen/DeepSeek from `testing/papers/` (currently gitignored/absent — restore or re-download). Target: ≥80 rows with `origin: "proper_cases"`-grade verification.

## 10. Edge cases investigated

| Case | Backend behaviour | POC behaviour | Verdict |
|---|---|---|---|
| Empty / very short doc | `chunk_paper_text` → 0 chunks → `index_paper` returns 0; `answer_question` → `indexing_in_progress` | Same | OK |
| Very long paper (≤60 pp cap) | ~300 chunks; per-chunk embed ×300; sync LLM blocks loop | Same + double-embed in vector mode | POC worse; backport `to_thread` |
| Scanned PDF | Mistral OCR → vision fallback (`text_extraction.py` uncommitted change) | N/A (ingest takes `full_text`) | Keep extraction in backend |
| Equations/tables | Flattened; BM25 blind; vector path may carry them | Same | Both weak; numeric questions need vector path or a number-aware tokenizer |
| Duplicate uploads | `content_hash` same → `existing >= len` skip; `ON CONFLICT DO NOTHING` | `chunks.json` exists → skip | OK |
| Same paper, different formatting | Hash differs → re-index | Same | Expected |
| Failed embedding / provider outage | Any `None` → pgvector skipped, BM25 active | Any `None` → BM25-only persist | Both degrade correctly |
| Partial indexing | `existing < len` → re-insert with `ON CONFLICT DO NOTHING`; **stale chunks never deleted** | Persisted entry wins if `≥` chunks | Backend stale-chunk bug is real |
| DB unavailable | `get_db_optional` → `None` → in-memory paths | N/A (no DB) | Backend degrades; POC has no DB path |
| Concurrent uploads | Race on in-memory store; pgvector unique constraint saves DB | File lock `_LOCK`; atomic `os.replace` | POC file-write is safer; DB path needs a lock/doc |
| Multiple Uvicorn workers | Cross-worker cache miss → `indexing_in_progress` loops | Same (process-local `_INDEXES`) unless `chunks.json` reloads — it **does** reload from disk in `get_index_entry` | POC strictly better here; production should be DB-only |
| Worker restart | In-memory lost; pgvector survives | `chunks.json` survives | POC better; DB-only fix preferred |
| Stale cache after model change | Backend: stale (no model in key) | POC: invalidates on model mismatch | Port POC's check |
| Missing metadata / wrong dim | `Vector(3072)` insert fails or poisons | Checks `len(embeddings) == len(chunks)` | Port validation |
| Chunker change | Stale chunks persist forever | Same (no version) | **Both need `chunker_version`** |
| Prompt injection in passages | Delimited `BEGIN/END PAPER CONTEXT` + "untrusted" labels | Identical prompt | Good — do not weaken |
| Unanswerable / OOS | Refusal with `indexing_in_progress` false branch | Same | Good |
| Long histories | Per-turn 4000-char cap, **no turn cap** → 413-shrink eats grounding | Same | Fix by turn-cap in both |

## 11. Performance & cost implications

- **Why baseline/poc tie:** chunking, embeddings, prompt, and LLM are identical by construction. The only deltas are stemming, fusion, persistence, and async wrappers — all of which are **portable without the framework**.
- **Latency:** No measured p50/p95 exists in either codebase. Backend chat latency is dominated by the **blocking** `groq_client.call` (up to 120 s timeout) and single-shot embedding RTT. Fixing the event-loop block (`asyncio.to_thread`) is the largest safe latency win — it is a 1-line backend change, not a LlamaIndex feature.
- **Cost:** Vector mode doubles embed cost in the POC (manual `embed_batch` + `VectorStoreIndex` re-embed). Backend pays ~67 embed calls/paper. A `batchEmbedContents` migration would cut both — independent of framework choice.
- **Dependencies:** `llama-index-core` pulls `aiohttp, banks, dataclasses-json, dirtyjson, filetype, fsspec, nest-asyncio, networkx, nltk, pillow, platformdirs, pyyaml, tenacity, tiktoken, tinytag, tqdm, wrapt, llama-index-workflows…`. Production `requirements.txt` currently has 18 entries. Adding the framework is a supply-chain and maintenance commitment with no retrieval payoff.

## 12. Production risks & compatibility

1. **`chunks.json` on Railway:** container disk is ephemeral; there is no volume guarantee. Multi-instance deploys would diverge. Postgres (already provisioned with pgvector) is the correct source of truth — see `docker-compose.yml` (`pgvector/pg16`) and `init_db` idempotent table creation.
2. **pgvector extension absent → silent skip.** `init_db` logs a warning and proceeds; `paper_chunks` creation may fail in a loop and still "succeed". If the extension is missing, retrieval quietly degrades to BM25. This is intentional (never take the app down) but must be **alerted**, not just logged.
3. **Version drift.** POC pins `llama-index-core>=0.12,<1.0` — wide range across a fast-moving API. Framework upgrades are a recurring tax.
4. **Storage divergence.** Shipping the POC alongside `paper_chunks` means two vector stores to reconcile, back up, and garbage-collect. The `embedding_model`/`dim` validation in `ingest.py` is good but only covers the file store.
5. **API contract:** both keep `{answer, citations[], indexing_in_progress?}` — verified by `test_contract.py` — so a swap would be frontend-safe. That safety is a reason the POC exists at all, but it does not imply the swap is worth doing.

## 13. Migration complexity

| Option | Effort | Risk |
|---|---|---|
| **A. Keep custom, backport 3 fixes** (stemming, RRF fusion, model-aware cache + `asyncio.to_thread` + `chunker_version`/`embedding_model` columns) | **~1 day, ~150 lines in `rag.py`** | Low. No new deps, no storage migration |
| **B. Adopt POC as the store** (`chunks.json` + LI artifact) | Days–weeks (storage ops, GC, backups, multi-instance story) | Medium-high. Loses pgvector; double-embed cost unless fixed |
| **C. Custom pgvector adapter + LlamaIndex on top** | Adapter (~150 lines) + framework upgrade tax forever | Medium. Value still comes from custom code |
| **D. Full LlamaIndex migration** (nodes, parsers, query engines) | Weeks; re-embed + re-validate prompts | High. Stock response synthesis weakens grounding; chunker change invalidates all vectors |

## 14. Decision table & recommended next steps

| Area | Keep custom | Adopt LlamaIndex | Reason |
|---|---|---|---|
| PDF extraction | ✅ | | Mistral OCR + vision fallback is product-specific; LI readers add nothing |
| Chunking | ✅ | | 600/100 parity is fine; add `page_number` + `chunker_version` |
| Embeddings | ✅ | | Gemini adapter works; port model-aware cache key only |
| Vector storage | ✅ | | `paper_chunks` + pgvector is already right; add HNSW index |
| Dense retrieval | ✅ | | Same cosine; `<=>` SQL is 12 lines |
| Hybrid retrieval | ✅ (port concept) | | RRF fusion + `0.6/0.4` rerank from POC, implemented in ~70 lines of `rag.py` |
| Reranking | ✅ | | Linear blend now; evaluate a cross-encoder later against the 60-row dataset |
| Query transformation | ✅ | | Anaphoric expansion only; HyDE costs latency for unproven gain |
| Response generation | ✅ | | Custom numbers-rule prompt must stay; stock query engines would weaken it |
| Observability | | ✅ (optional, low priority) | Instrumentation spans are the only LI piece with a plausible standalone benefit |

### Next steps (in order)

1. **Backport** to `backend/app/services/rag.py` (no new deps): sequential stemming (`_normalize_token`), RRF hybrid fusion replacing the exclusive fallback, model-aware embedding cache key, `asyncio.to_thread(groq_client.call)`, per-turn history cap + grounding-preserving 413 trim.
2. **Add** `embedding_model`, `embedding_dim`, `chunker_version`, `page_number` columns to `PaperChunk`; `CREATE INDEX … USING hnsw (embedding vector_cosine_ops)`; invalidate/re-index flow keyed on `chunker_version`.
3. **Verify** the 60-row dataset (`llamaindex/tests/fixtures/eval_dataset.json`) against local PDFs; promote drafted rows to gold; then run the fair A/B (§7) in **keyed** (dense+hybrid) mode with `nDCG@5`, `precision@5`, p50/p95, cost — the current BM25-only run cannot discriminate.
4. **Re-run** `cd backend; python -m pytest tests -q` and `cd llamaindex; python -m pytest tests/test_chunking.py tests/test_retriever.py tests/test_contract.py tests/test_chat_mock.py -q` after the backport. Reproduce live run: `cd llamaindex; python scripts/proper_e2e.py --tier 2`.
5. **Revisit** LlamaIndex only if/when a cross-encoder reranker or HyDE experiment beats the backported baseline on the gold set — then adopt as a postprocessor, not as the platform.

**Final recommendation: Keep the existing custom RAG. Do not adopt LlamaIndex as a framework. Port the POC's custom retrieval ideas (stemming fix, RRF fusion, persistence-aware validation, async wrappers) directly into `backend/app/services/rag.py`.**

## Appendix: key file references

- `backend/app/services/rag.py:41-70` embeddings · `:73-128` chunking · `:173-245` BM25 · `:255-338` index · `:340-424` retrieve · `:426-569` answer (prompt+LLM)
- `backend/app/models/orm.py:23-38` `PaperChunk` · `backend/app/core/database.py:16-59` sessions · `backend/app/services/llm.py:69-149` blocking client · `backend/app/services/pipeline.py:120-129` fire-and-forget index · `backend/app/api/endpoints/chat.py:62-133` chat endpoint
- `llamaindex/src/{config,embeddings,chunking,ingest,retriever,chat,llm_client,eval}.py` · `llamaindex/scripts/proper_e2e.py` · `llamaindex/data/verification/pytest-2/*/rag_compare.json` · `llamaindex/tests/fixtures/proper_cases.json`
