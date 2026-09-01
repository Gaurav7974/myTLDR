# AGENTS.md

Project-specific instructions for AI agents working on this codebase.

## Project Overview

AI Research Paper Platform — users upload a paper (PDF/arXiv/URL) and get:
1. A plain-language explanation at chosen difficulty (beginner/intermediate/expert)
2. An auto-generated architecture diagram extracted from the paper

**Core principle:** If the system can't confidently verify the diagram, show text-only — never a wrong diagram.

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** Next.js
- **LLM:** Groq (structured/JSON output mode)
- **DB:** Postgres (SQLite for local dev)
- **Graph layout:** dagre / ELK.js
- **Diagram render:** React Flow or raw SVG

## Project Structure

```
backend/
  app/
    api/endpoints/    # FastAPI routes (chat, stream, submit)
    core/             # Security, config
    models/           # DB models, IR models, graph models
    services/         # LLM, RAG, pipeline, layout, graph, cache
    main.py           # FastAPI app entry
    config.py         # Settings
    database.py       # DB connection
  venv/               # Python virtual environment
  requirements.txt
  run.py

frontend/
  # Next.js app (mock data, backend integration pending)
```

## Key Conventions

### Backend
- All LLM calls go through `services/llm.py`
- Structured output via Groq's JSON mode
- PDF text extraction: PyMuPDF for text pages, Mistral OCR for image pages (`services/text_extraction.py`)
- RAG pipeline in `services/rag.py`
- Architecture graph extraction in `services/graph_extraction.py`; figure-based extraction in `services/figure_extraction.py`
- ArchitectureIR extraction/validation in `services/ir_extraction.py` + `ir_validator.py`
- 3D forward-pass stage extraction in `services/forward_pass.py`
- Layout: React Flow positions in `services/flow_layout.py`; 3D SceneGraph in `services/scene_layout.py`
- Orchestration in `services/pipeline.py`; caching in `services/cache.py`

### Frontend
- Next.js App Router
- Parsed Text tab renders backend `parsed_text` (per-page, OCR images inlined); mock data is fallback only

## What's Done vs What's Next

### Done
- Backend implemented and import-verified
- Frontend connected to backend via SSE `/api/stream`

### Next
1. End-to-end test with real papers and live API keys

## Environment Variables

Required in `backend/.env`:
- Groq API key
- Database URL (Postgres or SQLite path)
- OCR API key (Mistral, powers OCR + vision + text LLM)

## Important Notes

- Do NOT commit `.env` files
- Do NOT commit `venv/` or `__pycache__/`
- All `*.md` planning docs are gitignored except `README.md`
- Do NOT mention phases or plans in public-facing docs
