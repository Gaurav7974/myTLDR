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
- RAG pipeline in `services/rag_service.py`
- Graph extraction in `services/graph_service.py` + `graph_validator.py`
- Layout computed in `services/layout.py` + `layout_engine.py`

### Frontend
- Next.js App Router
- Currently uses mock data — backend integration is the next step

## What's Done vs What's Next

### Done
- Backend code written (NOT yet verified — no `pip install` run)
- Frontend scaffold with mock data

### Next
1. Install backend deps, verify imports
2. Connect frontend to backend
3. End-to-end test with real papers

## Environment Variables

Required in `backend/.env`:
- Groq API key
- Database URL (Postgres or SQLite path)
- OCR API key (external service)

## Testing

- `backend/test_security.py` — security tests

## Important Notes

- Do NOT commit `.env` files
- Do NOT commit `venv/` or `__pycache__/`
- All `*.md` planning docs are gitignored except `README.md`
- The `flashattention.pdf` in root is a test fixture
- Do NOT mention phases or plans in public-facing docs
