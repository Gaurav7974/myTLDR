# AI Research Paper Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat&logo=nextdotjs&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-enabled-3C3C3C?style=flat&logoColor=white)
![Status](https://img.shields.io/badge/status-active_development-FFA500?style=flat)

**Upload a paper. Get a plain-language explanation and an auto-generated architecture diagram.**

[Features](#features) • [How It Works](#how-it-works) • [Quick Start](#quick-start) • [API](#api)

</div>

---

## Features

- **Plain-language explanations** at 3 difficulty levels (beginner, intermediate, expert)
- **Architecture diagram extraction** — automatically builds a visual graph from any paper's methodology
- **3D model deconstruction** — sequential forward-pass pipeline visualization (Brendan Bycroft style)
- **RAG-powered chat** — ask questions about the paper using pgvector semantic search
- **Multimodal extraction** — handles PDFs, arXiv URLs, and direct PDF links
- **Content-addressable caching** — repeat requests on the same paper are instant
- **Trust-first design** — if the system can't verify the diagram, it shows text-only instead of a wrong diagram

---

## How It Works

```
PDF / arXiv / URL
       │
       ▼
┌─────────────────┐
│  Text Extraction │ ← PyMuPDF + Mistral OCR fallback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Content Hash    │ ← SHA256 cache key
└────────┬────────┘
         │
    ┌────┴────┐
    │ Cache?   │──HIT──→ Return cached result
    └────┬────┘
         │ MISS
         ▼
┌─────────────────┐
│  Explanation     │ ← Mistral/Groq, difficulty-parameterized
│  Generation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Graph           │ ← Structured JSON extraction (schema-validated)
│  Extraction      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Verification    │ ← Second LLM call: does the graph misrepresent the text?
│  Pass            │    Yes → discard diagram, show text-only
└────────┬────────┘    No  → proceed to render
         │
         ▼
┌─────────────────┐
│  Layout + Render │ ← BFS layered layout → React Flow / 3D scene
└─────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (for PostgreSQL + pgvector)

### 1. Clone & Setup Backend

```bash
# Copy environment config
cp backend/.env.example backend/.env
# Edit .env with your API keys

# Start database
cd backend && docker compose up -d

# Install deps & run
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py
```

Backend runs at `http://localhost:8000`

### 2. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

### 3. Required API Keys

| Service | Env Variable | Purpose |
|---------|-------------|---------|
| Mistral | `MISTRAL_API_KEY` | LLM extraction + OCR + vision |
| Groq | `GROQ_API_KEY` | Optional text fallback |

---

## API

### `POST /api/stream`

Upload a paper and receive SSE progress events.

**Input:** multipart form
- `source`: `"upload" \| "url" \| "arxiv"`
- `difficulty_level`: `"beginner" \| "intermediate" \| "expert"`
- `file` or `url`

**Output:** SSE stream
```
data: {"stage": "extracting", "message": "Reading PDF..."}
data: {"stage": "explaining", "message": "Generating explanation..."}
data: {"stage": "diagramming", "message": "Building architecture graph..."}
data: {"stage": "done", "explanation": "...", "graph": {...}, "diagram_available": true}
data: [DONE]
```

### `POST /api/chat`

Ask questions about a processed paper (RAG-powered).

### `POST /api/submit`

Upload a paper for processing.

---

## Project Structure

```
backend/
├── app/
│   ├── api/endpoints/    # FastAPI routes
│   ├── core/             # Config, database, security
│   ├── models/           # Pydantic schemas, SQLAlchemy ORM
│   ├── services/         # LLM, extraction, RAG, caching, layout
│   └── main.py           # FastAPI application entry
├── docker-compose.yml    # PostgreSQL + pgvector
└── requirements.txt

frontend/
└── src/
    ├── app/              # Next.js App Router
    ├── components/       # React UI components
    ├── lib/              # API client, layout engine
    └── types/            # TypeScript interfaces
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Pydantic + SQLAlchemy (async) |
| Frontend | Next.js 16 + React 19 + Tailwind |
| LLM | Mistral (primary) + Groq (fallback) |
| OCR | Mistral OCR |
| Database | PostgreSQL 16 + pgvector |
| Diagrams | React Flow + custom 3D scene graph |
| Caching | Content-hash Postgres lookups |

---

## License

MIT
