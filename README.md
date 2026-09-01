# AI Research Paper Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat&logo=nextdotjs&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-enabled-3C3C3C?style=flat&logoColor=white)
![Tests](https://img.shields.io/badge/tests-155_passing-brightgreen?style=flat&logo=checkmarx&logoColor=white)

**Upload a paper. Get a plain-language explanation and an interactive, figure-faithful architecture diagram.**

[Features](#features) • [How It Works](#how-it-works) • [Quick Start](#quick-start) • [Testing](#testing) • [API](#api) • [Tech Stack](#tech-stack)

</div>

---

## Features

- **Plain-language explanations** at 3 difficulty levels (beginner, intermediate, expert)
- **Figure-faithful architecture diagrams** — reconstructs the paper's actual architecture figure with extracted coordinates, groups, arrows, and styles
- **Trust-first design** — if the system can't verify the diagram against the source figure, it shows text-only instead of a wrong diagram
- **3D model deconstruction** — sequential forward-pass pipeline visualization (Brendan Bycroft style)
- **RAG-powered chat** — ask questions about the paper using pgvector semantic search
- **Multimodal extraction** — handles PDFs, arXiv URLs, and direct PDF links
- **Content-addressable caching** — repeat requests on the same paper are instant

---

## How It Works

```
PDF / arXiv / URL
       │
       ▼
┌──────────────────┐
│  Text Extraction  │ ← PyMuPDF (text pages) + Mistral OCR (image pages)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Content Hash     │ ← SHA256 cache key
└────────┬─────────┘
         │
    ┌────┴────┐
    │ Cache?   │──HIT──→ Return cached result
    └────┬────┘
         │ MISS
         ▼
┌──────────────────┐     ┌───────────────────┐
│  Vision Graph     │────▶│  Figure-Faithful   │
│  Extraction       │     │  Verification       │
│  (from figures)   │     │  (vs source image)  │
└────────┬─────────┘     └─────────┬─────────┘
         │                          │
         │              ┌───────────┴───────────┐
         │              │ valid + confident?     │
         │              │ No  → text-only        │
         │              │ Yes → merge with text  │
         │              └───────────┬───────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐     ┌───────────────────┐
│  Text Graph       │     │  Merged Graph      │
│  Enrichment       │────▶│  (geometry +       │
│  (semantics)      │     │   semantics)       │
└──────────────────┘     └─────────┬───────────┘
                                    │
                                    ▼
                         ┌───────────────────┐
                         │  Render            │
                         │  - Figure coords   │
                         │  - Groups + arrows │
                         │  - Styles + math   │
                         │  - Inspector panel │
                         └───────────────────┘
```

### Trust-First Verification

A reconstructed diagram is only shown when **all four gates** pass:

1. **Vision extraction** produces valid geometry with normalized bounding boxes
2. **Faithfulness verification** confirms every node/edge matches the source figure (no hallucinations)
3. **Confidence threshold** is met (0.65+)
4. **Source attribution** is confirmed (vision or merged source)

If any gate fails, the system falls back to text-only or auto-layout — never a wrong polished diagram.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (for local PostgreSQL + pgvector)

### 1. Clone & Setup Backend

```bash
git clone https://github.com/Gaurav7974/myTLDR.git
cd myTLDR

# Copy environment config
cp backend/.env.example backend/.env
# Edit .env with your API keys (see below)

# Start database
cd backend && docker compose up -d

# Install deps & run
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
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
| Mistral | `MISTRAL_API_KEY` | LLM extraction + OCR + vision + figure verification |
| Groq | `GROQ_API_KEY` | Text extraction fallback (optional) |

---

## Testing

### Backend (72 tests)

```bash
cd backend
pytest tests/ -v
```

Covers: schema validation, vision extraction parse success/failure, figure-faithfulness verification, pipeline merge/fallback decision matrix, cache version policy, text enrichment.

### Frontend (83 tests)

```bash
cd frontend
npm test
```

Covers: trust-first gate functions, layout helpers (figure coordinates, auto-layout BFS, edge style resolution, group positioning, SVG path generation), API payload transform (rich shape detection, legacy shape conversion).

### Type Check & Build

```bash
cd frontend
npx tsc --noEmit   # type check
npm run build      # production build
```

---

## API

### `POST /api/stream`

Upload a paper and receive SSE progress events.

**Input:** multipart form
- `source`: `"upload" | "url" | "arxiv"`
- `difficulty_level`: `"beginner" | "intermediate" | "expert"`
- `file` or `url`

**Output:** SSE stream
```
data: {"stage": "extracting", "message": "Reading PDF..."}
data: {"stage": "explaining", "message": "Generating explanation..."}
data: {"stage": "diagramming", "message": "Building architecture graph..."}
data: {"stage": "done", "explanation": "...", "graph": {...}, "diagram_available": true}
data: [DONE]
```

### Graph Payload Shape

The `graph` field in the `done` event is a rich `ArchitectureGraph`:

```jsonc
{
  "version": "2.0",
  "source "vision",  // "vision" | "merged" | "text" | "legacy"
  "layout_mode": "figure",  // "figure" | "auto" | "legacy"
  "confidence": 0.88,
  "verification": {
    "valid": true,
    "confidence": 0.88,
    "hallucinations": [],
    "missing_elements": []
  },
  "groups": [
    { "id": "encoder", "label": "Encoder", "bbox": { "x": 0.05, "y": 0.1, "w": 0.4, "h": 0.3 }, "color": "#93c5fd" }
  ],
  "nodes": [
    {
      "id": "attn", "label": "Self-Attention", "type": "attention",
      "bbox": { "x": 0.1, "y": 0.2, "w": 0.2, "h": 0.1 },
      "group_id": "encoder",
      "description": "Scaled dot-product attention.",
      "math_formula": "softmax(QK^T / sqrt(d_k)) V",
      "paper_term": "self-attention",
      "provenance": { "source": "vision", "figure_id": "fig:1", "figure_page": 3 },
      "confidence": 0.9
    }
  ],
  "edges": [
    {
      "source": "attn", "target": "ffn", "type": "data_flow",
      "path_points": [{ "x": 0.18, "y": 0.25 }, { "x": 0.3, "y": 0.25 }],
      "style": { "stroke": "#3b82f6", "name": "solid" }
    }
  ]
}
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
│   ├── api/endpoints/    # FastAPI routes (stream, chat, submit)
│   ├── core/             # Config, database, security
│   ├── models/           # Pydantic schemas (graph, IR), SQLAlchemy ORM
│   ├── services/         # LLM, extraction, RAG, caching, layout, graph merge
│   └── main.py           # FastAPI application entry
├── tests/                # Pytest test suite (72 tests)
│   ├── conftest.py
│   ├── test_graph_schema.py
│   ├── test_figure_extraction.py
│   ├── test_pipeline.py
│   ├── test_cache.py
│   └── test_text_extraction.py
├── docker-compose.yml    # PostgreSQL + pgvector
└── requirements.txt

frontend/
└── src/
    ├── app/              # Next.js App Router
    ├── components/       # React UI components
    │   ├── ArchitectureDiagram.tsx  # Figure-faithful SVG renderer
    │   └── tabs/         # Tab views (3D visualizer, chat, etc.)
    ├── lib/              # API client, layout engine
    │   ├── api.test.ts           # API transform tests
    │   └── architectureDiagramLayout.test.ts  # Layout tests
    └── types/            # TypeScript interfaces
        ├── architecture_graph.ts  # Graph types + gate functions
        └── architecture_graph.test.ts  # Gate function tests
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Pydantic v2 + SQLAlchemy (async) |
| Frontend | Next.js 16 + React 19 + Tailwind 4 |
| LLM | Groq (text extraction) + Mistral (vision + OCR + fallback) |
| OCR | Mistral OCR |
| Database | PostgreSQL 16 + pgvector |
| Diagrams | Custom SVG renderer (figure coordinates) + 3D scene graph |
| Caching | Content-hash Postgres lookups |
| Testing | pytest (backend) + vitest (frontend) |
| Deploy | Railway (backend) + Vercel (frontend) |

---

## License

MIT
