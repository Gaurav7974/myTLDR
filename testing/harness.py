"""
Harness module to configure environment and import backend services
safely from inside the testing/ folder without altering backend code.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure root and backend are on sys.path
TESTING_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTING_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

# Load environment from backend/.env
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Verify key imports from backend
from app.core.config import settings
from app.services.text_extraction import text_extractor, Figure, PageText, ExtractionResult
from app.services.explanation import explanation_service
from app.services.ir_extraction import ir_extractor
from app.services.graph_extraction import graph_extractor
from app.services.rag import rag_service, chunk_paper_text, bm25_keyword_search
from app.services.paper_validator import paper_validator

# Ensure subdirectories exist
PAPERS_DIR = TESTING_DIR / "papers"
RESULTS_DIR = TESTING_DIR / "results"
REPORTS_DIR = TESTING_DIR / "reports"

for d in (PAPERS_DIR, RESULTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"[HARNESS] Initialized. Testing root: {TESTING_DIR}")
print(f"[HARNESS] Settings loaded: MISTRAL_KEY={'SET' if settings.MISTRAL_API_KEY else 'UNSET'}, "
      f"GROQ_KEY={'SET' if settings.GROQ_API_KEY else 'UNSET'}, "
      f"GOOGLE_KEY={'SET' if settings.GOOGLE_EMBEDDING_KEY else 'UNSET'}")
