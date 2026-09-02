"""Application configuration."""

import os
from typing import Set
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))


class Settings:
    """App settings loaded from environment."""
    
    # API
    APP_NAME: str = "AI Research Paper Platform"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security & API Keys — fail closed in production
    MASTER_API_KEY: str | None = os.getenv("MASTER_API_KEY") or None
    API_KEYS: Set[str] = {
        k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
    }

    def require_master_key(self) -> str | None:
        if not self.MASTER_API_KEY and not self.DEBUG:
            raise RuntimeError("MASTER_API_KEY must be set in production (DEBUG=false)")
        return self.MASTER_API_KEY
    
    # Rate Limiting (SlowAPI format)
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
    RATE_LIMIT_STREAM: str = os.getenv("RATE_LIMIT_STREAM", "10/minute")
    RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "30/minute")
    RATE_LIMIT_SUBMIT: str = os.getenv("RATE_LIMIT_SUBMIT", "20/minute")
    
    # ── LLM (primary: Mistral; optional fallback: Groq) ──
    # A single Mistral key (env OCR_KEY) powers OCR + vision + the text LLM.
    MISTRAL_API_KEY: str = os.getenv("OCR_KEY", "")
    MISTRAL_LLM_URL: str = "https://api.mistral.ai/v1/chat/completions"
    # Default is mistral-small-latest: it is the strongest multimodal Mistral model
    # available on the standard (non-enterprise) tier. mistral-large-latest is gated to
    # a higher subscription tier (returns 403). Override via MISTRAL_LLM_MODEL if you upgrade.
    MISTRAL_LLM_MODEL: str = os.getenv("MISTRAL_LLM_MODEL", "mistral-small-latest")

    # Mistral OCR
    MISTRAL_OCR_URL: str = "https://api.mistral.ai/v1/ocr"
    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"
    # Pages with fewer text chars than this are treated as image/scanned pages and OCR'd
    OCR_PAGE_TEXT_THRESHOLD: int = 200
    # Max individual page OCR calls per document (whole-doc OCR is used beyond this)
    OCR_MAX_PAGE_CALLS: int = 8
    # Inline images sent to the frontend in parsed_text (payload safety caps)
    MAX_INLINE_IMAGES: int = 20
    MAX_INLINE_IMAGE_KB: int = 2048

    # Vision (architecture-from-image). mistral-small-latest is multimodal on this tier
    # (mistral-large-latest is tier-gated). Override VISION_MODEL to a larger model if upgraded.
    VISION_API_KEY: str = os.getenv("VISION_API_KEY", os.getenv("OCR_KEY", ""))
    VISION_API_URL: str = os.getenv("VISION_API_URL", "https://api.mistral.ai/v1/chat/completions")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "mistral-small-latest")

    # Optional Groq fallback for the TEXT path only (Groq is text-only: no vision/OCR).
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_EXTRACT_MODEL: str = os.getenv("GROQ_EXTRACT_MODEL", "llama-3.3-70b-versatile")
    GROQ_EVAL_MODEL: str = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")
    GROQ_EXPLAIN_MODEL: str = os.getenv("GROQ_EXPLAIN_MODEL", "llama-3.3-70b-versatile")
    LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", GROQ_EXTRACT_MODEL)
    LLM_FALLBACK_ENABLED: bool = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    
    # Embedding (Google Gemini)
    GOOGLE_EMBEDDING_KEY: str = os.getenv("GOOGLE_EMBEDDING_KEY", "")
    GOOGLE_EMBEDDING_MODEL: str = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIM: int = 3072
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/papers")
    
    # Processing
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_SECTION_CHARS: int = 18000
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BACKOFF_BASE: int = 3  # seconds
    
    # arXiv
    ARXIV_RATE_LIMIT_SECONDS: int = 3
    ARXIV_USER_AGENT: str = "PaperExplainer/1.0"


settings = Settings()
