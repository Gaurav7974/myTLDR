"""Application configuration."""

import os
from typing import Set
from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)


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
    # Used only by server-side Next.js proxies. Browser traffic carries it,
    # so rate limiting falls back to the forwarded client IP for this key.
    PROXY_API_KEY: str | None = os.getenv("PROXY_API_KEY") or None

    def require_master_key(self) -> str | None:
        configured = self.MASTER_API_KEY or self.API_KEYS or self.PROXY_API_KEY
        if not configured and not self.DEBUG:
            raise RuntimeError("An API key must be set in production (DEBUG=false)")
        return self.MASTER_API_KEY

    # Rate Limiting (SlowAPI format)
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
    RATE_LIMIT_STREAM: str = os.getenv("RATE_LIMIT_STREAM", "10/minute")
    RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "30/minute")
    RATE_LIMIT_SUBMIT: str = os.getenv("RATE_LIMIT_SUBMIT", "20/minute")

    # ── LLM (primary: Gemini; secondary: Mistral; tertiary: Groq) ──
    # Gemini (Google AI Studio, OpenAI-compatible endpoint) is the platform's
    # primary LLM + vision provider. The account keys in .env (VISION_MODEL_API_KEY
    # and/or GOOGLE_EMBEDDING_KEY, "AQ.…") are multimodal and generous; the Mistral
    # accounts configured here are on a hard tier quota (OCR = 0 RPM forever,
    # ministral-8b = 2 RPM/key, small/medium = 0) so they cannot feed this pipeline.
    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        os.getenv("VISION_MODEL_API_KEY", os.getenv("GOOGLE_EMBEDDING_KEY", "")),
    )
    GEMINI_LLM_URL: str = os.getenv(
        "GEMINI_LLM_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # Primary Mistral key (env OCR_KEY) powers OCR + vision + the text LLM.
    # An optional second key (MISTRAL_API_KEY_2) from a second account is rotated
    # round-robin with the primary to halve per-key rate-limit pressure.
    MISTRAL_API_KEY: str = os.getenv("OCR_KEY", "")
    MISTRAL_API_KEY_2: str = os.getenv("MISTRAL_API_KEY_2", "")
    MISTRAL_LLM_URL: str = "https://api.mistral.ai/v1/chat/completions"
    # Default is ministral-8b-latest: multimodal (vision + text), 128k context,
    # 188 req/min quota per key on standard tier. mistral-small-latest is rate-limited to 0.
    MISTRAL_LLM_MODEL: str = os.getenv("MISTRAL_LLM_MODEL", "ministral-8b-latest")

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

    # Vision (architecture-from-image). Defaults to the Gemini path: multimodal
    # with JSON-mode support, uses the same AQ. key as the text path.
    VISION_API_KEY: str = os.getenv("VISION_API_KEY", GEMINI_API_KEY)
    VISION_API_KEY_2: str = os.getenv("VISION_API_KEY_2", "")
    VISION_API_URL: str = os.getenv("VISION_API_URL", GEMINI_LLM_URL)
    VISION_MODEL: str = os.getenv("VISION_MODEL", GEMINI_MODEL)

    # Optional Groq fallback for the TEXT path only (Groq is text-only: no vision/OCR).
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    # llama-3.3-70b-versatile was decommissioned by Groq (404); gpt-oss-120b is the live flagship fallback.
    GROQ_EXTRACT_MODEL: str = os.getenv("GROQ_EXTRACT_MODEL", "openai/gpt-oss-120b")
    GROQ_EVAL_MODEL: str = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-120b")
    GROQ_EXPLAIN_MODEL: str = os.getenv("GROQ_EXPLAIN_MODEL", "openai/gpt-oss-120b")
    LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", GROQ_EXTRACT_MODEL)
    LLM_FALLBACK_ENABLED: bool = (
        os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    )

    # Embedding (Google Gemini)
    GOOGLE_EMBEDDING_KEY: str = os.getenv("GOOGLE_EMBEDDING_KEY", "")
    GOOGLE_EMBEDDING_MODEL: str = os.getenv(
        "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
    )
    EMBEDDING_DIM: int = 3072

    # RAG retrieval — cosine similarity floor; 0.25 is a good default for
    # research-paper embeddings (gemini-embedding-001 / 3072-dim).
    # Lower if you want wider recall; raise to tighten grounding.
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.25"))

    # RAG versioning — bump CHUNKER_VERSION whenever chunking logic or parameters change;
    # stored chunker_version mismatches trigger a re-index in index_paper.
    CHUNKER_VERSION: str = os.getenv("CHUNKER_VERSION", "v1")
    # Hybrid fusion (RRF). Default off until the eval harness proves it helps on the
    # gold dataset; safe to enable per-environment without a code change.
    RAG_HYBRID_ENABLED: bool = (
        os.getenv("RAG_HYBRID_ENABLED", "false").lower() == "true"
    )
    RAG_VECTOR_WEIGHT: float = float(os.getenv("RAG_VECTOR_WEIGHT", "0.6"))
    RAG_BM25_WEIGHT: float = float(os.getenv("RAG_BM25_WEIGHT", "0.4"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/papers"
    )

    # Processing
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_PDF_PAGES: int = (
        60  # Maximum page limit for uploaded/linked papers to prevent OCR starvation
    )
    MAX_SECTION_CHARS: int = 18000  # legacy flat cap — retained for compat, use SECTION_CHAR_CEILING for new paths
    # LLM-based architecture section detection: how much of the paper to send to the locator
    ARCH_SECTION_LLM_MAX_CHARS: int = 80000
    # Dynamic ceiling for any section fed to LLM — safely accommodates up to 60-page papers
    # while keeping JSON structured outputs reliable.
    SECTION_CHAR_CEILING: int = 60000
    LLM_MAX_RETRIES: int = 4
    LLM_RETRY_BACKOFF_BASE: int = 3  # seconds

    @property
    def max_section_ceiling(self) -> int:
        """Ceiling for dynamic section sizing — prefer SECTION_CHAR_CEILING, fallback to legacy."""
        return getattr(self, "SECTION_CHAR_CEILING", 60000)

    def dynamic_section_limit(self, detected_len: int) -> int:
        """Size limit to actual section length up to ceiling (Fix #3)."""
        if detected_len <= 0:
            return self.max_section_ceiling
        return min(detected_len, self.max_section_ceiling)

    # arXiv
    ARXIV_RATE_LIMIT_SECONDS: int = 3
    ARXIV_USER_AGENT: str = "PaperExplainer/1.0"


settings = Settings()
