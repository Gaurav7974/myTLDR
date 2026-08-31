"""FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.api import router
from app.core.security import limiter

app = FastAPI(
    title=settings.APP_NAME,
    description="AI Research Paper Analysis & 3D Deconstruction Platform API with pgvector RAG & Security Shield",
    version="2.0.0"
)

# Attach SlowAPI Limiter to app state
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom JSON response when rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Too many requests. Limit: {exc.detail}. Please retry shortly or provide an API Key for higher quotas.",
            "retry_after": getattr(exc, "retry_after", 60)
        }
    )


# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "security": {
            "rate_limiting": "active (SlowAPI)",
            "ssrf_protection": "active",
            "api_key_auth": "enabled"
        }
    }
