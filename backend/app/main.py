"""FastAPI application."""
import uuid, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.logger import setup_logging, request_id_var, get_logger
from app.api import router
from app.core.security import limiter

setup_logging()
log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create extension + tables (init_db is idempotent). Never let DB
    # problems take the app down — endpoints degrade without it.
    try:
        from app.core.database import init_db
        await init_db()
        log.info("database initialized")
    except Exception as e:
        log.warning(f"database init skipped (degraded mode): {e}")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="AI Research Paper Analysis & 3D Deconstruction Platform API with pgvector RAG & Security Shield",
    version="2.0.0",
    lifespan=lifespan,
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


# ── HTTP middleware (request-id / logging) ───────────────────────────────────
# IMPORTANT: @app.middleware("http") decorators are prepended to the stack, so
# the LAST decorator registered becomes the OUTERMOST wrapper. We register
# add_request_id BEFORE CORSMiddleware so that CORS ends up outermost and can
# always set Access-Control-Allow-Origin headers — even when inner middleware
# raises an exception.
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    start = time.time()
    try:
        resp = await call_next(request)
        log.info(f"{request.method} {request.url.path} → {resp.status_code} {int((time.time()-start)*1000)}ms", extra={"status": resp.status_code, "duration_ms": int((time.time()-start)*1000)})
        resp.headers["X-Request-ID"] = rid
        return resp
    except Exception as e:
        log.error(f"Unhandled {request.url.path}: {e}", exc_info=True)
        raise

# ── CORS ─────────────────────────────────────────────────────────────────────
# Registered AFTER the HTTP middleware so it sits OUTERMOST in the stack and
# always runs first on both preflight OPTIONS and regular requests.
# allow_origin_regex covers any *.vercel.app preview/prod URL.
# Explicit origins are the authoritative list; CORS_ORIGINS env var can extend
# it (comma-separated). Wildcard "*" is stripped — incompatible with credentials.
import os as _os
_cors_origins = [
    o.strip()
    for o in _os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://my-tldr.vercel.app",
    ).split(",")
    if o.strip() and o.strip() != "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    log.info("health check")
    return {"status": "ok", "app": settings.APP_NAME, "security": {"rate_limiting": "active (SlowAPI)", "ssrf_protection": "active", "api_key_auth": "enabled"}}
