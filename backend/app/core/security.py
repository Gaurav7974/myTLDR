"""Security module: Rate limiting, API key authentication, and SSRF prevention."""

import re
import socket
import ipaddress
from urllib.parse import urlparse
from typing import Optional
from fastapi import Request, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings

# API Key headers
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def get_rate_limit_key(request: Request) -> str:
    """
    Extract rate limiting key:
    If a valid API key is present, rate limit per API key.
    Otherwise, rate limit per client IP (respecting X-Forwarded-For from Vercel).
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()

    if api_key and (api_key == settings.MASTER_API_KEY or api_key in settings.API_KEYS):
        return f"apikey:{api_key}"

    # Prefer real client IP from Vercel proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    return get_remote_address(request)


# Global SlowAPI Limiter
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[settings.RATE_LIMIT_DEFAULT]
)


def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth)
) -> Optional[str]:
    """
    Verify API key for programmatic access.
    Accepts X-API-Key header or Authorization: Bearer <key>.
    """
    key = header_key or (bearer_creds.credentials if bearer_creds else None)
    if not key:
        return None

    if key == settings.MASTER_API_KEY or key in settings.API_KEYS:
        return key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired API Key"
    )


# Canonical arXiv regex patterns
ARXIV_MODERN_REGEX = re.compile(r'^\d{4}\.\d{4,5}(?:v\d+)?$')
ARXIV_LEGACY_REGEX = re.compile(r'^[a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}(?:v\d+)?$')


def sanitize_arxiv_id(raw_id: str) -> str:
    """
    Sanitize and validate an arXiv ID.
    Blocks directory traversal, injection characters, and invalid formats.
    """
    if not raw_id or not isinstance(raw_id, str):
        raise HTTPException(status_code=400, detail="arXiv ID cannot be empty")

    cleaned = raw_id.strip()

    # Strip URL prefixes if full link was provided
    if "arxiv.org" in cleaned:
        # Extract ID from https://arxiv.org/abs/1706.03762 or /pdf/1706.03762.pdf
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([a-zA-Z0-9\.\-/]+?)(?:\.pdf)?(?:[?#]|$)', cleaned)
        if match:
            cleaned = match.group(1)

    # Strip leading "arxiv:" case-insensitively
    if cleaned.lower().startswith("arxiv:"):
        cleaned = cleaned[6:].strip()

    # Strip trailing .pdf
    if cleaned.lower().endswith(".pdf"):
        cleaned = cleaned[:-4]

    # Validate against canonical formats
    if ARXIV_MODERN_REGEX.match(cleaned) or ARXIV_LEGACY_REGEX.match(cleaned):
        return cleaned

    raise HTTPException(
        status_code=400,
        detail=f"Invalid arXiv ID format: '{raw_id}'. Expected format e.g. '1706.03762' or 'hep-th/9901001'."
    )


def validate_and_sanitize_url(url: str) -> str:
    """
    Validate URL and enforce SSRF defense:
    - Restrict scheme to HTTP / HTTPS
    - Perform DNS pre-flight resolution
    - Block private, loopback, link-local, cloud metadata, and broadcast IPs
    """
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    trimmed_url = url.strip()
    parsed = urlparse(trimmed_url)

    # 1. Enforce strict scheme
    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid URL protocol: '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: Missing hostname")

    # 2. Block obvious local names
    lower_host = hostname.lower()
    if lower_host in ("localhost", "0.0.0.0", "127.0.0.1", "::1", "metadata.google.internal"):
        raise HTTPException(
            status_code=400,
            detail="SSRF Protection: Access to localhost or internal services is forbidden."
        )

    # 3. Resolve hostname to all IP addresses and check each
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise HTTPException(
            status_code=400,
            detail=f"Could not resolve hostname '{hostname}'. Host is unreachable."
        )

    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]

        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid IP address resolved from host")

        # Check for non-public IP ranges
        if (
            ip_obj.is_private or
            ip_obj.is_loopback or
            ip_obj.is_link_local or
            ip_obj.is_multicast or
            ip_obj.is_reserved or
            ip_obj.is_unspecified
        ):
            raise HTTPException(
                status_code=400,
                detail=f"SSRF Protection: Request to internal/private IP '{ip_str}' is blocked."
            )

    return trimmed_url
