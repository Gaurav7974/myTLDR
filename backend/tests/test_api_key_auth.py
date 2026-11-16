"""API-key enforcement: no key must never mean free access on a configured server."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import Request

from app.core.config import settings
from app.core.security import get_rate_limit_key
from app.main import app


@pytest.fixture
def client():
    # No context manager: request tests must not trigger lifespan validation.
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_keys_by_default(monkeypatch):
    monkeypatch.setattr(settings, "MASTER_API_KEY", None)
    monkeypatch.setattr(settings, "API_KEYS", set())
    monkeypatch.setattr(settings, "PROXY_API_KEY", None)
    monkeypatch.setattr(settings, "DEBUG", False)


def _mock_rag():
    return patch(
        "app.api.endpoints.chat.rag_service.answer_question",
        new=AsyncMock(return_value={"answer": "ok", "citations": []}),
    )


def test_missing_key_rejected_when_key_configured(monkeypatch, client):
    monkeypatch.setattr(settings, "MASTER_API_KEY", "test-key")
    with _mock_rag():
        response = client.post("/api/chat", json={"query": "hello"})

    assert response.status_code == 401


def test_wrong_key_rejected(monkeypatch, client):
    monkeypatch.setattr(settings, "MASTER_API_KEY", "test-key")
    with _mock_rag():
        response = client.post(
            "/api/chat", json={"query": "hello"}, headers={"X-API-Key": "wrong"}
        )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "test-key"},
        {"Authorization": "Bearer test-key"},
    ],
)
def test_valid_key_accepted(monkeypatch, client, headers):
    monkeypatch.setattr(settings, "MASTER_API_KEY", "test-key")
    with _mock_rag():
        response = client.post("/api/chat", json={"query": "hello"}, headers=headers)

    assert response.status_code == 200


def test_missing_key_allowed_when_no_keys_configured(client):
    with _mock_rag():
        response = client.post("/api/chat", json={"query": "hello"})

    assert response.status_code == 200


def test_every_api_route_carries_auth_dependency():
    from app.core.security import verify_api_key

    for route in app.routes:
        if getattr(route, "path", "").startswith("/api/"):
            deps = route.dependant.flat_dependant.dependencies
            assert any(d.call is verify_api_key for d in deps), route.path


def test_health_reports_real_auth_state(monkeypatch, client):
    assert client.get("/health").json()["security"]["api_key_auth"] == "disabled"

    monkeypatch.setattr(settings, "MASTER_API_KEY", "test-key")
    assert client.get("/health").json()["security"]["api_key_auth"] == "enabled"


def test_require_master_key_fails_closed_in_production(monkeypatch):
    with pytest.raises(RuntimeError):
        settings.require_master_key()

    monkeypatch.setattr(settings, "DEBUG", True)
    assert settings.require_master_key() is None


def test_proxy_key_rate_limit_uses_forwarded_ip(monkeypatch):
    monkeypatch.setattr(settings, "PROXY_API_KEY", "proxy-key")
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/stream",
            "headers": [
                (b"x-api-key", b"proxy-key"),
                (b"x-forwarded-for", b"1.2.3.4, 10.0.0.1"),
            ],
            "client": ("127.0.0.1", 12345),
        }
    )

    assert get_rate_limit_key(request) == "ip:1.2.3.4"


def test_lifespan_fails_without_key_in_production():
    with patch("app.core.database.init_db", new=AsyncMock()):
        with pytest.raises(RuntimeError, match="API key"):
            with TestClient(app):
                pass
