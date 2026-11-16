"""429 handling: fail over immediately; cool a provider down after two strikes."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.services import llm as llm_mod


def _client(provider_urls):
    c = llm_mod.ChatClient.__new__(llm_mod.ChatClient)
    c.providers = [
        {"name": name, "url": url, "key": "k", "model": "m"}
        for name, url in provider_urls
    ]
    c.max_retries = 4
    c.backoff_base = 3
    c._min_interval = {}
    c._throttle_state = {}
    c._throttle_lock = threading.Lock()
    c._consec_429 = {}
    c._cooldown_until = {}
    c._breaker_lock = threading.Lock()
    return c


def _response(status, body=None):
    resp = MagicMock()
    resp.status_code = status
    if body is not None:
        resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


_OK = {"choices": [{"message": {"content": "ok"}}]}


def test_429_fails_over_without_sleeping():
    c = _client([("a", "https://a"), ("b", "https://b")])
    session = MagicMock()
    session.post.side_effect = [_response(429), _response(200, _OK)]
    with (
        patch("app.services.llm._session", session),
        patch("app.services.llm.time.sleep") as sleep,
    ):
        out = c.call([{"role": "user", "content": "hi"}])
    assert out == "ok"
    sleep.assert_not_called()
    assert [call.args[0] for call in session.post.call_args_list] == [
        "https://a",
        "https://b",
    ]


def test_two_429s_cool_provider_down_for_subsequent_calls():
    c = _client([("a", "https://a"), ("b", "https://b")])
    session = MagicMock()
    session.post.side_effect = [
        _response(429),  # call 1: a fails -> falls back to b
        _response(200, _OK),
        _response(429),  # call 2: a fails again -> breaker trips
        _response(200, _OK),
        _response(200, _OK),  # call 3: a is skipped, only b is hit
    ]
    with (
        patch("app.services.llm._session", session),
        patch("app.services.llm.time.sleep"),
    ):
        c.call([{"role": "user", "content": "hi"}])
        c.call([{"role": "user", "content": "hi"}])
        out = c.call([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert [call.args[0] for call in session.post.call_args_list].count(
        "https://a"
    ) == 2
    assert c._cooldown_until["a"] > time.time()
