"""LLM client. Primary provider is Mistral (text + JSON mode), with an optional
Groq failover for the text path. Both expose an OpenAI-compatible /v1/chat/completions
endpoint, so a single client drives them.

Kept the legacy symbol name groq_client so existing imports keep working.
"""

import time, threading, json, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.core.config import settings
from app.core.logger import get_logger
log = get_logger(__name__)
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=Retry(total=0)))
_sem = threading.Semaphore(4)


class ChatClient:
    """OpenAI-compatible chat client with provider failover and Mistral key rotation.

    When MISTRAL_API_KEY_2 is configured, every LLM call atomically picks the next
    key in a round-robin so quota is split evenly across both Mistral accounts.
    """

    def __init__(self):
        self.providers = []
        # Primary: Mistral (text + vision-capable).
        # If a second Mistral key is configured, register it alongside the first.
        # _next_mistral_key() rotates between them per-call so rate-limit quota
        # is shared across both accounts - effectively doubling throughput.
        if settings.MISTRAL_API_KEY:
            mistral_keys = [settings.MISTRAL_API_KEY]
            if settings.MISTRAL_API_KEY_2:
                mistral_keys.append(settings.MISTRAL_API_KEY_2)
                log.info(f"[LLM] Mistral key rotation enabled: {len(mistral_keys)} keys configured")
            self._mistral_keys = mistral_keys
            self._mistral_key_idx = 0
            self._mistral_key_lock = threading.Lock()
            self.providers.append({
                "name": "mistral",
                "url": settings.MISTRAL_LLM_URL,
                "key": None,          # resolved per-call via _next_mistral_key()
                "model": settings.MISTRAL_LLM_MODEL,
                "_rotating": True,    # signals _post() to pick key dynamically
            })
        # Optional failover: Groq (text-only).
        if settings.LLM_FALLBACK_ENABLED and settings.GROQ_API_KEY:
            self.providers.append({
                "name": "groq",
                "url": settings.GROQ_API_URL,
                "key": settings.GROQ_API_KEY,
                "model": settings.LLM_FALLBACK_MODEL,
            })
        self.max_retries = settings.LLM_MAX_RETRIES
        self.backoff_base = settings.LLM_RETRY_BACKOFF_BASE

    # -- Key rotation ----------------------------------------------------------

    def _next_mistral_key(self) -> str:
        """Return the next Mistral API key in round-robin order (thread-safe)."""
        with self._mistral_key_lock:
            key = self._mistral_keys[self._mistral_key_idx % len(self._mistral_keys)]
            self._mistral_key_idx += 1
            return key

    # -- Public API ------------------------------------------------------------

    def call(self, messages: list, model: str = None,
             response_format: dict = None, temperature: float = 0.1) -> str:
        """Call the chat endpoint, trying each configured provider in order."""
        if not self.providers:
            raise Exception("No LLM provider configured (set OCR_KEY or GROQ_API_KEY)")

        msgs = list(messages)
        if response_format and response_format.get("type") == "json_object":
            last_user_idx = next(
                (i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"),
                None,
            )
            has_json = False
            if last_user_idx is not None:
                has_json = "json" in str(msgs[last_user_idx].get("content", "")).lower()
            if not has_json and len(msgs) > 0:
                msgs = [dict(m) for m in msgs]
                if last_user_idx is not None and isinstance(msgs[last_user_idx].get("content"), str):
                    msgs[last_user_idx]["content"] = msgs[last_user_idx]["content"] + "\n\nRespond strictly in valid JSON format."
                elif isinstance(msgs[0].get("content"), str):
                    msgs[0]["content"] = msgs[0]["content"] + "\n\nRespond strictly in valid JSON format."

        last_err = None
        for prov in self.providers:
            use_model = model if model == prov["model"] else prov["model"]
            try:
                return self._post(prov, msgs, use_model, response_format, temperature)
            except Exception as e:
                log.warning(f"[LLM] provider '{prov['name']}' failed: {e}")
                last_err = e
                continue
        raise last_err or Exception("All LLM providers failed")

    def _post(self, prov: dict, messages: list, model: str,
              response_format, temperature) -> str:
        # Resolve the API key: rotate for Mistral, use fixed key for Groq.
        api_key = self._next_mistral_key() if prov.get("_rotating") else prov["key"]
        key_hint = f" (key ...{api_key[-6:]})" if prov.get("_rotating") and len(self._mistral_keys) > 1 else ""

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(self.max_retries):
            try:
                with _sem:
                    resp = _session.post(
                        prov["url"],
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=120,
                    )
                if resp.status_code == 429:
                    wait = self.backoff_base * (attempt + 1)
                    log.warning(f"[RATE LIMIT] Waiting {wait}s on {prov['name']}{key_hint}...")
                    time.sleep(wait)
                    # On 429, rotate to the other key for the retry attempt
                    if prov.get("_rotating") and len(self._mistral_keys) > 1:
                        api_key = self._next_mistral_key()
                    continue
                if resp.status_code == 413:
                    log.error(f"[ERROR] Payload too large on {prov['name']}, trimming trailing text...", exc_info=True)
                    self._shrink_last_message(payload)
                    continue
                if resp.status_code >= 400:
                    log.warning(f"[{prov['name']} HTTP {resp.status_code}]: {resp.text[:500]}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base * (attempt + 1))
                    continue
                raise
        raise Exception(f"Failed after {self.max_retries} retries on {prov['name']}")

    def _shrink_last_message(self, payload: dict, min_chars: int = 2000) -> None:
        """Trim the trailing paper body of the last message, keeping schema/instructions."""
        msg = payload["messages"][-1]
        content = msg["content"]
        if len(content) <= min_chars:
            raise Exception("Payload still too large after minimum trimming")
        if isinstance(content, str):
            msg["content"] = content[: max(min_chars, len(content) // 2)]
        elif isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text", "")
                    if len(txt) > min_chars:
                        part["text"] = txt[: max(min_chars, len(txt) // 2)]
                        break

    def extract_json(self, messages: list, model: str = None) -> dict:
        """Call the API and parse a JSON response."""
        msgs = list(messages)
        last_user_idx = next(
            (i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"),
            None,
        )
        if last_user_idx is not None:
            content = msgs[last_user_idx].get("content", "")
            if isinstance(content, str) and "json" not in content.lower():
                msgs[last_user_idx] = {
                    **msgs[last_user_idx],
                    "content": content + "\n\nRespond strictly in valid JSON format.",
                }
        elif len(msgs) > 0:
            content = msgs[0].get("content", "")
            if isinstance(content, str) and "json" not in content.lower():
                msgs[0] = {**msgs[0], "content": content + "\n\nRespond strictly in valid JSON format."}

        response = self.call(
            messages=msgs,
            model=model,
            response_format={"type": "json_object"},
        )
        return json.loads(response)


# Legacy symbol name kept for backward compatibility with existing imports.
groq_client = ChatClient()
llm_client = groq_client
