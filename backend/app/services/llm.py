"""LLM client. Primary provider is Mistral (text + JSON mode), with an optional
Groq failover for the text path. Both expose an OpenAI-compatible /v1/chat/completions
endpoint, so a single client drives them.

Kept the legacy symbol name `groq_client` so existing imports keep working.
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
    """OpenAI-compatible chat client with provider failover."""

    def __init__(self):
        self.providers = []
        # Primary: Mistral (text + vision-capable).
        if settings.MISTRAL_API_KEY:
            self.providers.append({
                "name": "mistral",
                "url": settings.MISTRAL_LLM_URL,
                "key": settings.MISTRAL_API_KEY,
                "model": settings.MISTRAL_LLM_MODEL,
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

    def call(self, messages: list, model: str = None,
             response_format: dict = None, temperature: float = 0.1) -> str:
        """Call the chat endpoint, trying each configured provider in order.

        `model` is a hint (callers pass settings.MISTRAL_LLM_MODEL). It is only
        forwarded to the provider that owns that model name; failover providers
        keep their own configured model, otherwise Groq would receive
        "mistral-small-latest" and return 404.
        """
        if not self.providers:
            raise Exception("No LLM provider configured (set OCR_KEY or GROQ_API_KEY)")
        last_err = None
        for prov in self.providers:
            use_model = model if model == prov["model"] else prov["model"]
            try:
                return self._post(prov, messages, use_model,
                                  response_format, temperature)
            except Exception as e:
                log.warning(f"[LLM] provider '{prov['name']}' failed: {e}")
                last_err = e
                continue
        raise last_err or Exception("All LLM providers failed")

    def _post(self, prov: dict, messages: list, model: str,
              response_format, temperature) -> str:
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
                    resp = _session.post(prov["url"], headers={"Authorization": f"Bearer {prov['key']}", "Content-Type": "application/json"}, json=payload, timeout=45)
                if resp.status_code == 429:
                    wait = self.backoff_base * (attempt + 1)
                    log.warning(f"[RATE LIMIT] Waiting {wait}s on {prov['name']}...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 413:
                    # Request too large: shrink the trailing paper body while keeping
                    # the leading schema/instructions intact, then retry.
                    log.error(f"[ERROR] Payload too large on {prov['name']}, trimming trailing text...", exc_info=True)
                    self._shrink_last_message(payload)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base * (attempt + 1))
                    continue
                raise
        raise Exception(f"Failed after {self.max_retries} retries on {prov['name']}")

    def _shrink_last_message(self, payload: dict, min_chars: int = 2000) -> None:
        """Trim the trailing paper body of the last message, keeping schema/instructions.

        The JSON schema and task instructions always sit at the HEAD of the prompt,
        while the variable paper text is appended at the TAIL. Truncating the tail
        therefore preserves the schema instead of silently destroying it.
        """
        msg = payload["messages"][-1]
        content = msg["content"]
        if len(content) <= min_chars:
            raise Exception("Payload still too large after minimum trimming")
        # content may be a string (text) or a list (multimodal); only trim strings.
        if isinstance(content, str):
            msg["content"] = content[: max(min_chars, len(content) // 2)]
        elif isinstance(content, list):
            # Last text part is the paper body — trim it.
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text", "")
                    if len(txt) > min_chars:
                        part["text"] = txt[: max(min_chars, len(txt) // 2)]
                        break

    def extract_json(self, messages: list, model: str = None) -> dict:
        """Call the API and parse a JSON response."""
        msgs = list(messages)
        has_json_word = any("json" in str(m.get("content", "")).lower() for m in msgs)
        if not has_json_word and len(msgs) > 0:
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\nRespond strictly in valid JSON format."}

        response = self.call(
            messages=msgs,
            model=model,
            response_format={"type": "json_object"},
        )
        return json.loads(response)


# Legacy symbol name kept for backward compatibility with existing imports.
groq_client = ChatClient()
llm_client = groq_client
