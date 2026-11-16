"""Final live verification: one grounded answer, one off-topic refusal."""

import httpx
import time
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.services.text_extraction import text_extractor
from app.services.pipeline import compute_content_hash

text = text_extractor.extract_from_pdf_bytes(
    (REPO / "testing" / "papers" / "1706.03762.pdf").read_bytes()
).text
h = compute_content_hash(text)
qs = [
    "What BLEU scores were achieved on German and French?",
    "What is the capital of France?",
    "What optimizer was used?",
]
with httpx.Client(timeout=120) as c:
    for q in qs:
        t0 = time.time()
        r = c.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "paper_hash": h,
                "paper_title": "Attention Is All You Need",
                "query": q,
                "full_text": text,
            },
        )
        d = r.json()
        cits = d.get("citations", [])
        answer = d.get("answer", "")
        secs = round(time.time() - t0, 1)
        print(
            f"{q[:55]} -> citations={len(cits)} answer_chars={len(answer)} "
            f"llm_failure={answer.startswith('The AI service is temporarily')} ({secs}s)"
        )
