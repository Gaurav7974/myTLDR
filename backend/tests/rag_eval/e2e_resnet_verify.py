"""Prove the stream-upload path persists to pgvector after the constraint fix.

Uploads ResNet via /api/stream, reads SSE until done, waits for the background
indexer, then counts paper_chunks rows for the returned content_hash.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import sqlalchemy  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

PDF = REPO / "papers" / "1512.03385.pdf"
BASE = "http://127.0.0.1:8000"

boundary = "----resnet-verify"
body = (
    (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="source"\r\n\r\nupload\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="difficulty_level"\r\n\r\nintermediate\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="resnet.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    + PDF.read_bytes()
    + f"\r\n--{boundary}--\r\n".encode()
)

req = urllib.request.Request(
    f"{BASE}/api/stream",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
content_hash = None
with urllib.request.urlopen(req, timeout=600) as resp:
    for line in resp:
        t = line.decode("utf-8", "replace")
        if "content_hash" in t and '"stage": "done"' in t:
            try:
                payload = json.loads(t[t.index("{") :])
                content_hash = payload.get("content_hash")
            except Exception:
                pass

print("content_hash:", content_hash)
if not content_hash:
    print("STREAM DID NOT COMPLETE")
    raise SystemExit(1)

deadline = time.time() + 180
while time.time() < deadline:
    time.sleep(5)

    async def _count():
        engine = create_async_engine(
            "postgresql+asyncpg://postgres:postgres@localhost:5433/papers"
        )
        async with engine.connect() as conn:
            r = await conn.execute(
                text(
                    "SELECT count(*), min(chunker_version), min(embedding_model) "
                    "FROM paper_chunks WHERE paper_hash = :h"
                ),
                {"h": content_hash},
            )
            row = r.first()
        await engine.dispose()
        return row

    import asyncio

    row = asyncio.run(_count())
    n = row[0]
    print(
        f"rows in pgvector for uploaded paper: {n} (version={row[1]}, model={row[2]})",
        flush=True,
    )
    if n and row[1]:
        print("PROOF OK")
        raise SystemExit(0)
print("TIMED OUT waiting for rows")
raise SystemExit(1)
