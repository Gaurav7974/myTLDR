"""Test Mistral OCR API against the PDFs in this folder.

Reads OCR_KEY from backend/.env and calls https://api.mistral.ai/v1/ocr
for each PDF, then reports whether text and images came back.
"""
import base64
import os
import sys

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE, "..", "backend", ".env")
OCR_URL = "https://api.mistral.ai/v1/ocr"
OCR_MODEL = "mistral-ocr-latest"


def load_key():
    key = ""
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OCR_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def test_pdf(path, api_key):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n=== {name} ({size_mb:.2f} MB) ===")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    try:
        resp = requests.post(
            OCR_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OCR_MODEL,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{b64}",
                },
            },
            timeout=180,
        )
    except Exception as e:
        print(f"  NETWORK/REQUEST ERROR: {e}")
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  ERROR BODY: {resp.text[:800]}")
        return False

    data = resp.json()
    pages = data.get("pages", [])
    total_text = 0
    total_images = 0
    for i, page in enumerate(pages):
        md = page.get("markdown", "") or ""
        imgs = page.get("images", []) or []
        total_text += len(md)
        total_images += len(imgs)
        print(f"  page {i+1}: {len(md)} chars markdown, {len(imgs)} images")

    ok = total_text > 0
    print(f"  SUMMARY: {len(pages)} pages, {total_text} chars text, {total_images} images -> "
          f"{'OK' if ok else 'NO TEXT'}")
    return ok


def main():
    api_key = load_key()
    if not api_key:
        print("OCR_KEY not found in backend/.env")
        sys.exit(1)
    print(f"Using OCR key: {api_key[:6]}...{api_key[-4:]}")

    pdfs = sorted(
        os.path.join(BASE, f)
        for f in os.listdir(BASE)
        if f.lower().endswith(".pdf")
    )
    if not pdfs:
        print("No PDFs found in test folder")
        sys.exit(1)

    results = {}
    for pdf in pdfs:
        results[os.path.basename(pdf)] = test_pdf(pdf, api_key)

    print("\n================ STATUS ================")
    for name, ok in results.items():
        print(f"  {name}: {'WORKING' if ok else 'FAILED'}")
    all_ok = all(results.values())
    print(f"  OVERALL: {'MISTRAL OCR WORKING' if all_ok else 'SOME/ALL FAILED'}")


if __name__ == "__main__":
    main()
