import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND)

print("importing app modules...")
from app.config import settings
from app.services.text_extraction import TextExtractor
from app.services.llm import groq_client
from app.services.vision_service import vision_service

print("MISTRAL_LLM_MODEL =", settings.MISTRAL_LLM_MODEL)
print("VISION_MODEL       =", settings.VISION_MODEL)
print("OCR key present    =", bool(settings.MISTRAL_API_KEY))

te = TextExtractor()
pdf = os.path.join(os.path.dirname(__file__), "flashattention.pdf")
b = open(pdf, "rb").read()

print("\n[1] OCR (forced) -> text + figures")
res = te._mistral_ocr_pdf(b)
print("    text chars:", len(res.text), "| figures:", len(res.figures))
fig = next((f for f in res.figures if f.base64), None)
if fig:
    print("    figure id:", fig.id, "| base64 is data-URI:", fig.base64.startswith("data:"),
          "| caption:", (fig.caption[:50] if fig.caption else None))

print("\n[2] VISION graph from first figure (mistral-small-latest)")
g = vision_service.extract_architecture(fig, "FlashAttention") if fig else None
if g:
    print("    nodes:", len(g.nodes), "| edges:", len(g.edges))
    for n in g.nodes[:3]:
        print("      -", n.id, n.label)
else:
    print("    NO GRAPH (vision failed)")

print("\n[3] TEXT path via ChatClient.extract_json (mistral-small-latest, JSON mode)")
out = groq_client.extract_json(messages=[
    {"role": "user", "content": "The paper proposes FlashAttention, a fast attention algorithm. Return JSON {\"name\": \"FlashAttention\", \"paradigm\": \"transformer\"}"}
])
print("    parsed:", out)
print("\nE2E OK" if (res.text and g and out) else "\nE2E PARTIAL (see above)")
