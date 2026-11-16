"""OCR fallback tests.

When the Mistral OCR endpoint is unavailable (e.g. the account's OCR request
limit is 0) the pages must still be transcribed — via the Mistral vision
model — and the response shape must stay identical so the normal parsing
path is reused unchanged.
"""
from unittest.mock import MagicMock, patch

import pymupdf

from app.services.text_extraction import text_extractor


def _pdf(page_count: int = 2) -> bytes:
    doc = pymupdf.open()
    for i in range(page_count):
        doc.new_page().insert_text((72, 100), f"Page {i + 1} of {page_count}")
    data = doc.tobytes()
    doc.close()
    return data


def _ocr_ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _ocr_unavailable() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("429 Too Many Requests")
    return resp


def test_ocr_endpoint_is_used_when_available():
    ocr_resp = _ocr_ok({"pages": [{"index": 0, "markdown": "from ocr", "images": []}]})
    client = MagicMock()
    with patch("app.services.text_extraction.requests.post", return_value=ocr_resp), \
         patch("app.services.llm.groq_client", client):
        result = text_extractor._call_mistral_ocr(_pdf(1))
    assert result["pages"][0]["markdown"] == "from ocr"
    client.call.assert_not_called()


def test_ocr_endpoint_failure_falls_back_to_vision_transcription():
    client = MagicMock()
    client.call.return_value = "Figure 1: The Transformer.\n\nSelf-attention layer."
    with patch("app.services.text_extraction.requests.post", return_value=_ocr_unavailable()), \
         patch("app.services.llm.groq_client", client):
        result = text_extractor._call_mistral_ocr(_pdf(2))
    assert [p["index"] for p in result["pages"]] == [0, 1]
    assert all(p["markdown"].startswith("Figure 1") for p in result["pages"])
    assert client.call.call_count == 2


def test_vision_fallback_sends_the_rendered_page_image():
    client = MagicMock()
    client.call.return_value = "text"
    with patch("app.services.text_extraction.requests.post", return_value=_ocr_unavailable()), \
         patch("app.services.llm.groq_client", client):
        text_extractor._call_mistral_ocr(_pdf(1))
    parts = client.call.call_args.kwargs["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")


def test_vision_fallback_tolerates_a_failing_page():
    client = MagicMock()
    client.call.side_effect = RuntimeError("vision down")
    with patch("app.services.text_extraction.requests.post", return_value=_ocr_unavailable()), \
         patch("app.services.llm.groq_client", client):
        result = text_extractor._call_mistral_ocr(_pdf(2))
    assert [p["markdown"] for p in result["pages"]] == ["", ""]


def test_fallback_output_feeds_the_normal_ocr_parser():
    client = MagicMock()
    client.call.return_value = "Transcribed body text."
    with patch("app.services.text_extraction.requests.post", return_value=_ocr_unavailable()), \
         patch("app.services.llm.groq_client", client):
        raw = text_extractor._call_mistral_ocr(_pdf(1))
        parsed = text_extractor._parse_ocr_response(raw)
    assert parsed.page_count == 1
    assert parsed.pages[0].markdown == "Transcribed body text."
    assert parsed.pages[0].source == "mistral_ocr"


def test_vision_fallback_drops_placeholder_image_refs():
    """The model can emit a fake image URL; no real image sits behind it."""
    client = MagicMock()
    client.call.return_value = "![Image](https://example.com/image.png)\n\nFigure 1: caption."
    with patch("app.services.text_extraction.requests.post", return_value=_ocr_unavailable()), \
         patch("app.services.llm.groq_client", client):
        raw = text_extractor._call_mistral_ocr(_pdf(1))
        parsed = text_extractor._parse_ocr_response(raw)
    assert "example.com" not in parsed.pages[0].markdown
    assert "Figure 1: caption." in parsed.pages[0].markdown
