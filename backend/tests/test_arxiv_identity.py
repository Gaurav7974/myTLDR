"""arXiv identity verification tests (issue: typo'd ID silently decodes wrong paper)."""
from unittest.mock import patch, MagicMock

import pytest

from app.services.text_extraction import text_extractor


def _atom_xml(entry_id: str | None, title: str = "Some Paper") -> str:
    entry = ""
    if entry_id is not None:
        entry = (
            "<entry>"
            f"<id>{entry_id}</id>"
            f"<title>{title}</title>"
            "<summary>Abstract text.</summary>"
            "<author><name>Jane Doe</name></author>"
            "</entry>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"{entry}</feed>"
    )


def test_parse_returns_canonical_id():
    meta = text_extractor._parse_arxiv_xml(
        _atom_xml("http://arxiv.org/abs/1512.03385v2", "Deep Residual Learning")
    )
    assert meta["arxiv_id_base"] == "1512.03385"
    assert meta["title"] == "Deep Residual Learning"


def test_parse_no_entry_gives_empty_base():
    meta = text_extractor._parse_arxiv_xml(_atom_xml(None))
    assert meta["arxiv_id_base"] == ""


def _mock_get(xml: str):
    resp = MagicMock()
    resp.text = xml
    resp.raise_for_status.return_value = None
    return resp


def test_metadata_ok_when_ids_match():
    xml = _atom_xml("http://arxiv.org/abs/1512.03385v2", "Deep Residual Learning")
    with patch("app.services.text_extraction.requests.get", return_value=_mock_get(xml)), \
         patch.object(text_extractor, "_rate_limit_arxiv", return_value=None):
        meta = text_extractor.get_arxiv_metadata("1512.03385")
    assert meta["title"] == "Deep Residual Learning"
    assert meta["arxiv_id_base"] == "1512.03385"


def test_metadata_raises_on_id_mismatch():
    # arXiv returned a different paper than requested — must fail loud
    xml = _atom_xml("http://arxiv.org/abs/1512.03386v1", "Analogue Wormholes")
    with patch("app.services.text_extraction.requests.get", return_value=_mock_get(xml)), \
         patch.object(text_extractor, "_rate_limit_arxiv", return_value=None):
        with pytest.raises(ValueError, match="different paper"):
            text_extractor.get_arxiv_metadata("1512.03385")


def test_metadata_raises_on_unknown_id():
    with patch("app.services.text_extraction.requests.get", return_value=_mock_get(_atom_xml(None))), \
         patch.object(text_extractor, "_rate_limit_arxiv", return_value=None):
        with pytest.raises(ValueError, match="does not match any known paper"):
            text_extractor.get_arxiv_metadata("9999.99999")
