"""Phase B4: page-aware, boundary-windowed chunking."""

from app.services.rag import chunk_paper_text
from app.services.text_extraction import TextExtractor


def test_single_long_line_no_longer_collapses_into_few_chunks():
    full_text = ("Attention is all you need. " * 4000).strip()
    chunks = chunk_paper_text(full_text, chunk_size=600, overlap=100)
    assert len(chunks) > 50  # 180k chars must produce ~300 chunks, not 2
    assert all(len(c["content"]) <= 700 for c in chunks)


def test_page_offsets_assign_page_numbers():
    extractor = TextExtractor()
    page1 = "Page one content about transformers and attention. " * 30
    page2 = "Page two content about residual bottlenecks. " * 30
    from app.services.text_extraction import PageText

    result = extractor._build_result(
        [
            PageText(page_number=1, markdown=page1, source="pymupdf"),
            PageText(page_number=2, markdown=page2, source="pymupdf"),
        ]
    )
    chunks = chunk_paper_text(result.text, page_offsets=result.page_offsets)
    pages = {c["page_number"] for c in chunks}
    assert 1 in pages and 2 in pages
    assert all(p is not None for p in pages)


def test_without_offsets_page_number_is_none_and_contract_holds():
    chunks = chunk_paper_text("Abstract. Attention is all you need. " * 10)
    assert chunks
    assert all("page_number" in c for c in chunks)
    assert all(c["page_number"] is None for c in chunks)
    assert all({"chunk_index", "section_title", "content"} <= set(c) for c in chunks)


def test_offsets_survive_reference_tail_strip():
    extractor = TextExtractor()
    body = "Transformer method content. " * 100
    tail = "\nReferences\n[1] Some paper 2020.\n" + " [1] Ref line.\n" * 30
    from app.services.text_extraction import PageText

    result = extractor._build_result(
        [
            PageText(page_number=1, markdown=body, source="pymupdf"),
            PageText(page_number=2, markdown=tail, source="pymupdf"),
        ]
    )
    offsets = result.page_offsets
    # Page 2 was stripped away (short doc? no, strip only if >500 chars & in last 60%)
    assert all(off < len(result.text) or off < 1 for _, off in offsets)
    chunks = chunk_paper_text(result.text, page_offsets=offsets)
    for c in chunks:
        assert c["page_number"] is not None
