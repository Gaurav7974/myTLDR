"""Text extraction from PDFs and arXiv papers."""

import re
import time
import base64
import requests
import pymupdf  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urljoin
from app.core.config import settings
from app.core.logger import get_logger
log = get_logger(__name__)
from app.core.security import sanitize_arxiv_id, validate_and_sanitize_url


@dataclass
class Figure:
    """A figure/diagram extracted from the paper, with its base64 image and caption."""
    id: str
    page: int
    base64: Optional[str]
    caption: Optional[str] = None
    is_whole_page_fallback: bool = False


@dataclass
class PageText:
    """Extracted content of a single PDF page, ready for the Parsed Text tab."""
    page_number: int  # 1-based
    markdown: str
    source: str  # "pymupdf" | "mistral_ocr"


@dataclass
class ExtractionResult:
    """Result of extracting a paper: per-page content, flat text, and figures.

    `text` is the plain-text join of all pages (images stripped) and feeds the
    LLM/RAG pipeline; `pages` keeps the per-page markdown (with inline images)
    for the frontend reader.
    """
    text: str
    figures: List[Figure] = field(default_factory=list)
    pages: List[PageText] = field(default_factory=list)
    page_count: int = 0


class TextExtractor:
    """Extracts text and figures from PDFs and arXiv papers."""

    def __init__(self):
        self._last_arxiv_call = 0

    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text per page. PyMuPDF for text pages, Mistral OCR for image pages."""
        if not pdf_bytes or not pdf_bytes.lstrip()[:5] == b"%PDF-":
            raise ValueError("Uploaded file is not a valid PDF (missing %PDF header)")
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages: List[PageText] = [
            PageText(page_number=i + 1, markdown=page.get_text(), source="pymupdf")
            for i, page in enumerate(doc)
        ]
        page_count = len(pages)
        doc.close()

        if not pages:
            return ExtractionResult(text="", figures=[], pages=[], page_count=0)

        ocr_candidates = [
            p for p in pages
            if len(p.markdown.strip()) < settings.OCR_PAGE_TEXT_THRESHOLD
        ]

        if len(ocr_candidates) == page_count:
            # No text layer anywhere — one whole-document OCR call covers all pages
            log.info("[OCR] No text layer found, calling Mistral OCR on whole document...")
            try:
                return self._mistral_ocr_pdf(pdf_bytes)
            except Exception as e:
                log.warning(f"[WARN] Mistral OCR failed: {e} — returning raw text layer")
                return self._build_result(pages)

        # Hybrid: OCR only the image/scanned pages individually
        hybrid_figures: List[Figure] = []
        for p in ocr_candidates[:settings.OCR_MAX_PAGE_CALLS]:
            try:
                md, figs = self._mistral_ocr_single_page(pdf_bytes, p.page_number - 1)
                if md:
                    p.markdown = md
                    p.source = "mistral_ocr"
                hybrid_figures.extend(figs)
            except Exception as e:
                log.warning(f"[WARN] OCR failed for page {p.page_number}: {e}")
        if len(ocr_candidates) > settings.OCR_MAX_PAGE_CALLS:
            log.warning(f"[WARN] {len(ocr_candidates)} image pages exceed OCR_MAX_PAGE_CALLS; only the first {settings.OCR_MAX_PAGE_CALLS} were OCR'd")
        result = self._build_result(pages, hybrid_figures)
        if not result.figures:
            try:
                doc2 = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                # Harvest every raster XObject, always (cap 60 safety, not 5)
                for i, pg in enumerate(doc2):
                    if len(result.figures) >= 60: break
                    for img in pg.get_images(full=True):
                        if len(result.figures) >= 60: break
                        try:
                            xref = img[0]
                            bbox = pg.get_image_bbox(img)
                            if not bbox: continue
                            if len(b64:=base64.b64encode(pg.get_pixmap(clip=bbox, dpi=150).tobytes("png")).decode()) > settings.MAX_INLINE_IMAGE_KB*1024: continue
                            result.figures.append(Figure(id=f"fig-{i+1}-{xref}", page=i+1, base64=b64, caption=f"Figure {len(result.figures)+1} p{i+1}"))
                        except Exception: continue
                # General vector fallback: no raster but page has drawings/text → render crop around shape-dense region
                if not result.figures:
                    for i, pg in enumerate(doc2):
                        if len(result.figures) >= 60: break
                        drawings, text_len = len(pg.get_drawings()), len(pg.get_text().strip())
                        if drawings == 0 and text_len < 200: continue
                        # Coarse crop: shape-dense region (where drawings cluster) vs text
                        try:
                            # Use whole page for now; page-level classifier (step 2) will filter which page holds diagram
                            b64 = base64.b64encode(pg.get_pixmap(dpi=120).tobytes("png")).decode()
                            # Heuristic: high drawings + moderate text suggests diagram page
                            is_diagram_page = drawings > 5 or (drawings > 0 and text_len < 2000)
                            if is_diagram_page or i < 5:  # keep first 5 vector pages as candidates
                                result.figures.append(Figure(id=f"page-{i+1}-render", page=i+1, base64=b64, caption=f"Page {i+1} render", is_whole_page_fallback=True))
                        except Exception: continue
                doc2.close()
            except Exception as e: log.warning(f"[WARN] pixmap fallback failed: {e}")
        return result

    def _mistral_ocr_pdf(self, pdf_bytes: bytes) -> ExtractionResult:
        """Call Mistral OCR API for image-based PDFs, returning per-page text + figures."""
        result = self._call_mistral_ocr(pdf_bytes)
        return self._parse_ocr_response(result)

    def _mistral_ocr_single_page(self, pdf_bytes: bytes, page_index: int) -> tuple:
        """OCR one page (0-based) of a PDF; returns (markdown with inline images, figures)."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            doc.select([page_index])
            single_bytes = doc.tobytes()
        finally:
            doc.close()
        result = self._call_mistral_ocr(single_bytes)
        pages = result.get("pages", [])
        if not pages:
            return "", []
        page = pages[0]
        raw_md = page.get("markdown", "") or ""
        md = self._inline_images(raw_md, page.get("images", []) or [], {"count": 0})
        captions = self._extract_figure_captions(raw_md)
        figures = []
        for img in page.get("images", []) or []:
            img_id = img.get("id")
            b64 = img.get("image_base64") or img.get("base64")
            if not img_id or not b64:
                continue
            figures.append(Figure(
                id=img_id,
                page=page_index,
                base64=b64,
                caption=captions.get(img_id),
            ))
        return md, figures

    def _call_mistral_ocr(self, pdf_bytes: bytes) -> dict:
        """POST a PDF to Mistral OCR and return the raw response JSON."""
        if not settings.MISTRAL_API_KEY:
            raise Exception("Mistral OCR API key not configured")

        # Encode PDF as base64
        b64 = base64.b64encode(pdf_bytes).decode()

        resp = requests.post(
            settings.MISTRAL_OCR_URL,
            headers={
                "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": settings.MISTRAL_OCR_MODEL,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{b64}"
                },
                # Request the actual figure images so they can be fed to a vision model later.
                "include_image_base64": True,
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_ocr_response(self, ocr_result: dict) -> ExtractionResult:
        """Parse Mistral OCR response into per-page content and figures."""
        raw_pages = ocr_result.get("pages", [])
        pages: List[PageText] = []
        figures: List[Figure] = []
        inline_state = {"count": 0}
        for i, page in enumerate(raw_pages):
            md = page.get("markdown", "") or ""
            page_idx = page.get("index", i)
            md = self._inline_images(md, page.get("images", []) or [], inline_state)
            if md:
                pages.append(PageText(
                    page_number=page_idx + 1,
                    markdown=md,
                    source="mistral_ocr"
                ))
            captions = self._extract_figure_captions(page.get("markdown", "") or "")
            for img in page.get("images", []) or []:
                img_id = img.get("id")
                if not img_id:
                    continue
                b64 = img.get("image_base64") or img.get("base64")
                if not b64:
                    continue
                figures.append(Figure(
                    id=img_id,
                    page=page_idx,
                    base64=b64,
                    caption=captions.get(img_id),
                ))
        return self._build_result(pages, figures)

    def _inline_images(self, markdown: str, images: list, inline_state: dict) -> str:
        """Replace `![id](id)` image refs with inline data URIs, subject to payload caps.

        Images over the size/count caps are stripped so the frontend never sees a
        broken ref. `inline_state` carries the running count across pages.
        """
        by_id = {}
        for img in images:
            img_id = img.get("id")
            if img_id:
                by_id[img_id] = img.get("image_base64") or img.get("base64")

        max_chars = settings.MAX_INLINE_IMAGE_KB * 1024

        def _replace(m):
            alt = m.group("alt")
            ref = m.group("ref")
            if ref.startswith("data:") or ref.startswith("http"):
                return m.group(0)
            b64 = by_id.get(ref)
            if not b64:
                return ""
            if inline_state["count"] >= settings.MAX_INLINE_IMAGES or len(b64) > max_chars:
                return ""
            inline_state["count"] += 1
            mime = "image/png" if ref.lower().endswith(".png") else "image/jpeg"
            alt_text = alt if alt and alt != ref else "Figure"
            return f"![{alt_text}](data:{mime};base64,{b64})"

        return re.sub(r"!\[(?P<alt>[^\]]*)\]\((?P<ref>[^)]+)\)", _replace, markdown)

    def _strip_images(self, markdown: str) -> str:
        """Remove image syntax so base64 payloads never reach the LLM/RAG text path."""
        return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)

    def _build_result(self, pages: List[PageText], figures: Optional[List[Figure]] = None) -> ExtractionResult:
        """Join per-page content into the flat text the pipeline consumes."""
        text_parts = [self._strip_images(p.markdown) if p.source == "mistral_ocr" else p.markdown for p in pages]
        return ExtractionResult(text="\n\n".join(p for p in text_parts if p), figures=figures or [], pages=pages, page_count=len(pages))
    
    def _extract_figure_captions(self, markdown: str) -> dict:
        """Map each image id (referenced as `![id](id)`) to its following caption line."""
        captions: dict = {}
        lines = markdown.splitlines()
        for i, line in enumerate(lines):
            m = re.search(r"!\[(?P<id>[^\]]+)\]\((?P=id)\)", line)
            if not m:
                continue
            img_id = m.group("id")
            for nxt in lines[i + 1:i + 4]:
                stripped = nxt.strip()
                if stripped and re.match(r"(?i)\s*(figure|fig\.?)\b", stripped):
                    captions[img_id] = stripped
                    break
        return captions
    
    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL with SSRF-safe redirect handling and byte cap."""
        safe_url = validate_and_sanitize_url(url)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        # Do not auto-follow redirects — validate each Location header
        current_url = safe_url
        for _ in range(5):  # max redirects
            resp = requests.get(
                current_url,
                timeout=30,
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": settings.ARXIV_USER_AGENT}
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                if not loc:
                    resp.raise_for_status()
                # Resolve relative / protocol-relative targets against the
                # current URL first — urlparse rejects them as scheme-less,
                # which broke every arXiv download.
                current_url = validate_and_sanitize_url(urljoin(current_url, loc))
                continue
            resp.raise_for_status()
            cl = resp.headers.get("Content-Length")
            if cl and int(cl) > max_bytes:
                raise ValueError(f"File exceeds maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB}MB)")
            content = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise ValueError(f"File exceeds maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB}MB)")
            return bytes(content)
        raise ValueError("Too many redirects")
    
    def download_arxiv_pdf(self, arxiv_id: str) -> bytes:
        """Download PDF from arXiv with canonical ID sanitization and rate limiting."""
        clean_id = sanitize_arxiv_id(arxiv_id)
        self._rate_limit_arxiv()
        url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        return self.download_pdf(url)
    
    def get_arxiv_metadata(self, arxiv_id: str) -> dict:
        """Fetch paper metadata from arXiv API with sanitized ID."""
        clean_id = sanitize_arxiv_id(arxiv_id)
        self._rate_limit_arxiv()
        url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": settings.ARXIV_USER_AGENT
        })
        resp.raise_for_status()
        return self._parse_arxiv_xml(resp.text)
    
    def _rate_limit_arxiv(self):
        """Enforce rate limiting for arXiv API (sync — caller must be in to_thread)."""
        elapsed = time.time() - self._last_arxiv_call
        if elapsed < settings.ARXIV_RATE_LIMIT_SECONDS:
            time.sleep(settings.ARXIV_RATE_LIMIT_SECONDS - elapsed)
        self._last_arxiv_call = time.time()
    
    def _parse_arxiv_xml(self, xml_text: str) -> dict:
        """Parse arXiv Atom XML response."""
        import xml.etree.ElementTree as ET
        
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"
        }
        
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ns)
        
        if entry is None:
            return {"title": "", "authors": [], "abstract": "", "categories": []}
        
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text)
        
        categories = []
        for cat in entry.findall("atom:category", ns):
            categories.append(cat.get("term", ""))
        
        return {
            "title": title.text.strip() if title is not None else "",
            "authors": authors,
            "abstract": summary.text.strip() if summary is not None else "",
            "categories": categories
        }
    
    def extract_paper_metadata_from_text(self, text: str) -> dict:
        """Extract paper title, authors, published year, abstract, and keywords from first page text using LLM."""
        from app.services.llm import groq_client
        first_page = text[:3500]
        prompt = f"""Extract the exact metadata from the first page of this academic research paper:
---
{first_page}
---

Respond ONLY with a valid JSON object matching:
{{
  "title": "Exact Full Paper Title",
  "authors": ["Author 1", "Author 2", "Author 3"],
  "published_year": "2024",
  "abstract": "Exact abstract text or concise summary",
  "categories": ["Computer Vision", "Generative Models"]
}}"""
        try:
            res = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": "You are an expert research paper librarian. Extract the exact title, authors, year, and abstract from the paper text."},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "title": res.get("title") or "Uploaded Research Paper",
                "authors": res.get("authors") or ["Lead Author et al."],
                "abstract": res.get("abstract") or "",
                "published": str(res.get("published_year") or "Recent"),
                "categories": res.get("categories") or ["Artificial Intelligence"]
            }
        except Exception as e:
            log.warning(f"[WARN] LLM metadata extraction failed: {e}")
            lines = [l.strip() for l in first_page.split("\n") if len(l.strip()) > 5]
            guessed_title = lines[0] if lines else "Uploaded Research Paper"
            return {
                "title": guessed_title,
                "authors": ["Research Team"],
                "abstract": "",
                "published": "Recent",
                "categories": ["Machine Learning"]
            }
    
    # ── LLM-based architecture section detection (Fix #2) ──────────
    ARCH_SECTION_PROMPT = """You are an expert research-paper parser. Given the full text of a paper, locate the architecture / method / model section that describes the model's components, layers, blocks, and their connections.

Instructions:
- Read the full paper text below and identify the contiguous span that is the architecture/method description (headings like "Architecture", "Model", "Method", "Approach", "Proposed Method", "Network Architecture", "Model Design", etc., plus subsections).
- Return the VERBATIM text of that section (copy exactly from the source, do not paraphrase or invent). If the section is long, return its full contiguous text up to {ceiling} characters; do not truncate mid-sentence unless you must hit the ceiling.
- If the paper has NO clearly identifiable architecture/method section (e.g. survey, dataset paper, or the section is missing/ambiguous), set found=false.
- Confidence: 0.0-1.0 — how confidently you located a real architecture section.

Output strictly valid JSON:
{{"found": true|false, "section_text": "verbatim excerpt or empty string", "confidence": 0.0-1.0, "reason": "one-sentence justification, include the heading you matched if found"}}

Full paper text (truncated to {max_chars} chars for this call):
---
{text}
---"""

    def find_architecture_section(self, text: str) -> str | None:
        """LLM-based architecture/method section locator (Fix #2).

        Returns the verbatim section text on success, or None if the LLM cannot
        confidently locate a section. Callers must treat None as an explicit
        failure state (diagram/explanation unavailable) — NOT as a silent
        fallback to arbitrary intro text.
        """
        if not text or len(text.strip()) < 100:
            log.warning("[ARCH_SECTION] Input too short, no architecture section")
            return None

        # Truncate to what the locator LLM can ingest (keep head + method area)
        locator_input = text[: settings.ARCH_SECTION_LLM_MAX_CHARS]
        ceiling = settings.SECTION_CHAR_CEILING

        prompt = self.ARCH_SECTION_PROMPT.format(
            ceiling=ceiling,
            max_chars=settings.ARCH_SECTION_LLM_MAX_CHARS,
            text=locator_input,
        )

        # Import lazily to avoid circular deps at module load
        try:
            from app.services.llm import groq_client
        except Exception as e:
            log.error(f"[ARCH_SECTION] LLM client unavailable: {e}")
            return self._find_architecture_section_regex_fallback(text, reason="llm_unavailable")

        try:
            data = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": "You are a precise research-paper section locator. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=settings.MISTRAL_LLM_MODEL,
            )
        except Exception as e:
            log.warning(f"[ARCH_SECTION] LLM call failed: {e} — no fallback to arbitrary text")
            return self._find_architecture_section_regex_fallback(text, reason=f"llm_error: {e}")

        found = bool(data.get("found", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        section_text = (data.get("section_text") or "").strip()
        reason = data.get("reason") or ""

        # Explicit failure checks
        if not found:
            log.warning(f"[ARCH_SECTION] LLM could not locate architecture section: {reason}")
            return None
        if confidence < 0.5:
            log.warning(f"[ARCH_SECTION] Low confidence {confidence:.2f}: {reason} — treating as not found")
            return None
        if len(section_text) < 200:
            log.warning(f"[ARCH_SECTION] Section too short ({len(section_text)} chars), confidence {confidence:.2f}: {reason}")
            return None
        # Verify extracted text actually appears in source (anti-hallucination)
        # Check that a 60-char sample from the LLM output exists in the source
        sample = section_text[:60].strip()
        if sample and sample not in text and sample[:40] not in text:
            log.warning(f"[ARCH_SECTION] LLM section_text not found verbatim in source — possible hallucination: {reason}")
            return None

        # Dynamic sizing (Fix #3): cap to ceiling, otherwise keep actual length
        if len(section_text) > ceiling:
            log.info(f"[ARCH_SECTION] Truncating {len(section_text)} -> {ceiling} chars (ceiling)")
            section_text = section_text[:ceiling]

        log.info(f"[ARCH_SECTION] Found section ({len(section_text)} chars, conf={confidence:.2f}): {reason[:120]}")
        return section_text

    def find_architecture_section_or_fallback(self, text: str) -> str | None:
        """Compatibility wrapper that preserves old call sites expecting a string.

        New code should call find_architecture_section() directly and handle None.
        This wrapper is kept for tests that assert a string is returned.
        """
        result = self.find_architecture_section(text)
        if result is not None:
            return result
        # Do NOT silently return first N chars — surface None
        return None

    def _find_architecture_section_regex_fallback(self, text: str, reason: str = "") -> str | None:
        """Legacy regex locator — used ONLY for diagnostics, never as silent success.

        Returns None to force explicit failure propagation. Logs what regex would
        have returned so operators can compare LLM vs regex on real papers.
        """
        patterns = [
            r"(?im)^\s*\d+\.?\s+(?:model\s+)?architecture",
            r"(?im)^\s*\d+\.?\s+method(?:s)?",
            r"(?im)^\s*\d+\.?\s+approach",
            r"(?im)^\s*\d+\.?\s+proposed",
            r"(?im)^\s*\d+\.?\s+network\s+architecture",
            r"(?im)^\s*\d+\.?\s+model\s+design",
            r"(?im)^\s*\d+\.?\s+model\s+structure",
            r"(?im)^\s*\d+\.?\s+overall\s+architecture",
        ]
        start = next((text.rfind('\n', 0, m.start()) + 1 for p in patterns if (m := re.search(p, text))), None)
        if start is None or start < 0:
            log.warning(f"[ARCH_SECTION] Regex fallback also found nothing (LLM reason: {reason})")
            return None
        nxt = re.search(r"\n(?=\d+\.?\s+[A-Za-z])", text[start + 200 :])
        if nxt:
            end = start + 200 + nxt.start()
            section = text[start:end]
        else:
            section = text[start : start + settings.SECTION_CHAR_CEILING]
        if len(section) > settings.SECTION_CHAR_CEILING:
            section = section[: settings.SECTION_CHAR_CEILING]
        log.info(f"[ARCH_SECTION] Regex fallback would have returned {len(section)} chars (LLM reason: {reason}) — NOT used, surfacing failure")
        # Important: still return None — do not let regex substitute for LLM failure
        return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for content hashing."""
        # Strip whitespace, lowercase, remove page numbers
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
        return text


text_extractor = TextExtractor()
