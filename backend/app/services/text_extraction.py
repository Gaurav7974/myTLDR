"""Text extraction from PDFs and arXiv papers."""

import re
import time
import base64
import requests
import pymupdf  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional
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
                # Harvest embedded images with bbox (figure-accurate) before whole-page fallback
                for i, pg in enumerate(doc2):
                    if len(result.figures) >= 5: break
                    for img in pg.get_images(full=True):
                        try:
                            xref = img[0]
                            for inst in pg.get_drawings(): pass  # ensure bbox cache
                            bbox = pg.get_image_bbox(img)
                            if not bbox: continue
                            if len(b64:=base64.b64encode(pg.get_pixmap(clip=bbox, dpi=150).tobytes("png")).decode()) > settings.MAX_INLINE_IMAGE_KB*1024: continue
                            result.figures.append(Figure(id=f"fig-{i+1}-{xref}", page=i+1, base64=b64, caption=f"Figure {len(result.figures)+1} p{i+1}"))
                            if len(result.figures) >= 5: break
                        except Exception: continue
                # If still none, render whole pages
                if not result.figures:
                    for i, pg in enumerate(doc2):
                        if i >= 3: break
                        b64 = base64.b64encode(pg.get_pixmap(dpi=120).tobytes("png")).decode()
                        result.figures.append(Figure(id=f"page-{i+1}-render", page=i+1, base64=b64, caption=f"Page {i+1} render"))
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
        """Download PDF from URL with strict SSRF defense and byte threshold."""
        safe_url = validate_and_sanitize_url(url)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        resp = requests.get(
            safe_url,
            timeout=30,
            stream=True,
            headers={"User-Agent": settings.ARXIV_USER_AGENT}
        )
        resp.raise_for_status()

        # Check content-length header if provided
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            raise ValueError(f"File exceeds maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB}MB)")

        # Stream chunks safely
        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise ValueError(f"File exceeds maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB}MB)")

        return bytes(content)
    
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
        """Enforce rate limiting for arXiv API."""
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
    
    def find_architecture_section(self, text: str) -> str:
        """Locate the architecture/method section. Fallback to first N chars."""
        patterns = [r"(?im)^\s*\d+\.?\s+(?:model\s+)?architecture", r"(?im)^\s*\d+\.?\s+method(?:s)?", r"(?im)^\s*\d+\.?\s+approach", r"(?im)^\s*\d+\.?\s+proposed", r"(?im)^\s*\d+\.?\s+network\s+architecture", r"(?im)^\s*\d+\.?\s+model\s+design", r"(?im)^\s*\d+\.?\s+model\s+structure", r"(?im)^\s*\d+\.?\s+overall\s+architecture"]
        start = next((text.rfind('\n',0,m.start())+1 for p in patterns if (m:=re.search(p,text))), None)
        if start is None or start <0:
            log.warning("[WARN] No architecture section found, using first N chars")
            return text[:settings.MAX_SECTION_CHARS]
        next_section = re.search(r"\n(?=\d+\.?\s+[A-Za-z])", text[start+200:])
        if next_section:
            end = start + 200 + next_section.start()
            section = text[start:end]
        else:
            section = text[start:start + settings.MAX_SECTION_CHARS]
        
        # Hard cap
        if len(section) > settings.MAX_SECTION_CHARS:
            section = section[:settings.MAX_SECTION_CHARS]
        
        log.info(f"Found architecture section ({len(section)} chars)")
        return section
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for content hashing."""
        # Strip whitespace, lowercase, remove page numbers
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
        return text


text_extractor = TextExtractor()
