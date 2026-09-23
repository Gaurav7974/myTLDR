"""Text extraction from PDFs and arXiv papers."""

import re
import time
import base64
import requests
import pymupdf  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional
from html.parser import HTMLParser
from urllib.parse import urljoin
from app.core.config import settings
from app.core.logger import get_logger
log = get_logger(__name__)
from app.core.security import sanitize_arxiv_id, validate_and_sanitize_url

# Matches "References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY" as a
# standalone section header at the start of a line, with optional leading
# digit/dot numbering (e.g. "5 References", "5. References").
# Only searched in the last 60 % of the document to avoid false positives
# in the body text (e.g. "As shown in previous references [1–5]...").
_TAIL_STRIP_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+\.?\s+)?(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)[ \t]*(?:\n|$)"
)


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


class _ArxivAbsParser(HTMLParser):
    """Parse title, authors, abstract, and subjects from arXiv /abs/{id} HTML."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.authors: List[str] = []
        self.abstract = ""
        self.categories: List[str] = []
        self._in_abstract = False
        self._abstract_chunks: List[str] = []
        self._in_subject = False
        self._subject_chunks: List[str] = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if tag == "meta":
            name = attr_dict.get("name", "")
            content = attr_dict.get("content", "")
            if name == "citation_title" and not self.title:
                self.title = content.strip()
            elif name == "citation_author":
                author_name = content.strip()
                if author_name and author_name not in self.authors:
                    self.authors.append(author_name)
        elif tag == "blockquote" and "abstract" in attr_dict.get("class", ""):
            self._in_abstract = True
        elif tag == "td" and "subjects" in attr_dict.get("class", ""):
            self._in_subject = True

    def handle_endtag(self, tag):
        if tag == "blockquote" and self._in_abstract:
            self._in_abstract = False
            raw = "".join(self._abstract_chunks)
            raw = re.sub(r"^\s*Title:\s*", "", raw, flags=re.I)
            raw = re.sub(r"^\s*Abstract:\s*", "", raw, flags=re.I)
            self.abstract = re.sub(r"\s+", " ", raw).strip()
        elif tag == "td" and self._in_subject:
            self._in_subject = False
            raw = "".join(self._subject_chunks)
            matches = re.findall(r"\(([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)\)", raw)
            if not matches:
                matches = re.findall(r"\b([a-z\-]+(?:\.[A-Z\-]+))\b", raw)
            self.categories = matches

    def handle_data(self, data):
        if self._in_abstract:
            self._abstract_chunks.append(data)
        elif self._in_subject:
            self._subject_chunks.append(data)


class TextExtractor:
    """Extracts text and figures from PDFs and arXiv papers."""

    def __init__(self):
        self._last_arxiv_call = 0

    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text per page. PyMuPDF for text pages, Mistral OCR for image pages."""
        if not pdf_bytes or not pdf_bytes.lstrip()[:5] == b"%PDF-":
            raise ValueError("Uploaded file is not a valid PDF (missing %PDF header)")
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        if page_count > settings.MAX_PDF_PAGES:
            doc.close()
            raise ValueError(
                f"Paper has {page_count} pages, which exceeds the {settings.MAX_PDF_PAGES}-page limit. "
                f"Please upload or submit a paper with {settings.MAX_PDF_PAGES} pages or fewer."
            )
        pages: List[PageText] = [
            PageText(page_number=i + 1, markdown=page.get_text(), source="pymupdf")
            for i, page in enumerate(doc)
        ]
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

        # Hybrid: OCR only the image/scanned pages concurrently (bounded to 3 workers)
        hybrid_figures: List[Figure] = []
        if ocr_candidates:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            target_pages = ocr_candidates[:settings.OCR_MAX_PAGE_CALLS]
            def _ocr_page(p):
                try:
                    md, figs = self._mistral_ocr_single_page(pdf_bytes, p.page_number - 1)
                    return p, md, figs
                except Exception as e:
                    log.warning(f"[WARN] OCR failed for page {p.page_number}: {e}")
                    return p, None, []

            with ThreadPoolExecutor(max_workers=min(3, len(target_pages))) as executor:
                futures = [executor.submit(_ocr_page, p) for p in target_pages]
                for f in as_completed(futures):
                    p, md, figs = f.result()
                    if md:
                        p.markdown = md
                        p.source = "mistral_ocr"
                    hybrid_figures.extend(figs)
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

    def strip_tail_sections(self, text: str) -> str:
        """Strip References / Bibliography and everything after from paper text.

        Strategy:
        - Only searches the LAST 60% of the document so inline mentions like
          "as listed in the references [1–5]" in the body are never cut.
        - Takes the LAST match in that window so a paper whose bibliography
          section has a sub-heading also titled "References" doesn't get cut
          at the sub-heading rather than the actual list.
        - Logs how many chars were saved so we can audit in production.
        - Returns the original text unchanged when no header is found.
        """
        if not text or len(text) < 500:
            return text

        # Search only in the last 60 % of the document
        search_from = max(0, int(len(text) * 0.40))
        tail = text[search_from:]

        last_match = None
        for m in _TAIL_STRIP_RE.finditer(tail):
            last_match = m  # keep updating → we want the LAST occurrence

        if last_match is not None:
            cut = search_from + last_match.start()
            stripped = text[:cut].rstrip()
            saved = len(text) - len(stripped)
            pct = int(100 * saved / len(text)) if len(text) else 0
            log.info(
                f"[TEXT] Stripped tail sections at char {cut}/{len(text)} "
                f"(−{saved} chars / −{pct}%)"
            )
            return stripped
        return text

    def _build_result(self, pages: List[PageText], figures: Optional[List[Figure]] = None) -> ExtractionResult:
        """Join per-page content into the flat text the pipeline consumes.

        References / bibliography are stripped from the flat ``text`` before it
        leaves this method — every downstream consumer (LLM calls, RAG indexer)
        automatically benefits. The per-page ``pages`` list is NOT altered so
        the frontend Parsed Text reader still shows the full paper.
        """
        text_parts = [self._strip_images(p.markdown) if p.source == "mistral_ocr" else p.markdown for p in pages]
        raw_text = "\n\n".join(p for p in text_parts if p)
        clean_text = self.strip_tail_sections(raw_text)
        return ExtractionResult(text=clean_text, figures=figures or [], pages=pages, page_count=len(pages))
    
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
                headers={
                    "User-Agent": settings.ARXIV_USER_AGENT,
                    "Accept": "application/pdf,application/octet-stream,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                }
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
        """Download PDF from arXiv with canonical ID sanitization."""
        clean_id = sanitize_arxiv_id(arxiv_id)
        url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        return self.download_pdf(url)
    
    def _fetch_arxiv_atom(self, clean_id: str) -> Optional[dict]:
        """Fetch metadata from arXiv Atom API with mirror redundancy and proper Accept headers."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/atom+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        endpoints = [
            f"https://export.arxiv.org/api/query?id_list={clean_id}",
            f"https://arxiv.org/api/query?id_list={clean_id}",
            f"http://export.arxiv.org/api/query?id_list={clean_id}",
        ]
        for endpoint in endpoints:
            self._rate_limit_arxiv()
            try:
                resp = requests.get(endpoint, timeout=15, headers=headers)
                status = getattr(resp, "status_code", 200)
                if status == 200 or not isinstance(status, int):
                    meta = self._parse_arxiv_xml(resp.text)
                    if meta.get("arxiv_id_base"):
                        return meta
                log.warning(f"arXiv Atom query {endpoint} returned status {resp.status_code}")
            except Exception as e:
                log.warning(f"arXiv Atom query {endpoint} failed: {e}")
        return None

    def _fetch_arxiv_html(self, clean_id: str) -> Optional[dict]:
        """Fallback: Scrape canonical Highwire Press metadata from https://arxiv.org/abs/{clean_id}."""
        requested_base = re.sub(r'v\d+$', '', clean_id)
        url = f"https://arxiv.org/abs/{clean_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = requests.get(url, timeout=15, headers=headers)
            if resp.status_code == 404:
                raise ValueError(
                    f"arXiv ID '{clean_id}' does not match any known paper. "
                    "Check the ID for typos and try again."
                )
            if resp.status_code == 200:
                parser = _ArxivAbsParser()
                parser.feed(resp.text)
                if parser.title:
                    cleaned_authors = []
                    for a in parser.authors:
                        if "," in a:
                            parts = [p.strip() for p in a.split(",", 1)]
                            cleaned_authors.append(f"{parts[1]} {parts[0]}")
                        else:
                            cleaned_authors.append(a)
                    return {
                        "title": parser.title,
                        "authors": cleaned_authors if cleaned_authors else parser.authors,
                        "abstract": parser.abstract,
                        "categories": parser.categories,
                        "arxiv_id": clean_id,
                        "arxiv_id_base": requested_base,
                    }
        except ValueError:
            raise
        except Exception as e:
            log.warning(f"arXiv abs HTML fetch failed for {clean_id}: {e}")
        return None

    def get_arxiv_metadata(self, arxiv_id: str) -> dict:
        """Fetch paper metadata from arXiv API or HTML abstract page with sanitized ID.

        Uses a resilient multi-tier fallback:
        1. Query export.arxiv.org/api/query (Atom 1.0 XML) with RFC-compliant headers.
        2. Failover to mirrors: arxiv.org/api/query and http://export.arxiv.org/api/query.
        3. Failover to Highwire Press HTML metadata scraper on arxiv.org/abs/{clean_id}.

        Raises:
            ValueError: if arXiv returns no entry (nonexistent ID), if the
                returned entry's canonical ID differs from the requested one,
                or if no usable title is present.
        """
        clean_id = sanitize_arxiv_id(arxiv_id)
        requested_base = re.sub(r'v\d+$', '', clean_id)
        
        # Tier 1 & 2: Atom XML API (export.arxiv.org / arxiv.org mirrors)
        meta = self._fetch_arxiv_atom(clean_id)
        
        # Tier 3: HTML abstract page fallback (bypasses 406/429/WAF issues on export.arxiv.org)
        if not meta or not meta.get("title"):
            meta = self._fetch_arxiv_html(clean_id)

        if not meta or not meta.get("arxiv_id_base"):
            raise ValueError(
                f"arXiv ID '{clean_id}' does not match any known paper. "
                "Check the ID for typos and try again."
            )
        if meta["arxiv_id_base"] != requested_base:
            raise ValueError(
                f"arXiv returned a different paper ({meta['arxiv_id_base']}) "
                f"than requested ({requested_base}). Refusing to continue."
            )
        if not meta.get("title"):
            raise ValueError(
                f"arXiv returned no title for '{clean_id}'. "
                "The paper may be unavailable; try again later."
            )
        return meta
    
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
            return {"title": "", "authors": [], "abstract": "", "categories": [], "arxiv_id": "", "arxiv_id_base": ""}

        id_el = entry.find("atom:id", ns)
        canonical_id = ""
        if id_el is not None and id_el.text:
            # Entry id looks like http://arxiv.org/abs/1512.03386v2
            id_match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?$', id_el.text.strip())
            if id_match:
                canonical_id = id_match.group(1)

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
            "categories": categories,
            "arxiv_id": canonical_id,
            "arxiv_id_base": canonical_id,
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
                "title": res.get("title") or "Research Paper",
                "authors": res.get("authors") or [],
                "abstract": res.get("abstract") or "",
                "published": str(res.get("published_year") or ""),
                "categories": res.get("categories") or []
            }
        except Exception as e:
            log.warning(f"[WARN] LLM metadata extraction failed: {e}")
            lines = [l.strip() for l in first_page.split("\n") if len(l.strip()) > 5]
            guessed_title = lines[0] if lines else "Research Paper"
            return {
                "title": guessed_title,
                "authors": [],
                "abstract": "",
                "published": "",
                "categories": []
            }
    
    # ── LLM-based architecture section detection (Fix #2 + Fast-Span) ──────────
    ARCH_SECTION_PROMPT = """You are an expert research-paper parser. Given the full text of a paper, locate the architecture / method / model section that describes the paper's PROPOSED model components, layers, blocks, and their connections.

CRITICAL RULES:
1. If the paper is a SURVEY, REVIEW, or OVERVIEW paper summarizing existing works or third-party models, you MUST return found=false.
2. If the paper has NO single unified proposed architecture, return found=false.
3. ONLY return found=true if the authors propose a novel, specific model architecture.

Instructions:
- When found=true, identify:
  - start_heading: The exact heading where the proposed architecture begins (e.g. "3 Model Architecture", "2 Methodology").
  - end_heading: The heading where the architecture description ends and training/experiments begin (e.g. "4 Experiments", "4 Training Setup").
- Confidence: 0.0-1.0 — how confidently you located a real architecture section.

Output strictly valid JSON:
{{"found": true|false, "start_heading": "exact heading text or empty string", "end_heading": "exact heading text or empty string", "confidence": 0.0-1.0, "reason": "one-sentence justification, include the heading matched if found"}}

Full paper text (truncated to {max_chars} chars for this call):
---
{text}
---"""

    def find_architecture_section(self, text: str) -> str | None:
        """LLM-based architecture/method section locator (Fix #2 + Fast-Span).

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
        start_heading = (data.get("start_heading") or "").strip()
        end_heading = (data.get("end_heading") or "").strip()
        section_text = (data.get("section_text") or "").strip()
        reason = data.get("reason") or ""

        # Explicit failure checks
        if not found:
            log.warning(f"[ARCH_SECTION] LLM could not locate architecture section: {reason}")
            return None
        if confidence < 0.5:
            log.warning(f"[ARCH_SECTION] Low confidence {confidence:.2f}: {reason} — treating as not found")
            return None

        # High-performance in-memory slicing via heading coordinates
        if start_heading:
            start_pos = text.lower().find(start_heading.lower())
            if start_pos == -1:
                # Fuzzy search using primary keywords
                words = [w for w in re.split(r'\W+', start_heading) if len(w) > 3]
                if words:
                    m = re.search(re.escape(words[0]), text, re.IGNORECASE)
                    start_pos = m.start() if m else -1

            if start_pos != -1:
                end_pos = -1
                if end_heading:
                    end_pos = text.lower().find(end_heading.lower(), start_pos + 500)
                if end_pos == -1 or end_pos <= start_pos:
                    end_pos = min(len(text), start_pos + ceiling)
                else:
                    end_pos = min(end_pos, start_pos + ceiling)

                section_text = text[start_pos:end_pos].strip()

        if len(section_text) < 200:
            log.warning(f"[ARCH_SECTION] Section too short ({len(section_text)} chars), confidence {confidence:.2f}: {reason}")
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
