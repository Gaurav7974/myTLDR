"""Text extraction from PDFs and arXiv papers."""

import re
import time
import base64
import requests
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional
from app.config import settings
from app.core.security import sanitize_arxiv_id, validate_and_sanitize_url


@dataclass
class Figure:
    """A figure/diagram extracted from the paper, with its base64 image and caption."""
    id: str
    page: int
    base64: Optional[str]
    caption: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of extracting a paper: the full text plus any extracted figures."""
    text: str
    figures: List[Figure] = field(default_factory=list)


class TextExtractor:
    """Extracts text and figures from PDFs and arXiv papers."""
    
    def __init__(self):
        self._last_arxiv_call = 0
    
    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text (and figures) from PDF bytes. Uses PyMuPDF first, falls back to Mistral OCR."""
        # Try direct text extraction first
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # If we got meaningful text, use it (no figures available from a raw text layer)
        if len(text.strip()) > 200:
            return ExtractionResult(text=text, figures=[])
        
        # No text layer — use Mistral OCR (which also returns figures)
        print("[OCR] No text layer found, calling Mistral OCR...")
        return self._mistral_ocr_pdf(pdf_bytes)
    
    def _mistral_ocr_pdf(self, pdf_bytes: bytes) -> ExtractionResult:
        """Call Mistral OCR API for image-based PDFs, returning text + figures."""
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
        result = resp.json()
        
        # Parse into text + figures
        return self._parse_ocr_response(result)
    
    def _parse_ocr_response(self, ocr_result: dict) -> ExtractionResult:
        """Parse Mistral OCR response into text and a list of figures (with base64 + caption)."""
        pages = ocr_result.get("pages", [])
        text_parts = []
        figures: List[Figure] = []
        for page in pages:
            md = page.get("markdown", "") or ""
            if md:
                text_parts.append(md)
            page_idx = page.get("index", len(figures))
            captions = self._extract_figure_captions(md)
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
        return ExtractionResult(text="\n\n".join(text_parts), figures=figures)
    
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
            print(f"[WARN] LLM metadata extraction failed: {e}")
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
        # Prefer top-level numbered sections
        patterns = [
            r"(?:\n)(\d+\.\s+(?:model\s+)?architecture)",
            r"(?:\n)(\d+\.\s+method(?:s)?)",
            r"(?:\n)(\d+\.\s+approach)",
            r"(?:\n)(\d+\.\s+proposed)",
            r"(?:\n)(\d+\.\s+network\s+architecture)",
            r"(?:\n)(\d+\.\s+model\s+design)",
            r"(?:\n)(\d+\.\s+model\s+structure)",
            r"(?:\n)(\d+\.\s+overall\s+architecture)",
        ]
        
        start = None
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                line_start = text.rfind('\n', 0, match.start()) + 1
                start = line_start
                break
        
        if start is None:
            print("[WARN] No architecture section found, using first N chars")
            return text[:settings.MAX_SECTION_CHARS]
        
        # Find next top-level numbered section
        next_section = re.search(r"\n(?=\d+\.\s+[A-Z])", text[start+200:])
        if next_section:
            end = start + 200 + next_section.start()
            section = text[start:end]
        else:
            section = text[start:start + settings.MAX_SECTION_CHARS]
        
        # Hard cap
        if len(section) > settings.MAX_SECTION_CHARS:
            section = section[:settings.MAX_SECTION_CHARS]
        
        print(f"Found architecture section ({len(section)} chars)")
        return section
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for content hashing."""
        # Strip whitespace, lowercase, remove page numbers
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
        return text


text_extractor = TextExtractor()
