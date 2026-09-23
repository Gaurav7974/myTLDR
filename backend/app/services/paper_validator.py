"""Academic research paper validation service."""

import re
from typing import Tuple, Optional
from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

# Fast heuristic patterns for non-academic documents
NON_PAPER_PATTERNS = [
    re.compile(r"(?i)\b(invoice\s+(?:number|no|#)|balance\s+due|amount\s+due|total\s+due|billing\s+address|payment\s+(?:method|terms)|tax\s+invoice)\b"),
    re.compile(r"(?i)\b(curriculum\s+vitae|work\s+experience|employment\s+history|professional\s+summary|education\s+and\s+qualifications|skills\s+summary)\b"),
    re.compile(r"(?i)\b(purchase\s+order|order\s+confirmation|shipping\s+address|receipt\s+no|cashier|subtotal)\b"),
]

# Academic structure indicators
ACADEMIC_SECTION_PATTERNS = [
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?abstract\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?introduction\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:methodology|methods|proposed\s+(?:method|approach|model))\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:references|bibliography|works\s+cited)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?(?:experiments|experimental\s+results|evaluation|results\s+and\s+discussion)\b"),
    re.compile(r"(?im)^\s*(?:\d+\.?\s*)?conclusion\b"),
]

CITATION_PATTERNS = [
    re.compile(r"\[\d+(?:,\s*\d+)*\]"),  # [1], [1, 2]
    re.compile(r"\([A-Z][a-zA-Z\s]+(?:et\s+al\.)?,?\s*\d{4}\)"),  # (Vaswani et al., 2017)
    re.compile(r"(?i)\b(arxiv:\d{4}\.\d{4,5}|doi:10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b"),
]


class PaperValidator:
    """Validates whether an uploaded document is a genuine academic or scientific research paper."""

    PAPER_CLASSIFIER_PROMPT = """You are an academic paper classifier. Analyze the following document text and determine whether it is a legitimate academic or scientific research paper (e.g. conference paper, journal article, preprint, technical report, workshop paper, thesis).

Reject documents that are NOT academic research papers:
- Invoices, receipts, billing statements, financial reports, spreadsheets
- Resumes, CVs, job applications
- Product manuals, brochures, advertising, pitch decks
- Legal contracts, court filings, government forms
- News articles, blogs, recipes, stories/fiction, correspondence

Document sample:
--- BEGIN DOCUMENT SAMPLE ---
{sample}
--- END DOCUMENT SAMPLE ---

Respond strictly with valid JSON:
{{
  "is_research_paper": true,
  "confidence": 0.95,
  "document_type": "research_paper",
  "reason": "Clear explanation of why it is or is not an academic research paper"
}}"""

    def check_heuristics(self, text: str) -> Tuple[Optional[bool], str]:
        """Fast structural pre-check.
        Returns:
            (False, reason) if strongly non-academic (e.g. invoice or resume)
            (True, reason) if overwhelming academic markers present
            (None, "") if inconclusive and needs LLM evaluation
        """
        if not text or len(text.strip()) < 100:
            return False, "Text is too short to be an academic research paper."

        # Check explicit non-paper red flags
        for pat in NON_PAPER_PATTERNS:
            if pat.search(text[:3000]):
                return False, "Document contains non-academic markers (invoice, resume, or commercial form)."

        # Count academic sections
        matches = sum(1 for pat in ACADEMIC_SECTION_PATTERNS if pat.search(text))
        has_citations = any(pat.search(text) for pat in CITATION_PATTERNS)

        if matches >= 3 and has_citations:
            return True, "Strong academic sections and citation formatting identified."

        if matches == 0 and not has_citations and len(text) > 1000:
            return False, "Document lacks academic structure (no Abstract, Introduction, Methods, or References)."

        return None, ""

    def validate_research_paper(self, text: str) -> Tuple[bool, str]:
        """Validate whether the text belongs to an academic research paper.
        Uses heuristics first, then an LLM classifier gate.
        """
        if not text or len(text.strip()) < 100:
            return False, "Document text is empty or too short."

        # 1. Quick heuristic pre-filter
        h_result, h_reason = self.check_heuristics(text)
        if h_result is False:
            log.info(f"[PAPER_VALIDATOR] Rejected by heuristics: {h_reason}")
            return False, h_reason

        # 2. LLM Classifier Gate (for ambiguous or borderline documents)
        sample = text[:3500].strip()
        try:
            from app.services.llm import groq_client
            prompt = self.PAPER_CLASSIFIER_PROMPT.format(sample=sample)
            data = groq_client.extract_json(
                messages=[
                    {"role": "system", "content": "You are a precise academic research paper classifier. Output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=settings.MISTRAL_LLM_MODEL
            )
            is_paper = bool(data.get("is_research_paper", False))
            confidence = float(data.get("confidence", 0.0) or 0.0)
            doc_type = data.get("document_type", "unknown")
            reason = data.get("reason", "")

            log.info(f"[PAPER_VALIDATOR] LLM classification: is_paper={is_paper}, conf={confidence:.2f}, type={doc_type}, reason={reason}")

            if is_paper and confidence >= 0.6:
                return True, reason or "Verified academic research paper."

            reject_msg = reason or f"Document identified as {doc_type} rather than an academic research paper."
            return False, reject_msg

        except Exception as e:
            log.warning(f"[PAPER_VALIDATOR] LLM classification failed: {e}. Falling back to heuristics.")
            # Fallback to heuristics if LLM fails
            if h_result is True:
                return True, h_reason
            
            # Stricter fallback check
            has_abstract = bool(re.search(r"(?i)\babstract\b", text[:3000]))
            has_refs = bool(re.search(r"(?i)\b(references|bibliography)\b", text))
            has_cites = any(pat.search(text) for pat in CITATION_PATTERNS)
            if has_abstract and (has_refs or has_cites):
                return True, "Academic paper verified via abstract and citation references."
            
            return False, "Document does not appear to have an academic paper structure (missing Abstract or References)."


paper_validator = PaperValidator()
