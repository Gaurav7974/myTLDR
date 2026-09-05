"""Unit tests for academic research paper validator."""

from unittest.mock import MagicMock, patch
import pytest

from app.services.paper_validator import paper_validator


ACADEMIC_PAPER_TEXT = """
Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.
Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable.

1 Introduction
Recurrent neural networks, long short-term memory [1] and gated recurrent [2] neural networks have been firmly established as state of the art approaches in sequence modeling.
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks [3, 4].

3 Model Architecture
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

References
[1] Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. Neural computation, 1997.
[2] Kyunghyun Cho et al. Learning phrase representations using rnn encoder-decoder. In EMNLP, 2014.
"""

INVOICE_TEXT = """
TAX INVOICE
Acme Cloud Services Inc.
Invoice Number: INV-2024-99881
Billing Address: 123 Enterprise Way, Suite 400
Date: September 1, 2024
Due Date: September 15, 2024

Items:
1. Cloud Server Instances - 720 Hours - $360.00
2. Bandwidth Outbound - 500 GB - $45.00
3. PostgreSQL Managed Database - $120.00

Subtotal: $525.00
Tax (10%): $52.50
Total Due: $577.50
Payment Method: Credit Card (ending in 4242)
Thank you for your business!
"""

RESUME_TEXT = """
John Doe
Senior Machine Learning Engineer | San Francisco, CA | john@example.com

Professional Summary
Passionate machine learning engineer with 6+ years of experience building production recommendation systems and distributed ML pipelines.

Work Experience
Senior ML Engineer - AI Corp (2021 - Present)
- Designed and deployed large-scale transformer recommendation models serving 50M daily active users.
- Optimized model inference latency by 45% using TensorRT and quantization.

Software Engineer - Tech Solutions (2018 - 2021)
- Built automated ETL pipelines processing 10TB of event data daily.

Education
B.S. in Computer Science - University of California, Berkeley (2014 - 2018)

Skills Summary
Python, PyTorch, TensorFlow, Kubernetes, Docker, SQL, Distributed Systems
"""

STORY_TEXT = """
Chapter 1: The Mysterious Forest
The wind whispered through the ancient pines as Lily stepped onto the winding gravel path.
She had lived in the village of Oakhaven for fifteen years, but she had never dared to venture
past the crumbling stone bridge at twilight. The local elders always warned of strange lights
and phantom melodies that played when the moon was full. Her heart pounded against her ribs,
yet her curiosity drove her forward into the deepening shadows.
"""


def test_validator_rejects_empty_or_short_text():
    is_paper, reason = paper_validator.validate_research_paper("")
    assert is_paper is False
    assert "short" in reason.lower()

    is_paper, reason = paper_validator.validate_research_paper("Hello world, this is a short test document.")
    assert is_paper is False
    assert "short" in reason.lower()


def test_validator_rejects_invoice_via_heuristics():
    h_result, h_reason = paper_validator.check_heuristics(INVOICE_TEXT)
    assert h_result is False
    assert "non-academic" in h_reason.lower() or "invoice" in h_reason.lower()

    # Full validation check
    is_paper, reason = paper_validator.validate_research_paper(INVOICE_TEXT)
    assert is_paper is False


def test_validator_rejects_resume_via_heuristics():
    h_result, h_reason = paper_validator.check_heuristics(RESUME_TEXT)
    assert h_result is False
    assert "non-academic" in h_reason.lower() or "resume" in h_reason.lower()

    # Full validation check
    is_paper, reason = paper_validator.validate_research_paper(RESUME_TEXT)
    assert is_paper is False


def test_validator_accepts_academic_paper_via_heuristics():
    h_result, h_reason = paper_validator.check_heuristics(ACADEMIC_PAPER_TEXT)
    assert h_result is True

    # Fallback path when LLM unavailable
    with patch("app.services.llm.groq_client.extract_json", side_effect=RuntimeError("LLM unavailable")):
        is_paper, reason = paper_validator.validate_research_paper(ACADEMIC_PAPER_TEXT)
        assert is_paper is True


def test_validator_with_mock_llm_accepts_paper():
    mock_response = {
        "is_research_paper": True,
        "confidence": 0.98,
        "document_type": "research_paper",
        "reason": "Clear peer-reviewed academic paper with Abstract, Introduction, and References."
    }
    with patch("app.services.llm.groq_client.extract_json", return_value=mock_response):
        is_paper, reason = paper_validator.validate_research_paper(ACADEMIC_PAPER_TEXT)
        assert is_paper is True
        assert "Abstract" in reason


def test_validator_with_mock_llm_rejects_non_paper():
    mock_response = {
        "is_research_paper": False,
        "confidence": 0.95,
        "document_type": "story",
        "reason": "Narrative creative fiction, not a research paper."
    }
    with patch("app.services.llm.groq_client.extract_json", return_value=mock_response):
        is_paper, reason = paper_validator.validate_research_paper(STORY_TEXT)
        assert is_paper is False
        assert "not a research paper" in reason.lower() or "fiction" in reason.lower()
