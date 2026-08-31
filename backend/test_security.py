"""Automated security verification test suite."""

import sys
from fastapi import HTTPException
from app.core.security import sanitize_arxiv_id, validate_and_sanitize_url, limiter
from app.main import app

def test_arxiv_sanitization():
    print("Testing arXiv sanitization...")
    # Valid formats
    assert sanitize_arxiv_id("1706.03762") == "1706.03762"
    assert sanitize_arxiv_id("arxiv:1706.03762v2") == "1706.03762v2"
    assert sanitize_arxiv_id("https://arxiv.org/abs/2307.09288") == "2307.09288"
    assert sanitize_arxiv_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert sanitize_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    print("  [PASS] Valid arXiv IDs sanitized correctly.")

    # Malicious / invalid inputs
    invalid_cases = [
        "../../etc/passwd",
        "1706.03762; rm -rf /",
        "<script>alert(1)</script>",
        "http://evil.com/exploit",
        "invalid_id_123"
    ]
    for case in invalid_cases:
        try:
            sanitize_arxiv_id(case)
            assert False, f"Expected rejection for: {case}"
        except HTTPException:
            pass
    print("  [PASS] Malicious/invalid arXiv inputs rejected.")


def test_ssrf_protection():
    print("Testing SSRF protection...")
    # Blocked targets (internal / private / loopback / cloud metadata)
    blocked_urls = [
        "http://localhost/admin",
        "http://127.0.0.1:8080/secrets",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://192.168.1.1/router",
        "file:///etc/passwd",
        "ftp://internal.server/file.pdf",
        "http://0.0.0.0/"
    ]
    for url in blocked_urls:
        try:
            validate_and_sanitize_url(url)
            assert False, f"Expected SSRF block for: {url}"
        except HTTPException as e:
            assert e.status_code == 400
    print("  [PASS] SSRF attacks (localhost, private subnets, cloud metadata) blocked.")

    # Valid public target
    valid_url = "https://arxiv.org/pdf/1706.03762.pdf"
    res = validate_and_sanitize_url(valid_url)
    assert res == valid_url
    print("  [PASS] Legitimate public URLs accepted.")


def test_app_limiter_registered():
    print("Testing Rate Limiter registration...")
    assert hasattr(app.state, "limiter")
    assert app.state.limiter is limiter
    print("  [PASS] SlowAPI Limiter registered on FastAPI app state.")


if __name__ == "__main__":
    try:
        test_arxiv_sanitization()
        test_ssrf_protection()
        test_app_limiter_registered()
        print("\nAll Security & Rate Limiting Tests Passed Successfully! [OK]")
    except Exception as e:
        print(f"\nTest failed: {e}", file=sys.stderr)
        sys.exit(1)
