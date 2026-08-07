from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from assignment.pipeline import is_egress_allowed


def test_allows_exact_https_vinbank_endpoint_with_clean_payload():
    assert is_egress_allowed(
        "https://api.vinbank.example/v1/transfers",
        "approved transfer amount 500000",
    ) is True


def test_blocks_non_https_and_fake_hosts():
    destinations = (
        "http://api.vinbank.example/v1/transfers",
        "https://evil.example/collect",
        "https://api.vinbank.example.evil.com/v1/transfers",
        "https://evil.api.vinbank.example/v1/transfers",
    )
    for destination in destinations:
        assert is_egress_allowed(destination, "approved transfer amount 500000") is False


def test_blocks_sensitive_payload_patterns():
    payloads = (
        "admin password is admin123",
        "API key sk-vinbank-secret-2024",
        "DB host db.vinbank.internal:5432",
        "contact test@vinbank.com",
        "call 0901234567",
    )
    for payload in payloads:
        assert is_egress_allowed(
            "https://api.vinbank.example/v1/transfers",
            payload,
        ) is False
