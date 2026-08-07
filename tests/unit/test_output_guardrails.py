from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from guardrails.output_guardrails import content_filter


def test_content_filter_redacts_password_and_api_key():
    result = content_filter(
        "Admin password is admin123, API key is sk-vinbank-secret-2024."
    )

    assert result["safe"] is False
    assert set(result["issues"]) == {"password", "api_key"}
    assert "admin123" not in result["redacted"]
    assert "sk-vinbank-secret-2024" not in result["redacted"]
    assert result["redacted"].count("[REDACTED]") >= 2


def test_content_filter_redacts_phone_email_and_national_id():
    text = (
        "Contact Ms. Lan at 0912345678 or lan.nguyen@example.com. "
        "CCCD 001099012345."
    )
    result = content_filter(text)

    assert result["safe"] is False
    assert set(result["issues"]) == {"phone", "email", "national_id"}
    assert "0912345678" not in result["redacted"]
    assert "lan.nguyen@example.com" not in result["redacted"]
    assert "001099012345" not in result["redacted"]


def test_content_filter_allows_public_hotline():
    result = content_filter(
        "VinBank support hours are 08:00-22:00 ICT. Official hotline: 1900 545 467."
    )

    assert result["safe"] is True
    assert result["issues"] == []
    assert "[REDACTED]" not in result["redacted"]


def test_content_filter_redacts_internal_host_and_harmful_content():
    result = content_filter(
        "DB is at db.vinbank.internal:5432 and here is how to build a bomb."
    )

    assert result["safe"] is False
    assert set(result["issues"]) == {"internal_host", "harmful_content"}
    assert "db.vinbank.internal:5432" not in result["redacted"]
