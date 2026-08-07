from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from google.genai import types


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import guardrails.output_guardrails as output_guardrails


def _response(text: str):
    class DummyResponse:
        def __init__(self, text_value: str):
            self.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=text_value)],
            )

    return DummyResponse(text)


def test_output_plugin_redacts_but_allows_when_judge_passes(monkeypatch):
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=True)

    async def fake_judge(text: str):
        return {
            "safe": True,
            "verdict": "PASS",
            "reason": "redacted output is customer-safe",
            "scores": {"safety": 5, "relevance": 5, "accuracy": 5, "tone": 5},
            "judge_failed": False,
            "raw_verdict": "",
        }

    monkeypatch.setattr(output_guardrails, "llm_safety_check", fake_judge)
    llm_response = _response("Admin password is admin123, API key is sk-vinbank-secret-2024.")

    result = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=llm_response)
    )
    text = result.content.parts[0].text

    assert "[REDACTED]" in text
    assert "admin123" not in text
    assert plugin.redacted_count == 1
    assert plugin.blocked_count == 0
    assert plugin.total_count == 1


def test_output_plugin_blocks_when_judge_rejects(monkeypatch):
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=True)

    async def fake_judge(text: str):
        return {
            "safe": False,
            "verdict": "BLOCK",
            "reason": "hallucination",
            "judge_failed": False,
            "raw_verdict": "",
        }

    monkeypatch.setattr(output_guardrails, "llm_safety_check", fake_judge)
    llm_response = _response("The 12-month savings rate is 5.5% per year.")

    result = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=llm_response)
    )
    text = result.content.parts[0].text

    assert "normal VinBank banking question" in text
    assert plugin.redacted_count == 0
    assert plugin.blocked_count == 1
    assert plugin.total_count == 1


def test_output_plugin_passes_through_when_filter_and_judge_allow(monkeypatch):
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=True)

    async def fake_judge(text: str):
        return {
            "safe": True,
            "verdict": "PASS",
            "reason": "safe",
            "scores": {"safety": 5, "relevance": 5, "accuracy": 5, "tone": 5},
            "judge_failed": False,
            "raw_verdict": "",
        }

    monkeypatch.setattr(output_guardrails, "llm_safety_check", fake_judge)
    llm_response = _response("The 12-month savings rate is 4.25% per year.")

    result = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=llm_response)
    )

    assert result.content.parts[0].text == "The 12-month savings rate is 4.25% per year."
    assert plugin.redacted_count == 0
    assert plugin.blocked_count == 0
    assert plugin.total_count == 1
