from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import guardrails.output_guardrails as output_guardrails


def test_parse_judge_verdict_accepts_valid_json():
    result = output_guardrails._parse_judge_verdict(
        """
        {
          "safety": 5,
          "relevance": 4,
          "accuracy": 4,
          "tone": 5,
          "verdict": "PASS",
          "reason": "customer-safe banking answer"
        }
        """
    )

    assert result["safe"] is True
    assert result["verdict"] == "PASS"
    assert result["scores"]["safety"] == 5
    assert result["judge_failed"] is False


def test_parse_judge_verdict_rejects_missing_field():
    with pytest.raises(ValueError, match="judge_missing_fields"):
        output_guardrails._parse_judge_verdict(
            '{"safety":5,"relevance":5,"accuracy":5,"verdict":"PASS","reason":"ok"}'
        )


def test_parse_judge_verdict_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="judge_invalid_score:safety"):
        output_guardrails._parse_judge_verdict(
            '{"safety":6,"relevance":5,"accuracy":5,"tone":5,"verdict":"PASS","reason":"ok"}'
        )


def test_llm_safety_check_returns_pass_with_valid_judge_json(monkeypatch):
    output_guardrails.judge_runner = object()

    async def fake_chat(agent, runner, prompt):
        return (
            '{"safety":5,"relevance":5,"accuracy":5,"tone":5,"verdict":"PASS","reason":"ok"}',
            None,
        )

    monkeypatch.setattr(output_guardrails, "chat_with_agent", fake_chat)
    result = asyncio.run(
        output_guardrails.llm_safety_check("The 12-month savings rate is 4.25%.")
    )

    assert result["safe"] is True
    assert result["verdict"] == "PASS"
    assert result["judge_failed"] is False


def test_llm_safety_check_fail_closed_on_invalid_json(monkeypatch):
    output_guardrails.judge_runner = object()

    async def fake_chat(agent, runner, prompt):
        return ("not json", None)

    monkeypatch.setattr(output_guardrails, "chat_with_agent", fake_chat)
    result = asyncio.run(output_guardrails.llm_safety_check("leak"))

    assert result["safe"] is False
    assert result["verdict"] == "BLOCK"
    assert result["judge_failed"] is True
    assert result["reason"] == "judge_json_not_found"


def test_llm_safety_check_fail_closed_on_judge_exception(monkeypatch):
    output_guardrails.judge_runner = object()

    async def fake_chat(agent, runner, prompt):
        raise RuntimeError("timeout")

    monkeypatch.setattr(output_guardrails, "chat_with_agent", fake_chat)
    result = asyncio.run(output_guardrails.llm_safety_check("leak"))

    assert result["safe"] is False
    assert result["verdict"] == "BLOCK"
    assert result["judge_failed"] is True
    assert result["reason"] == "judge_error:RuntimeError"
