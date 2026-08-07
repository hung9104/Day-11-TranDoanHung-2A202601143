from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import build_observability, build_production_plugins
from assignment.rate_limiter import RateLimitPlugin
from guardrails.input_guardrails import InputGuardrailPlugin
from guardrails.output_guardrails import OutputGuardrailPlugin


def test_build_production_plugins_returns_expected_order():
    plugins = build_production_plugins(
        max_requests=12,
        window_seconds=45,
        use_llm_judge=False,
    )

    assert len(plugins) == 3
    assert isinstance(plugins[0], RateLimitPlugin)
    assert isinstance(plugins[1], InputGuardrailPlugin)
    assert isinstance(plugins[2], OutputGuardrailPlugin)
    assert plugins[0].max_requests == 12
    assert plugins[0].window_seconds == 45
    assert plugins[2].use_llm_judge is False


def test_build_observability_returns_audit_and_monitoring():
    audit_log, monitoring = build_observability()

    assert isinstance(audit_log, AuditLogPlugin)
    assert isinstance(monitoring, MonitoringAlert)
