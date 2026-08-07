from __future__ import annotations

import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import assignment.audit_log as audit_log_module
from assignment.audit_log import AuditLogPlugin


def test_record_input_generates_request_id_and_records_start(monkeypatch):
    plugin = AuditLogPlugin()
    monkeypatch.setattr(audit_log_module, "utc_now_iso", lambda: "2026-08-07T10:00:00+00:00")
    monkeypatch.setattr(audit_log_module.time, "time", lambda: 1000.0)

    request_id = plugin.record_input(user_id="user-1", text="password=admin123")

    assert isinstance(request_id, str)
    assert plugin._requests[request_id]["user_id"] == "user-1"
    assert plugin._requests[request_id]["started_at"] == "2026-08-07T10:00:00+00:00"
    assert "[REDACTED]" in plugin._requests[request_id]["input"]


def test_record_output_joins_request_and_computes_latency(monkeypatch):
    plugin = AuditLogPlugin()
    times = iter([1000.0, 1002.5])
    stamps = iter(["2026-08-07T10:00:00+00:00", "2026-08-07T10:00:02+00:00"])

    monkeypatch.setattr(audit_log_module.time, "time", lambda: next(times))
    monkeypatch.setattr(audit_log_module, "utc_now_iso", lambda: next(stamps))

    request_id = plugin.record_input(user_id="user-1", text="transfer request")
    record = plugin.record_output(
        user_id="user-1",
        text="approved transfer",
        blocked=False,
        layer=None,
        request_id=request_id,
        judge_decision={"verdict": "PASS"},
    )

    assert record["request_id"] == request_id
    assert record["input"] == "transfer request"
    assert record["output_preview"] == "approved transfer"
    assert record["ended_at"] == "2026-08-07T10:00:02+00:00"
    assert record["latency_ms"] == 2500.0
    assert record["judge_decision"] == {"verdict": "PASS"}


def test_export_json_creates_parent_and_writes_redacted_utf8(tmp_path, monkeypatch):
    plugin = AuditLogPlugin()
    times = iter([2000.0, 2001.0])
    stamps = iter(["2026-08-07T11:00:00+00:00", "2026-08-07T11:00:01+00:00"])

    monkeypatch.setattr(audit_log_module.time, "time", lambda: next(times))
    monkeypatch.setattr(audit_log_module, "utc_now_iso", lambda: next(stamps))

    request_id = plugin.record_input(
        user_id="user-2",
        text="Lien he toi qua email test@vinbank.com va API sk-secret-demo-12345678",
    )
    plugin.record_output(
        user_id="user-2",
        text="Tra loi co password=Secret!99",
        blocked=True,
        layer="output_guardrail",
        request_id=request_id,
        hitl_decision={"action": "queue_review"},
    )

    output_path = tmp_path / "outputs" / "audit_log.json"
    returned_path = plugin.export_json(str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert returned_path == output_path
    assert output_path.exists()
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["blocked"] is True
    assert data[0]["blocked_by"] == "output_guardrail"
    assert data[0]["hitl_decision"] == {"action": "queue_review"}
    assert "[REDACTED]" in data[0]["input"]
    assert "[REDACTED]" in data[0]["output_preview"]
    assert "sk-secret-demo-12345678" not in output_path.read_text(encoding="utf-8")
