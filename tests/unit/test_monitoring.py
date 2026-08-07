from __future__ import annotations

import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import assignment.monitoring as monitoring_module
from assignment.monitoring import MonitoringAlert


def test_snapshot_avoids_division_by_zero(monkeypatch):
    monitor = MonitoringAlert()
    monkeypatch.setattr(
        monitoring_module,
        "datetime",
        type(
            "FrozenDateTime",
            (),
            {"now": staticmethod(lambda tz=None: __import__("datetime").datetime(2026, 8, 7, 12, 0, 0, tzinfo=tz))},
        ),
    )

    snapshot = monitor.snapshot()

    assert snapshot["total_requests"] == 0
    assert snapshot["block_rate"] == 0.0
    assert snapshot["judge_fail_rate"] == 0.0
    assert snapshot["alerts"] == []
    assert "snapshot_at" in snapshot


def test_check_metrics_emits_alerts_for_all_thresholds():
    monitor = MonitoringAlert(
        block_rate_threshold=0.5,
        rate_limit_hit_threshold=5,
        judge_fail_rate_threshold=0.3,
        total_requests=10,
        blocked_requests=6,
        rate_limit_hits=7,
        judge_checks=10,
        judge_fails=4,
    )

    alerts = monitor.check_metrics()
    metrics = {alert.metric for alert in alerts}

    assert metrics == {"block_rate", "rate_limit_hits", "judge_fail_rate"}


def test_export_json_writes_metrics_file(tmp_path):
    monitor = MonitoringAlert(
        total_requests=4,
        blocked_requests=1,
        rate_limit_hits=0,
        judge_checks=2,
        judge_fails=0,
    )

    output_path = tmp_path / "outputs" / "metrics.json"
    returned_path = monitor.export_json(str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert returned_path == output_path
    assert output_path.exists()
    assert data["total_requests"] == 4
    assert data["blocked_requests"] == 1
    assert data["block_rate"] == 0.25
    assert data["thresholds"]["block_rate_threshold"] == 0.5
    assert isinstance(data["alerts"], list)
    assert "snapshot_at" in data
