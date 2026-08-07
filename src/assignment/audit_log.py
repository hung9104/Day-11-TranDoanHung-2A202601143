"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from uuid import uuid4


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._requests: dict[str, dict] = {}

    def _redact_text(self, text: str, limit: int = 240) -> str:
        if not isinstance(text, str):
            return ""

        redacted = text
        patterns = (
            r"\bsk-[A-Za-z0-9-]{8,}\b",
            r"\bpassword\s*[:=]\s*[^\s,;]+",
            r"\bpassword\s+is\s+[^\s,;]+",
            r"\badmin123\b",
            r"\bdb\.vinbank\.internal(?::\d{2,5})?\b",
            r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
            r"(?<!\d)0\d{9,10}(?!\d)",
            r"(?<!\d)(?:\d{9}|\d{12})(?!\d)",
        )
        for pattern in patterns:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

        compact = " ".join(redacted.split())
        if len(compact) > limit:
            return compact[: limit - 3] + "..."
        return compact

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        resolved_request_id = request_id or str(uuid4())
        started_at = utc_now_iso()
        started_ts = time.time()

        self._open[resolved_request_id] = started_ts
        self._requests[resolved_request_id] = {
            "request_id": resolved_request_id,
            "user_id": user_id,
            "input": self._redact_text(text),
            "started_at": started_at,
        }
        return resolved_request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        judge_decision: dict | str | None = None,
        hitl_decision: dict | str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        resolved_request_id = request_id or str(uuid4())
        ended_at = utc_now_iso()
        ended_ts = time.time()
        started_ts = self._open.pop(resolved_request_id, ended_ts)
        base_record = self._requests.pop(
            resolved_request_id,
            {
                "request_id": resolved_request_id,
                "user_id": user_id,
                "input": "",
                "started_at": ended_at,
            },
        )

        record = {
            **base_record,
            "user_id": user_id,
            "output_preview": self._redact_text(text),
            "blocked": blocked,
            "blocked_by": layer,
            "ended_at": ended_at,
            "latency_ms": round(max(0.0, ended_ts - started_ts) * 1000, 2),
        }
        if judge_decision is not None:
            record["judge_decision"] = judge_decision
        if hitl_decision is not None:
            record["hitl_decision"] = hitl_decision

        self.logs.append(record)
        return record

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
