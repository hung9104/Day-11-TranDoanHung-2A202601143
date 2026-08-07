from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from google.genai import types


SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import assignment.rate_limiter as rate_limiter_module
from assignment.rate_limiter import RateLimitPlugin


def _message():
    return types.Content(role="user", parts=[types.Part.from_text(text="hello")])


def test_allows_first_ten_and_blocks_next_five(monkeypatch):
    plugin = RateLimitPlugin(max_requests=10, window_seconds=60)
    invocation_context = SimpleNamespace(user_id="user-1")
    current_time = {"value": 1000.0}

    monkeypatch.setattr(rate_limiter_module.time, "time", lambda: current_time["value"])

    results = []
    for _ in range(15):
        results.append(
            asyncio.run(
                plugin.on_user_message_callback(
                    invocation_context=invocation_context,
                    user_message=_message(),
                )
            )
        )

    assert all(result is None for result in results[:10])
    assert all(result is not None for result in results[10:])
    assert plugin.total_count == 15
    assert plugin.blocked_count == 5
    assert len(plugin.user_windows["user-1"]) == 10


def test_other_user_has_separate_window(monkeypatch):
    plugin = RateLimitPlugin(max_requests=10, window_seconds=60)
    current_time = {"value": 2000.0}
    monkeypatch.setattr(rate_limiter_module.time, "time", lambda: current_time["value"])

    for _ in range(10):
        result = asyncio.run(
            plugin.on_user_message_callback(
                invocation_context=SimpleNamespace(user_id="user-1"),
                user_message=_message(),
            )
        )
        assert result is None

    blocked = asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id="user-1"),
            user_message=_message(),
        )
    )
    allowed_other = asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id="user-2"),
            user_message=_message(),
        )
    )

    assert blocked is not None
    assert allowed_other is None
    assert plugin.blocked_count == 1


def test_request_allowed_after_window_expires(monkeypatch):
    plugin = RateLimitPlugin(max_requests=2, window_seconds=60)
    invocation_context = SimpleNamespace(user_id="user-1")
    current_time = {"value": 3000.0}
    monkeypatch.setattr(rate_limiter_module.time, "time", lambda: current_time["value"])

    assert asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=invocation_context,
            user_message=_message(),
        )
    ) is None
    assert asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=invocation_context,
            user_message=_message(),
        )
    ) is None

    blocked = asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=invocation_context,
            user_message=_message(),
        )
    )
    assert blocked is not None

    current_time["value"] = 3061.0
    allowed_again = asyncio.run(
        plugin.on_user_message_callback(
            invocation_context=invocation_context,
            user_message=_message(),
        )
    )
    assert allowed_again is None
