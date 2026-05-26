"""Shared HTTP/logging helpers."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib import error


def describe_http_error(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            return f"HTTPError {exc.code} {exc.reason}: {body}"
        return f"HTTPError {exc.code} {exc.reason}"
    return str(exc)


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def payload_summary(payload: Any) -> str:
    if isinstance(payload, dict):
        parts = []
        for key, value in payload.items():
            if key.lower() in {"api_key", "authorization", "token"}:
                parts.append(f"{key}=<redacted>")
            elif key == "messages" and isinstance(value, list):
                message_shapes = [
                    {
                        "role": item.get("role"),
                        "content_chars": len(str(item.get("content", ""))),
                    }
                    for item in value
                    if isinstance(item, dict)
                ]
                parts.append(f"messages={message_shapes}")
            elif key == "input" and isinstance(value, list):
                parts.append(f"input_count={len(value)}, input_chars={[len(str(item)) for item in value]}")
            else:
                parts.append(f"{key}={short_repr(value)}")
        return "{" + ", ".join(parts) + "}"
    return short_repr(payload)


def response_preview(text: str, limit: int = 500) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def response_summary(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, list):
                parts.append(f"{key}=list(len={len(item)})")
            elif isinstance(item, dict):
                parts.append(f"{key}=dict(keys={sorted(str(k) for k in item.keys())[:12]})")
            else:
                parts.append(f"{key}={short_repr(item, 120)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        return f"list(len={len(value)}, first={short_repr(value[0], 160) if value else 'None'})"
    return short_repr(value)


def short_repr(value: Any, limit: int = 240) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text
