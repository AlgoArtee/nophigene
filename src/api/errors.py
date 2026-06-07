"""Consistent API error types and response envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import jsonify


@dataclass
class APIError(Exception):
    """An expected client-facing API failure."""

    code: str
    message: str
    status_code: int = 400
    details: Any | None = None

    def to_response(self):
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        return jsonify(payload), self.status_code


def error_response(code: str, message: str, status_code: int, details: Any | None = None):
    """Build an error response outside an exception handler."""
    return APIError(code, message, status_code, details).to_response()
