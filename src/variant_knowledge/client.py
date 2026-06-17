"""Small requests-based HTTP client for variant knowledge connectors."""

from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_USER_AGENT = "NophiGene-DynamicVariantKB/1.0"


class KnowledgeRequestError(RuntimeError):
    """Raised when a source request cannot be completed."""


class RequestClient:
    """Requests wrapper with conservative timeout and light throttling."""

    def __init__(self, *, timeout: int = DEFAULT_TIMEOUT_SECONDS, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._last_request_at: float = 0.0

    def _throttle(self, rate_limit_per_second: float | None) -> None:
        if not rate_limit_per_second or rate_limit_per_second <= 0:
            return
        minimum_interval = 1.0 / rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        rate_limit_per_second: float | None = None,
    ) -> Any:
        self._throttle(rate_limit_per_second)
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise KnowledgeRequestError(f"GET {url} failed: {exc}") from exc

    def post_json(
        self,
        url: str,
        *,
        json_payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        rate_limit_per_second: float | None = None,
    ) -> Any:
        self._throttle(rate_limit_per_second)
        try:
            response = self.session.post(url, json=json_payload, headers=headers, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise KnowledgeRequestError(f"POST {url} failed: {exc}") from exc
