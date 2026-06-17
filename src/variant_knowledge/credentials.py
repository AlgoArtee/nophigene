"""Credential resolution with explicit redaction boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .models import SourceSpec

REDACTED = "[redacted]"


@dataclass(frozen=True)
class ResolvedCredential:
    """A token or login hint resolved from environment or in-memory UI state."""

    source_key: str
    value: str = ""
    source: str = "none"

    @property
    def present(self) -> bool:
        return bool(self.value)

    def redacted_status(self) -> str:
        if not self.present:
            return "missing"
        return f"{self.source}:{REDACTED}"


class CredentialResolver:
    """Resolve source credentials without serializing secret values."""

    def __init__(self, provided: dict[str, str] | None = None) -> None:
        self.provided = provided or {}

    def resolve(self, spec: SourceSpec) -> ResolvedCredential:
        if spec.key in self.provided and self.provided[spec.key]:
            return ResolvedCredential(spec.key, str(self.provided[spec.key]), "session")
        if spec.env_var and os.environ.get(spec.env_var):
            return ResolvedCredential(spec.key, str(os.environ[spec.env_var]), "env")
        return ResolvedCredential(spec.key)


def credential_status_for_specs(
    specs: list[SourceSpec] | tuple[SourceSpec, ...],
    provided: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return redacted status strings keyed by source key."""
    resolver = CredentialResolver(provided)
    statuses: dict[str, str] = {}
    for spec in specs:
        if not spec.env_var and spec.access_type not in {"auth_api", "licensed"}:
            statuses[spec.key] = "not_required"
            continue
        statuses[spec.key] = resolver.resolve(spec).redacted_status()
    return statuses


def redact_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a copy with likely secret values redacted."""
    redacted: dict[str, object] = {}
    for key, value in payload.items():
        if any(token in key.lower() for token in ("token", "secret", "password", "credential", "api_key")):
            redacted[key] = REDACTED if value else ""
        else:
            redacted[key] = value
    return redacted
