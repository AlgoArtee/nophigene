"""Small requests-based HTTP client for variant knowledge connectors."""

from __future__ import annotations

import atexit
import hashlib
import os
import re
import ssl
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_USER_AGENT = "NophiGene-DynamicVariantKB/1.0"
TLS_REMEDIATION = (
    "TLS certificate verification failed. Configure NOPHIGENE_CA_BUNDLE with a PEM bundle "
    "trusted by this network, or update the local Python/Windows certificate store."
)
SENSITIVE_PARAM_PATTERN = re.compile(
    r"([?&](?:api[_-]?key|token|password|secret|credential)=)[^&\s]+",
    flags=re.IGNORECASE,
)
EXPLICIT_CA_ENV_VARS = ("NOPHIGENE_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
_WINDOWS_CA_BUNDLE_PATH: str | None = None


class KnowledgeRequestError(RuntimeError):
    """Raised when a source request cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "request_failed",
        remediation: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def _redact_sensitive_text(value: object) -> str:
    """Redact likely secret query parameters from request error text."""
    text = str(value)
    return SENSITIVE_PARAM_PATTERN.sub(r"\1[redacted]", text)


def _explicit_ca_bundle() -> str:
    for env_var in EXPLICIT_CA_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return ""


def _certifi_bundle_path() -> str:
    try:
        import certifi
    except ImportError:
        return ""
    return str(certifi.where())


def _windows_merged_ca_bundle() -> str:
    """Create a temporary certifi + Windows trust-store CA bundle."""
    global _WINDOWS_CA_BUNDLE_PATH
    if _WINDOWS_CA_BUNDLE_PATH and Path(_WINDOWS_CA_BUNDLE_PATH).is_file():
        return _WINDOWS_CA_BUNDLE_PATH

    certs: list[str] = []
    certifi_path = _certifi_bundle_path()
    if certifi_path:
        try:
            certs.append(Path(certifi_path).read_text(encoding="ascii"))
        except OSError:
            pass

    seen: set[str] = set()
    server_auth_oid = "1.3.6.1.5.5.7.3.1"
    for store_name in ("ROOT", "CA"):
        try:
            store_certs = ssl.enum_certificates(store_name)
        except (AttributeError, OSError):
            continue
        for cert_bytes, encoding, trust in store_certs:
            if encoding != "x509_asn":
                continue
            if trust is not True and server_auth_oid not in set(trust or ()):
                continue
            try:
                pem = ssl.DER_cert_to_PEM_cert(cert_bytes)
            except (TypeError, ValueError):
                continue
            fingerprint = hashlib.sha256(pem.encode("ascii", errors="ignore")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            certs.append(pem)

    if not certs:
        return certifi_path

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        delete=False,
        prefix="nophigene-ca-",
        suffix=".pem",
    )
    with handle:
        handle.write("\n".join(certs))
        handle.write("\n")
    _WINDOWS_CA_BUNDLE_PATH = handle.name
    atexit.register(lambda path=handle.name: Path(path).unlink(missing_ok=True))
    return handle.name


def _resolve_ca_bundle() -> str | bool:
    explicit = _explicit_ca_bundle()
    if explicit:
        return explicit
    if os.name == "nt":
        windows_bundle = _windows_merged_ca_bundle()
        if windows_bundle:
            return windows_bundle
    certifi_path = _certifi_bundle_path()
    if certifi_path:
        return certifi_path
    return True


class RequestClient:
    """Requests wrapper with conservative timeout and light throttling."""

    def __init__(self, *, timeout: int = DEFAULT_TIMEOUT_SECONDS, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.timeout = timeout
        self.verify = _resolve_ca_bundle()
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
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify,
            )
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError as exc:
            safe_error = _redact_sensitive_text(exc)
            raise KnowledgeRequestError(
                f"GET {url} failed: {TLS_REMEDIATION} Detail: {safe_error}",
                code="tls_certificate_verification_failed",
                remediation=TLS_REMEDIATION,
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise KnowledgeRequestError(f"GET {url} failed: {_redact_sensitive_text(exc)}") from exc

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
            response = self.session.post(
                url,
                json=json_payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify,
            )
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError as exc:
            safe_error = _redact_sensitive_text(exc)
            raise KnowledgeRequestError(
                f"POST {url} failed: {TLS_REMEDIATION} Detail: {safe_error}",
                code="tls_certificate_verification_failed",
                remediation=TLS_REMEDIATION,
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise KnowledgeRequestError(f"POST {url} failed: {_redact_sensitive_text(exc)}") from exc
