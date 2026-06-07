"""File-backed sample profile CRUD."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import APIError
from .serialization import read_json, utc_now, write_json_atomic

try:
    from ..analysis import PROJECT_ROOT
    from ..workflow import normalize_genome_build
except ImportError:
    from analysis import PROJECT_ROOT
    from workflow import normalize_genome_build

DEFAULT_PROFILE_PATH = PROJECT_ROOT / "data" / "api" / "sample_profiles.json"


def _resolve_local_path(value: Any, field_name: str, *, required: bool = True) -> str:
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise APIError("invalid_profile", f"'{field_name}' is required.", 422)
        return ""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _require_file(path_text: str, field_name: str) -> None:
    if not Path(path_text).is_file():
        raise APIError(
            "profile_file_not_found",
            f"Profile file for '{field_name}' does not exist: {path_text}",
            422,
        )


def _normalize_sources(payload: Any, field_name: str) -> list[dict[str, str]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise APIError("invalid_profile", f"'{field_name}' must be a list.", 422)
    normalized: list[dict[str, str]] = []
    for index, source in enumerate(payload):
        if not isinstance(source, dict):
            raise APIError(
                "invalid_profile",
                f"'{field_name}[{index}]' must be an object.",
                422,
            )
        path = _resolve_local_path(source.get("path"), f"{field_name}[{index}].path")
        _require_file(path, f"{field_name}[{index}].path")
        try:
            genome_build = normalize_genome_build(source.get("genome_build"))
        except ValueError as exc:
            raise APIError(
                "invalid_profile",
                f"Invalid genome build for '{field_name}[{index}]': {exc}",
                422,
            ) from exc
        item = {"path": path, "genome_build": genome_build}
        if field_name == "bam_sources" and source.get("reference_fasta"):
            reference = _resolve_local_path(
                source.get("reference_fasta"),
                f"{field_name}[{index}].reference_fasta",
            )
            _require_file(reference, f"{field_name}[{index}].reference_fasta")
            item["reference_fasta"] = reference
        normalized.append(item)
    return normalized


def normalize_profile(
    payload: Any,
    *,
    profile_id: str | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one profile payload."""
    if not isinstance(payload, dict):
        raise APIError("invalid_json", "Request body must be a JSON object.", 400)
    submitted_id = str(payload.get("id") or "").strip()
    if existing and submitted_id and submitted_id != existing["id"]:
        raise APIError("immutable_profile_id", "Profile IDs cannot be changed.", 409)
    final_id = profile_id or submitted_id or uuid.uuid4().hex
    display_name = str(payload.get("display_name") or "").strip()
    if not display_name:
        raise APIError("invalid_profile", "'display_name' is required.", 422)

    idat_prefix = _resolve_local_path(payload.get("idat_prefix"), "idat_prefix")
    for suffix in ("_Grn.idat", "_Red.idat"):
        _require_file(f"{idat_prefix}{suffix}", f"idat_prefix{suffix}")
    manifest_path = _resolve_local_path(payload.get("manifest_path"), "manifest_path")
    _require_file(manifest_path, "manifest_path")
    population_path = _resolve_local_path(
        payload.get("population_statistics_path"),
        "population_statistics_path",
        required=False,
    )
    if population_path:
        _require_file(population_path, "population_statistics_path")

    vcf_sources = _normalize_sources(payload.get("vcf_sources"), "vcf_sources")
    bam_sources = _normalize_sources(payload.get("bam_sources"), "bam_sources")
    if not vcf_sources and not bam_sources:
        raise APIError(
            "invalid_profile",
            "At least one VCF or BAM source is required.",
            422,
        )

    now = utc_now()
    try:
        default_genome_build = normalize_genome_build(
            payload.get("default_genome_build") or "hg19"
        )
    except ValueError as exc:
        raise APIError("invalid_profile", str(exc), 422) from exc

    return {
        "id": final_id,
        "display_name": display_name,
        "default_genome_build": default_genome_build,
        "idat_prefix": idat_prefix,
        "manifest_path": manifest_path,
        "population_statistics_path": population_path,
        "vcf_sources": vcf_sources,
        "bam_sources": bam_sources,
        "created_at": (existing or {}).get("created_at", now),
        "updated_at": now,
    }


class ProfileStore:
    """Thread-safe atomic JSON profile store."""

    def __init__(self, path: Path = DEFAULT_PROFILE_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _load(self) -> dict[str, dict[str, Any]]:
        payload = read_json(self.path, default={"profiles": []})
        profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
        return {
            str(profile["id"]): profile
            for profile in profiles
            if isinstance(profile, dict) and profile.get("id")
        }

    def _save(self, profiles: dict[str, dict[str, Any]]) -> None:
        write_json_atomic(
            self.path,
            {"schema_version": "1.0", "profiles": sorted(profiles.values(), key=lambda item: item["id"])},
        )

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(profile) for profile in self._load().values()]

    def get(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._load().get(profile_id)
            if profile is None:
                raise APIError("profile_not_found", f"Profile '{profile_id}' was not found.", 404)
            return deepcopy(profile)

    def create(self, payload: Any) -> dict[str, Any]:
        with self._lock:
            profiles = self._load()
            profile = normalize_profile(payload)
            if profile["id"] in profiles:
                raise APIError("profile_exists", f"Profile '{profile['id']}' already exists.", 409)
            profiles[profile["id"]] = profile
            self._save(profiles)
            return deepcopy(profile)

    def update(self, profile_id: str, payload: Any) -> dict[str, Any]:
        with self._lock:
            profiles = self._load()
            existing = profiles.get(profile_id)
            if existing is None:
                raise APIError("profile_not_found", f"Profile '{profile_id}' was not found.", 404)
            profile = normalize_profile(payload, profile_id=profile_id, existing=existing)
            profiles[profile_id] = profile
            self._save(profiles)
            return deepcopy(profile)

    def delete(self, profile_id: str) -> None:
        with self._lock:
            profiles = self._load()
            if profile_id not in profiles:
                raise APIError("profile_not_found", f"Profile '{profile_id}' was not found.", 404)
            del profiles[profile_id]
            self._save(profiles)


_DEFAULT_PROFILE_STORE: ProfileStore | None = None


def get_default_profile_store() -> ProfileStore:
    global _DEFAULT_PROFILE_STORE
    if _DEFAULT_PROFILE_STORE is None:
        _DEFAULT_PROFILE_STORE = ProfileStore()
    return _DEFAULT_PROFILE_STORE
