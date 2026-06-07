"""Flask routes for the versioned local API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException, NotFound

from .errors import APIError
from .openapi import build_openapi_document
from .workflow_runner import normalize_job_request

try:
    from ..bam_extraction import get_extraction_tool_status, get_hg38_reference_status
except ImportError:
    from bam_extraction import get_extraction_tool_status, get_hg38_reference_status

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def _profiles():
    return current_app.config["NOPHIGENE_PROFILE_STORE"]


def _jobs():
    return current_app.config["NOPHIGENE_JOB_MANAGER"]


def _json_body() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        raise APIError("invalid_json", "Request body must contain valid JSON.", 400)
    return payload


@api_v1.errorhandler(APIError)
def handle_api_error(error: APIError):
    return error.to_response()


@api_v1.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    if isinstance(error, HTTPException):
        return APIError(
            "http_error",
            error.description,
            error.code or 500,
        ).to_response()
    current_app.logger.exception("Unhandled API error", exc_info=error)
    return APIError(
        "internal_error",
        "The API could not complete the request.",
        500,
    ).to_response()


@api_v1.get("")
@api_v1.get("/")
def api_index():
    return jsonify(
        {
            "name": "NophiGene Local Gene Workflow API",
            "version": "1.0",
            "health": "/api/v1/health",
            "openapi": "/api/v1/openapi.json",
            "profiles": "/api/v1/profiles",
            "jobs": "/api/v1/jobs",
        }
    )


@api_v1.get("/profiles")
def list_profiles():
    profiles = _profiles().list()
    return jsonify({"profiles": profiles, "count": len(profiles)})


@api_v1.post("/profiles")
def create_profile():
    return jsonify(_profiles().create(_json_body())), 201


@api_v1.get("/profiles/<profile_id>")
def get_profile(profile_id: str):
    return jsonify(_profiles().get(profile_id))


@api_v1.put("/profiles/<profile_id>")
def update_profile(profile_id: str):
    return jsonify(_profiles().update(profile_id, _json_body()))


@api_v1.delete("/profiles/<profile_id>")
def delete_profile(profile_id: str):
    _profiles().delete(profile_id)
    return "", 204


@api_v1.get("/jobs")
def list_jobs():
    jobs = _jobs().list()
    return jsonify({"jobs": jobs, "count": len(jobs)})


@api_v1.post("/jobs")
def submit_job():
    normalized = normalize_job_request(_json_body())
    job = _jobs().submit(normalized)
    response = jsonify(job)
    response.status_code = 202
    response.headers["Location"] = f"/api/v1/jobs/{job['id']}"
    return response


@api_v1.get("/jobs/<job_id>")
def get_job(job_id: str):
    return jsonify(_jobs().get(job_id))


@api_v1.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    return jsonify(_jobs().cancel(job_id))


@api_v1.get("/jobs/<job_id>/result")
def get_job_result(job_id: str):
    return jsonify(_jobs().result(job_id))


@api_v1.get("/jobs/<job_id>/artifacts/<path:artifact_path>")
def get_job_artifact(job_id: str, artifact_path: str):
    manager = _jobs()
    manager.get(job_id)
    job_dir = (manager.jobs_root / job_id).resolve()
    requested = (job_dir / artifact_path).resolve()
    try:
        requested.relative_to(job_dir)
    except ValueError as exc:
        raise APIError("invalid_artifact_path", "Artifact path escapes the job directory.", 400) from exc
    if not requested.is_file():
        raise APIError("artifact_not_found", "Artifact was not found.", 404)
    try:
        return send_from_directory(
            str(job_dir),
            requested.relative_to(job_dir).as_posix(),
            as_attachment=requested.suffix.lower() == ".zip",
        )
    except NotFound as exc:
        raise APIError("artifact_not_found", "Artifact was not found.", 404) from exc


@api_v1.get("/health")
def health():
    manager = _jobs()
    extraction = get_extraction_tool_status()
    reference = get_hg38_reference_status()
    return jsonify(
        {
            "status": "ok" if manager.worker_alive else "degraded",
            "worker": {
                "alive": manager.worker_alive,
                "queue_depth": manager.queue_depth,
            },
            "extraction": {
                "available": extraction["available"],
                "docker_runtime": extraction["docker_runtime"],
                "missing_tools": extraction["missing_tools"],
                "message": extraction["message"],
            },
            "hg38_reference": {
                "ready": reference["ready"],
                "message": reference["message"],
            },
        }
    )


@api_v1.get("/openapi.json")
def openapi():
    return jsonify(build_openapi_document())
