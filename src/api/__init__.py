"""Versioned local REST API for NophiGene."""

from __future__ import annotations

from flask import Flask

from .jobs import get_default_job_manager
from .profiles import get_default_profile_store
from .routes import api_v1


def register_api(app: Flask) -> None:
    """Register the API blueprint and start its local background worker."""
    if "api_v1" in app.blueprints:
        return
    app.config.setdefault("NOPHIGENE_PROFILE_STORE", get_default_profile_store())
    app.config.setdefault("NOPHIGENE_JOB_MANAGER", get_default_job_manager())
    app.register_blueprint(api_v1)
    app.config["NOPHIGENE_JOB_MANAGER"].start()
