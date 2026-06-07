"""Regression checks for consistent UI port configuration."""

from pathlib import Path

from src import app, webapp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEB_PORT = 8766


def test_python_web_defaults_use_shared_preferred_port() -> None:
    args = app.build_parser().parse_args(["web"])

    assert app.DEFAULT_WEB_PORT == DEFAULT_WEB_PORT
    assert args.port == DEFAULT_WEB_PORT
    assert webapp.run_server.__defaults__ == ("0.0.0.0", DEFAULT_WEB_PORT, False)


def test_launchers_probe_health_and_track_selected_port() -> None:
    local_script = (PROJECT_ROOT / "scripts" / "start_nophigene_ui_local.ps1").read_text(
        encoding="utf-8"
    )
    docker_script = (PROJECT_ROOT / "scripts" / "start_nophigene_ui.ps1").read_text(
        encoding="utf-8"
    )
    stop_script = (PROJECT_ROOT / "scripts" / "stop_nophigene_ui_local.ps1").read_text(
        encoding="utf-8"
    )

    assert "[int]$Port = 8766" in local_script
    assert "[int]$Port = 8766" in docker_script
    assert "/api/v1/health" in local_script
    assert "/api/v1/health" in docker_script
    assert "Resolve-WebPort" in local_script
    assert "Resolve-WebPort" in docker_script
    assert ".nophigene-ui.port" in local_script
    assert ".nophigene-ui.port" in stop_script
    assert '${Port}:8766' in docker_script


def test_container_and_ui_copy_do_not_reference_old_port() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    template = (PROJECT_ROOT / "src" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "EXPOSE 8766" in dockerfile
    assert '"--port", "8766"' in dockerfile
    assert "127.0.0.1:8766" in template
    assert "127.0.0.1:8000" not in template
