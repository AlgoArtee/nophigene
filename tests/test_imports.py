"""Import smoke tests for the NophiGene analysis project."""

from src import analysis, app, bam_extraction, env, gene_region_extraction, human_protein_catalog, webapp


def test_analysis_module_imports() -> None:
    """Ensure the core analysis module remains importable."""
    assert analysis.DEFAULT_REGION


def test_web_modules_import() -> None:
    """Ensure the UI entrypoints remain importable."""
    assert app.build_parser() is not None
    assert webapp.app is not None


def test_gene_region_helpers_import() -> None:
    """Ensure the region helper module remains importable."""
    assert callable(gene_region_extraction.get_widest_region)


def test_human_protein_helpers_import() -> None:
    """Ensure the human protein catalog helper remains importable."""
    assert callable(human_protein_catalog.get_human_protein_catalog)


def test_bam_extraction_helpers_import() -> None:
    """Ensure the BAM extraction helper module remains importable."""
    assert callable(bam_extraction.build_extraction_commands)


def test_dotenv_loader_populates_missing_values_without_overriding(monkeypatch, tmp_path) -> None:
    """Ensure local .env files can seed process credentials safely."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "NOPHIGENE_TEST_TOKEN=from-file",
                'NOPHIGENE_QUOTED_VALUE="quoted"',
                "export NOPHIGENE_EXPORTED_VALUE=exported",
                "NOPHIGENE_EXISTING_VALUE=from-file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NOPHIGENE_TEST_TOKEN", raising=False)
    monkeypatch.delenv("NOPHIGENE_QUOTED_VALUE", raising=False)
    monkeypatch.delenv("NOPHIGENE_EXPORTED_VALUE", raising=False)
    monkeypatch.setenv("NOPHIGENE_EXISTING_VALUE", "already-set")

    loaded_path = env.load_dotenv(env_path)

    assert loaded_path == env_path
    assert env.os.environ["NOPHIGENE_TEST_TOKEN"] == "from-file"
    assert env.os.environ["NOPHIGENE_QUOTED_VALUE"] == "quoted"
    assert env.os.environ["NOPHIGENE_EXPORTED_VALUE"] == "exported"
    assert env.os.environ["NOPHIGENE_EXISTING_VALUE"] == "already-set"
