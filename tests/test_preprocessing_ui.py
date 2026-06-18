"""Regression tests for preprocessing form behavior in the Flask UI."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.analysis import load_gene_interpretation_database, load_gene_population_database
from src.webapp import _build_data_sources_payload, _classify_functional_family, app


def _stub_common_discovery(monkeypatch) -> None:
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])


def test_data_sources_payload_combines_curated_and_dynamic_sources() -> None:
    """The Result Viewer Data Sources payload should merge curated and dynamic provenance."""
    knowledge_base = load_gene_interpretation_database("DRD4")
    population_database = load_gene_population_database("DRD4")
    assert knowledge_base is not None
    assert population_database is not None

    payload = _build_data_sources_payload(
        knowledge_base=knowledge_base,
        population_database=population_database,
        population_insights={
            "summary": "Population summary",
            "variant_population_records": [{"variant": "rs1"}],
            "gene_population_patterns": [],
            "sources": [{"label": "Population source", "url": "https://example.com/pop"}],
        },
        methylation_insights={
            "summary": "Methylation summary",
            "clinical_context": "Methylation context",
            "evidence": [{"label": "Methylation paper", "url": "https://example.com/methylation"}],
            "whitelist_probe_reference_rows": [
                {"probe_id": "cg1", "papers": [{"label": "Probe paper", "url": "https://example.com/probe"}]}
            ],
        },
        dynamic_payload={
            "provider_statuses": [
                {
                    "source_key": "clinvar",
                    "name": "ClinVar",
                    "lane": "clinical",
                    "status": "ok",
                    "message": "ClinVar returned one record.",
                    "record_count": 1,
                    "homepage": "https://www.ncbi.nlm.nih.gov/clinvar/",
                },
                {
                    "source_key": "hgmd",
                    "name": "HGMD",
                    "lane": "licensed",
                    "status": "needs_export",
                    "message": "Upload a permitted export.",
                    "record_count": 0,
                    "warnings": ["HGMD needs a licensed export."],
                    "license_note": "License-gated source.",
                },
            ],
            "source_records": [
                {
                    "source_key": "clinvar",
                    "category": "clinical",
                    "label": "ClinVar DRD4 record",
                    "summary": "ClinVar summary",
                    "url": "https://example.com/clinvar-drd4",
                }
            ],
            "literature_records": [
                {
                    "source_key": "europe_pmc",
                    "category": "literature",
                    "title": "DRD4 literature",
                    "summary": "Literature summary",
                    "url": "https://example.com/lit",
                }
            ],
            "local_article_evidence": {
                "status": "ok",
                "message": "Extracted one local snippet.",
                "records": [
                    {
                        "source_key": "local_pdf_articles",
                        "title": "Local DRD4 PDF",
                        "snippet": "DRD4 local finding",
                        "url": "https://example.com/local",
                    }
                ],
                "provenance": {"warnings": [], "errors": []},
            },
        },
        selected_source_keys=["clinvar", "hgmd"],
    )

    cards = [card for group in payload["groups"] for card in group["cards"]]
    by_key = {card["source_key"]: card for card in cards}

    assert payload["dynamic_status"] == "available"
    assert by_key["curated_gene_bundle"]["status"] == "ok"
    assert any("NCBI Gene 1815" in link["label"] for link in by_key["curated_gene_bundle"]["links"])
    assert by_key["clinvar"]["status"] == "ok"
    assert by_key["hgmd"]["status"] == "needs_export"
    assert by_key["hgmd"]["license_note"] == "License-gated source."
    assert "europe_pmc_literature" in by_key
    assert by_key["local_pdf_articles"]["record_count"] == 1
    assert any(link["url"] == "https://example.com/probe" for link in by_key["methylation_evidence"]["links"])


def test_data_sources_payload_marks_selected_dynamic_sources_not_run() -> None:
    """When no dynamic KB is present, selected providers should appear as not_run cards."""
    payload = _build_data_sources_payload(
        knowledge_base={"database_name": "Mock KB", "gene_context": {}, "variant_records": []},
        population_database={"database_name": "Mock population DB"},
        population_insights={},
        methylation_insights={},
        dynamic_payload=None,
        selected_source_keys=["clinvar", "hgmd"],
    )

    cards = [card for group in payload["groups"] for card in group["cards"]]
    statuses = {card["source_key"]: card["status"] for card in cards}

    assert payload["dynamic_status"] == "not_run"
    assert statuses["clinvar"] == "not_run"
    assert statuses["hgmd"] == "not_run"


def test_preprocessing_template_preserves_clicked_submit_button() -> None:
    """The loading helper should preserve the clicked preprocessing action."""
    template_path = Path(__file__).resolve().parent.parent / "src" / "templates" / "index.html"
    template_text = template_path.read_text(encoding="utf-8")
    functional_map_template_path = (
        Path(__file__).resolve().parent.parent / "src" / "templates" / "functional_map.html"
    )
    functional_map_template_text = functional_map_template_path.read_text(encoding="utf-8")

    assert "const submitter = event.submitter instanceof HTMLElement ? event.submitter : null;" in template_text
    assert "formElement.requestSubmit(submitter);" in template_text
    assert "temporarySubmitterInput.name = submitter.name;" in template_text
    assert "Bundled named {{ result.variant_interpretations.gene_name }} markers" in template_text
    assert "{{ result.variant_interpretations.curated_named_markers_summary }}" in template_text
    assert "Genome and nucleotides" in template_text
    assert "{{ item.genome_location }}" in template_text
    assert "{{ item.nucleotide_change }}" in template_text
    assert "{% for link in item.research_links %}" in template_text
    assert "Genetic Variant Results" in template_text
    assert "Data Sources" in template_text
    assert 'data-subtab-target="data_sources"' in template_text
    assert 'data-subtab-panel="data_sources"' in template_text
    assert template_text.count("data-subtab-target=") == 3
    assert template_text.count("data-subtab-panel=") == 3
    assert "{{ result.variant_interpretations.sample_highlights.result_table_rows }}" not in template_text
    assert "Exact variant links in this sample" in template_text
    assert "{% for row in result.variant_interpretations.sample_highlights.result_table_rows %}" in template_text
    assert "{% for row in result.methylation_insights.summary_metric_rows %}" in template_text
    assert "{{ result.methylation_insights.whitelist_explanation }}" in template_text
    assert "{{ result.methylation_insights.gene_name_match_rule }}" in template_text
    assert "{{ result.methylation_insights.whitelist_probe_reference_summary }}" in template_text
    assert 'data-detail-target="qa"' in template_text
    assert "{{ app_structure_qa_items }}" not in template_text

    assert "{{ item.question }}" in template_text
    assert "App Structure Q&amp;A" in template_text
    assert "Curated probe-to-variant and paper links" in template_text
    assert "These are the exact observed whitelist probe rows used to build the whitelist mean beta shown above." in template_text
    assert "Predictive Theses" in template_text
    assert 'data-tab-target="predictive_theses"' in template_text
    assert 'data-tab-target="central_database"' in template_text
    assert 'data-tab-target="extraction"' in template_text
    assert 'data-tab-panel="extraction"' in template_text
    assert "Functional Map" in template_text
    assert "url_for('functional_map_page')" in template_text
    assert "Use in preprocessing" in functional_map_template_text
    assert "Open latest report" in functional_map_template_text
    assert "data-functional-search" in functional_map_template_text
    assert "function updateFunctionalMap()" in functional_map_template_text
    assert 'window.addEventListener("pageshow", updateFunctionalMap)' in functional_map_template_text
    assert "Extract Regional VCF" in template_text
    assert "Prepare Reference" in template_text
    assert "Knowledge Workflows" in template_text
    assert 'name="knowledge_workflow"' in template_text
    assert 'data-knowledge-workflow' in template_text
    assert 'data-knowledge-source="{{ source.key }}"' in template_text
    assert "syncSourcesFromWorkflow" in template_text
    assert "Core safety default" in template_text
    assert "Workflow summary" in template_text
    assert "preprocess-domain-form" in template_text
    assert 'data-preprocess-domain="genetics"' in template_text
    assert 'data-preprocess-domain="epigenetics"' in template_text
    assert 'data-preprocess-domain="knowledge"' in template_text
    assert "Genetics" in template_text
    assert "Epigenetics" in template_text
    assert "Knowledge Base Building" in template_text
    assert 'value="reset_preprocessing"' in template_text
    assert "Refresh Process" in template_text
    assert "preprocess-flow-arrow" in template_text
    assert "staged preprocessing flow" in template_text
    assert "two preprocessing steps" not in template_text
    assert "Use local PDF article folder" in template_text
    assert 'name="article_pdf_folder"' in template_text
    assert 'name="article_pdf_recursive"' in template_text
    assert 'name="max_article_pdfs"' in template_text
    assert "Query available database connectors and local article snippets." in template_text
    assert "Local article snippets" in template_text
    assert "Search computer for BAM files" in template_text
    assert 'name="bam_search_root"' in template_text
    assert 'value="search_bam_files"' in template_text
    assert "Browse BAM File" in template_text
    assert 'value="browse_bam_file"' in template_text
    assert "GRCh38 Extraction Suggested" in template_text
    assert "data-extraction-scope" in template_text
    assert "function updateExtractionScopeFields()" in template_text
    assert "Central Analysis Database" in template_text
    assert "central-database-table" in template_text
    assert "One row per observed variant" in template_text
    assert "curated biological context" in template_text
    assert "Only labeled variants" in template_text
    assert 'data-general-database-labeled-filter' in template_text
    assert 'data-general-database-row' in template_text
    assert 'data-variant-labeled="{{' in template_text
    assert "function updateGeneralDatabaseFilter()" in template_text
    assert 'name="analysis_scope"' in template_text
    assert "Report focus" in template_text
    assert "Promoter + gene" in template_text
    assert "focused reports do not update the central database" in template_text
    assert "function updateAnalysisScopeFields()" in template_text
    assert "Report focus" in template_text
    assert 'data-analysis-shell' in template_text
    assert 'class="analysis-shell is-form-collapsed"' in template_text
    assert 'id="analysis-form"' in template_text
    assert 'form="analysis-form"' in template_text
    assert 'data-analysis-form-toggle' in template_text
    assert "Change variables" in template_text
    assert "function setAnalysisFormCollapsed(collapsed)" in template_text
    assert 'if (name === "analysis")' in template_text
    assert "Variant Prediction" in template_text
    assert "Methylation Prediction" in template_text
    assert "matched case{{ \"\" if result.predictive_theses.matched_case_count == 1 else \"s\" }}" in template_text
    assert 'name="overwrite_general_database"' in template_text
    assert "Overwrite gene variant rows in general database" in template_text
    assert "results/general_gene_analysis_database.csv" in template_text
    assert "{{ result.general_database_status }}" in template_text
    assert '<details class="predictive-card">' in template_text
    assert '<article class="predictive-card">' not in template_text
    assert "grid-template-columns: 1fr;" in template_text
    assert "flex-wrap: wrap;" in template_text
    assert "width: min(98vw, 1920px);" in template_text
    assert "table-layout: fixed;" in template_text
    assert "overflow-wrap: anywhere;" in template_text
    assert "window.setTimeout(() => formElement.submit(), 40);" not in template_text
    assert "const preprocessLoadingStepsByAction = {" in template_text
    assert 'data-preprocess-loading-steps' in template_text
    assert "Confirm the selected coordinates for the current gene." in template_text
    assert ".preprocess-detail-grid {" in template_text
    assert ".preprocess-panel {\n      padding: 18px;\n      display: grid;" in template_text
    assert "panel.hidden = !isActive;" in template_text
    assert "[hidden] {" in template_text
    assert 'style="display:{% if initial_tab == ' in template_text
    assert 'panel.style.display = isActive ? "block" : "none";' in template_text
    assert 'data-variant-raw-table' in template_text
    assert 'variant-raw-data' in template_text
    assert "function renderVariantRawPage()" in template_text


def test_knowledge_sources_tab_accepts_licensed_export_upload(monkeypatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr("src.webapp.RESULTS_DIR", results_dir)
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "knowledge_sources",
            "knowledge_source": ["hgmd"],
            "source_import_hgmd": (
                io.BytesIO(b"gene,rsid,classification\nGENE1,rs1,Pathogenic\n"),
                "hgmd.csv",
            ),
        },
        content_type="multipart/form-data",
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Current import:" in page
    assert "import: ready" in page
    with client.session_transaction() as session_state:
        source_imports = session_state["knowledge_sources_state"]["source_imports"]

    assert "hgmd" in source_imports
    assert (Path(__file__).resolve().parent.parent / source_imports["hgmd"]).is_file()


def test_knowledge_workflows_render_core_safety_defaults(monkeypatch) -> None:
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    page = app.test_client().get("/").get_data(as_text=True)

    assert "Knowledge Workflows" in page
    assert 'value="clinical_variant_triage"' in page
    assert 'data-workflow-sources="clinvar,clingen,ensembl,dbsnp,civic,panelapp,mavedb,omim,oncokb,hgmd,varsome,franklin"' in page
    assert 'value="local_pdf_article_evidence"' in page
    assert 'data-workflow-sources="local_pdf_articles"' in page
    assert 'value="licensed_aggregator_review"' in page
    clinical_index = page.index('value="clinical_variant_triage"')
    local_articles_index = page.index('value="local_pdf_article_evidence"')
    licensed_index = page.index('value="licensed_aggregator_review"')
    assert "checked" in page[clinical_index:clinical_index + 420]
    assert "checked" not in page[local_articles_index:local_articles_index + 420]
    assert "checked" not in page[licensed_index:licensed_index + 420]


def test_preprocess_find_region_submission_updates_session(monkeypatch) -> None:
    """A valid preprocessing action should resolve the gene interval and persist it."""

    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr(
        "src.webapp.find_gene_region",
        lambda gene_symbol: {
            "gene_name": gene_symbol.upper(),
            "selected_region": "chr11:639677-643057",
            "selected_sources": ["NCBI RefSeq"],
            "candidate_regions": [{"source": "NCBI RefSeq", "region": "chr11:639677-643057"}],
        },
    )

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "preprocess",
            "gene_name": "drd4",
            "preprocess_action": "find_region",
        },
    )

    assert response.status_code == 200
    assert "Resolved DRD4 to standard promoter+gene region 11:636269-640706." in response.get_data(as_text=True)

    with client.session_transaction() as session_state:
        preprocess_state = session_state["preprocess_state"]

    assert preprocess_state["gene_name"] == "DRD4"
    assert preprocess_state["region"] == "11:636269-640706"
    assert preprocess_state["analysis_scope"] == "promoter_plus_gene"
    assert preprocess_state["scope_regions"]["promoter_plus_gene"] == "11:636269-640706"
    assert preprocess_state["scope_regions"]["promoter_only"] == "11:636269-637268"
    assert preprocess_state["scope_regions"]["gene_only"] == "11:637269-640706"
    assert preprocess_state["region_ready"] is True
    assert preprocess_state["manifest_ready"] is False
    assert preprocess_state["analysis_ready"] is False
    assert preprocess_state["selected_sources"] == ["NCBI RefSeq", "Local curated promoter/gene intervals"]


def test_preprocess_hg38_only_gene_suggests_extraction(monkeypatch) -> None:
    """Finding an hg38-only gene should prefill Extraction and guide the user there."""

    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr(
        "src.webapp.find_gene_region",
        lambda gene_symbol: {
            "gene_name": "POTEB3",
            "selected_region": "15:21405401-21440499",
            "selected_sources": ["NCBI RefSeq"],
            "candidate_regions": [{"source": "NCBI RefSeq", "region": "15:21405401-21440499"}],
        },
    )

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "preprocess",
            "gene_name": "POTEB3",
            "preprocess_action": "find_region",
        },
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "GRCh38 Extraction Suggested" in page
    assert "Use the Extraction tab with a GRCh38-aligned BAM" in page

    with client.session_transaction() as session_state:
        preprocess_state = session_state["preprocess_state"]
        extraction_state = session_state["extraction_state"]

    assert preprocess_state["hg38_extraction_suggested"] is True
    assert preprocess_state["hg38_extraction_region"] == "15:21405401-21441499"
    assert extraction_state["gene_name"] == "POTEB3"
    assert extraction_state["region"] == "15:21405401-21441499"
    assert extraction_state["output_vcf"] == "data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz"


def test_mt_rnr1_zero_probe_preprocessing_unlocks_analysis(monkeypatch, tmp_path: Path) -> None:
    """Curated mitochondrial genes should unlock analysis even with a zero-row EPIC subset."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("IlmnID,CHR,MAPINFO,UCSC_RefGene_Name\n", encoding="utf-8")
    output_path = tmp_path / "MT-RNR1_epigenetics_hg19.csv"
    captured_call: dict[str, object] = {}

    def fake_save_filtered_manifest(**kwargs: object) -> dict[str, object]:
        captured_call.update(kwargs)
        output_path.write_text("IlmnID,CHR,MAPINFO,UCSC_RefGene_Name\n", encoding="utf-8")
        return {"output_path": output_path, "probe_count": 0}

    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.save_filtered_manifest", fake_save_filtered_manifest)

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "MT-RNR1",
            "region": "MT:1-1601",
            "analysis_scope": "promoter_plus_gene",
            "scope_regions": {
                "promoter_plus_gene": "MT:1-1601",
                "promoter_only": "MT:1-647",
                "gene_only": "MT:648-1601",
            },
            "scope_region_source": "Local curated promoter/gene intervals",
            "manifest_source": str(manifest_path),
            "filtered_manifest": "",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": False,
            "analysis_ready": False,
            "probe_count": 0,
            "build": "hg19",
            "logs": [],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": True,
        }

    response = client.post(
        "/",
        data={
            "workflow": "preprocess",
            "gene_name": "MT-RNR1",
            "preprocess_region": "MT:1-1601",
            "manifest_source": str(manifest_path),
            "overwrite_filtered_manifest": "1",
            "preprocess_action": "select_methylation",
        },
    )

    assert response.status_code == 200
    assert captured_call["allow_empty"] is True

    with client.session_transaction() as session_state:
        preprocess_state = session_state["preprocess_state"]

    assert preprocess_state["manifest_ready"] is True
    assert preprocess_state["analysis_ready"] is True
    assert preprocess_state["probe_count"] == 0


def test_get_request_resets_preprocessing_workspace(monkeypatch) -> None:
    """A fresh page load should not keep the previous gene unlocked in the UI."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "IGF1R",
            "region": "15:99191768-99507759",
            "manifest_source": "data/infinium-methylationepic-v-1-0-b5-manifest-file.csv",
            "filtered_manifest": "src/gene_data/IGF1R_epigenetics_hg19.csv",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": True,
            "analysis_ready": True,
            "probe_count": 208,
            "build": "hg19",
            "logs": ["[stdout] stale state"],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": False,
        }

    response = client.get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-tab-target="analysis"' not in page
    assert 'id="gene_name" name="gene_name" type="text" value="DRD4"' in page

    with client.session_transaction() as session_state:
        assert "preprocess_state" not in session_state


def test_left_preprocessing_form_shows_only_genetics_by_default(monkeypatch) -> None:
    _stub_common_discovery(monkeypatch)

    page = app.test_client().get("/").get_data(as_text=True)

    assert 'data-preprocess-domain="genetics"' in page
    assert 'data-preprocess-domain="epigenetics"' not in page
    assert 'data-preprocess-domain="knowledge"' not in page
    assert "Refresh Process" in page


def test_left_preprocessing_form_reveals_epigenetics_after_region_ready(monkeypatch) -> None:
    _stub_common_discovery(monkeypatch)

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "TEST",
            "region": "1:1-100",
            "manifest_source": "data/manifest.csv",
            "filtered_manifest": "",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": False,
            "analysis_ready": False,
            "probe_count": 0,
            "build": "hg19",
            "logs": [],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": False,
        }

    page = client.post("/", data={"workflow": "knowledge_sources"}).get_data(as_text=True)

    assert 'data-preprocess-domain="genetics"' in page
    assert 'data-preprocess-domain="epigenetics"' in page
    assert 'data-preprocess-domain="knowledge"' not in page
    assert page.count('class="preprocess-flow-arrow"') == 1


def test_left_preprocessing_form_reveals_knowledge_after_manifest_ready(monkeypatch) -> None:
    _stub_common_discovery(monkeypatch)

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "TEST",
            "region": "1:1-100",
            "manifest_source": "data/manifest.csv",
            "filtered_manifest": "src/gene_data/TEST_epigenetics_hg19.csv",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": True,
            "analysis_ready": True,
            "probe_count": 12,
            "build": "hg19",
            "logs": [],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": False,
        }

    page = client.post("/", data={"workflow": "knowledge_sources"}).get_data(as_text=True)

    assert 'data-preprocess-domain="genetics"' in page
    assert 'data-preprocess-domain="epigenetics"' in page
    assert 'data-preprocess-domain="knowledge"' in page
    assert page.count('class="preprocess-flow-arrow"') == 2


def test_preprocess_refresh_resets_state_and_preserves_knowledge_sources(monkeypatch) -> None:
    _stub_common_discovery(monkeypatch)

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "TEST",
            "region": "1:1-100",
            "manifest_source": "data/manifest.csv",
            "filtered_manifest": "src/gene_data/TEST_epigenetics_hg19.csv",
            "region_candidates": [{"source": "mock", "region": "1:1-100"}],
            "selected_sources": ["mock"],
            "region_ready": True,
            "manifest_ready": True,
            "analysis_ready": True,
            "probe_count": 12,
            "build": "hg19",
            "logs": ["[stdout] old run"],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": True,
            "knowledge_vcf_source": "data/test.vcf",
            "dynamic_kb_ready": True,
            "dynamic_kb_path": "results/dynamic_knowledge_bases/test/variant_kb.json",
            "dynamic_kb_status": "ready",
        }
        session_state["knowledge_sources_state"] = {
            "selected_workflows": ["clinical_variant_triage"],
            "selected_sources": ["clinvar"],
            "source_imports": {"hgmd": "results/imports/hgmd.csv"},
            "notice": "keep me",
        }

    response = client.post(
        "/",
        data={
            "workflow": "preprocess",
            "preprocess_action": "reset_preprocessing",
            "gene_name": "STALE",
            "preprocess_region": "9:9-99",
        },
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Preprocessing was refreshed. Start again with Genetics." in page
    assert 'data-tab-target="analysis"' not in page
    assert 'data-preprocess-domain="genetics"' in page
    assert 'data-preprocess-domain="epigenetics"' not in page
    with client.session_transaction() as session_state:
        preprocess_state = session_state["preprocess_state"]
        knowledge_sources_state = session_state["knowledge_sources_state"]

    assert preprocess_state["gene_name"] == "DRD4"
    assert preprocess_state["region_ready"] is False
    assert preprocess_state["manifest_ready"] is False
    assert preprocess_state["analysis_ready"] is False
    assert preprocess_state["dynamic_kb_path"] == ""
    assert knowledge_sources_state["selected_sources"] == ["clinvar"]
    assert knowledge_sources_state["source_imports"] == {"hgmd": "results/imports/hgmd.csv"}


def test_extraction_tab_renders_with_disabled_tool_state(monkeypatch) -> None:
    """The Extraction tab should show a clear unavailable state outside Docker tooling."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: ["data/sample.hg38.bam"])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])
    monkeypatch.setattr(
        "src.webapp.get_extraction_tool_status",
        lambda: {
            "available": False,
            "docker_runtime": False,
            "local_override": False,
            "tools": {"samtools": None, "bcftools": None},
            "missing_tools": ["samtools", "bcftools"],
            "message": "Extraction is unavailable because samtools/bcftools are not on PATH.",
        },
    )
    monkeypatch.setattr(
        "src.webapp.get_hg38_reference_status",
        lambda: {
            "ready": False,
            "fasta": "data/reference/hg38/hg38.analysisSet.fa",
            "fai": "data/reference/hg38/hg38.analysisSet.fa.fai",
            "message": "Reference is not ready.",
        },
    )

    client = app.test_client()
    response = client.get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-tab-target="extraction"' in page
    assert "Docker-only extraction" in page
    assert "Extraction is unavailable because samtools/bcftools are not on PATH." in page
    assert "data/sample.hg38.bam" in page


def test_extraction_bam_search_adds_computer_paths(monkeypatch, tmp_path: Path) -> None:
    """Extraction should search a chosen folder and add BAMs to the picker."""
    bam_path = tmp_path / "nested" / "sample.hg38.bam"
    bam_path.parent.mkdir()
    bam_path.write_bytes(b"BAM placeholder")
    expected_path = bam_path.as_posix()

    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])
    monkeypatch.setattr(
        "src.webapp.get_extraction_tool_status",
        lambda: {
            "available": False,
            "docker_runtime": False,
            "local_override": False,
            "tools": {"samtools": None, "bcftools": None},
            "missing_tools": ["samtools", "bcftools"],
            "message": "Extraction is unavailable because samtools/bcftools are not on PATH.",
        },
    )
    monkeypatch.setattr(
        "src.webapp.get_hg38_reference_status",
        lambda: {
            "ready": False,
            "fasta": "data/reference/hg38/hg38.analysisSet.fa",
            "fai": "data/reference/hg38/hg38.analysisSet.fa.fai",
            "message": "Reference is not ready.",
        },
    )

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "extraction",
            "extraction_gene_name": "POTEB3",
            "extraction_genome_build": "hg38",
            "extraction_scope": "promoter_plus_gene",
            "extraction_region": "15:21405401-21441499",
            "bam_search_root": str(tmp_path),
            "output_vcf": "data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz",
            "extraction_action": "search_bam_files",
        },
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Found 1 BAM file" in page
    assert expected_path in page

    with client.session_transaction() as session_state:
        extraction_state = session_state["extraction_state"]

    assert extraction_state["bam_path"] == expected_path
    assert extraction_state["bam_search_results"] == [expected_path]


def test_extraction_browse_bam_file_updates_selected_path(monkeypatch, tmp_path: Path) -> None:
    """The local native file picker action should populate the BAM path."""
    bam_path = tmp_path / "picked.hg38.bam"
    bam_path.write_bytes(b"BAM placeholder")
    expected_path = bam_path.as_posix()

    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])
    monkeypatch.setattr("src.webapp.browse_bam_file", lambda initial_path="": expected_path)
    monkeypatch.setattr(
        "src.webapp.get_extraction_tool_status",
        lambda: {
            "available": False,
            "docker_runtime": False,
            "local_override": False,
            "tools": {"samtools": None, "bcftools": None},
            "missing_tools": ["samtools", "bcftools"],
            "message": "Extraction is unavailable because samtools/bcftools are not on PATH.",
        },
    )
    monkeypatch.setattr(
        "src.webapp.get_hg38_reference_status",
        lambda: {
            "ready": False,
            "fasta": "data/reference/hg38/hg38.analysisSet.fa",
            "fai": "data/reference/hg38/hg38.analysisSet.fa.fai",
            "message": "Reference is not ready.",
        },
    )

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "extraction",
            "extraction_gene_name": "POTEB3",
            "extraction_genome_build": "hg38",
            "extraction_scope": "promoter_plus_gene",
            "extraction_region": "15:21405401-21441499",
            "bam_search_root": "data",
            "output_vcf": "data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz",
            "extraction_action": "browse_bam_file",
        },
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert f"Selected BAM file {expected_path}." in page

    with client.session_transaction() as session_state:
        extraction_state = session_state["extraction_state"]

    assert extraction_state["bam_path"] == expected_path
    assert extraction_state["bam_search_results"] == [expected_path]


def test_successful_extraction_updates_run_analysis_defaults(monkeypatch) -> None:
    """A completed extraction should unlock analysis with the new regional VCF path."""
    output_path = Path(__file__).resolve().parent.parent / "data" / "extracted" / "POTEB3_hg38_promoter_plus_gene.vcf.gz"

    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: ["data/sample.hg38.bam"])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])
    monkeypatch.setattr(
        "src.webapp.get_extraction_tool_status",
        lambda: {
            "available": True,
            "docker_runtime": True,
            "local_override": False,
            "tools": {"samtools": "/usr/bin/samtools", "bcftools": "/usr/bin/bcftools"},
            "missing_tools": [],
            "message": "Extraction tools are available.",
        },
    )
    monkeypatch.setattr(
        "src.webapp.get_hg38_reference_status",
        lambda: {
            "ready": True,
            "fasta": "data/reference/hg38/hg38.analysisSet.fa",
            "fai": "data/reference/hg38/hg38.analysisSet.fa.fai",
            "message": "Reference FASTA and index are ready.",
        },
    )
    monkeypatch.setattr(
        "src.webapp.extract_region_vcf",
        lambda **kwargs: {
            "output_vcf": output_path,
            "resolved_region": "chr15:21405401-21441499",
            "commands": {"mpileup": "bcftools mpileup ...", "call": "bcftools call ..."},
        },
    )

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "extraction",
            "extraction_gene_name": "POTEB3",
            "extraction_genome_build": "hg38",
            "extraction_scope": "promoter_plus_gene",
            "extraction_region": "15:21405401-21441499",
            "bam_path": "data/sample.hg38.bam",
            "output_vcf": "data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz",
            "extraction_action": "extract_vcf",
        },
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-tab-target="analysis"' in page
    assert 'value="data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz"' in page
    assert 'value="chr15:21405401-21441499"' in page

    with client.session_transaction() as session_state:
        preprocess_state = session_state["preprocess_state"]

    assert preprocess_state["gene_name"] == "POTEB3"
    assert preprocess_state["analysis_ready"] is True
    assert preprocess_state["build"] == "hg38"
    assert preprocess_state["region"] == "chr15:21405401-21441499"


def test_app_structure_page_includes_general_probe_mapping_qa(monkeypatch) -> None:
    """The App Structure page should include the general literature-to-probe mapping explanation."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    client = app.test_client()
    response = client.get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "How are gene regions retrieved, and how are promoter-only, gene-only, and promoter+gene scopes computed?" in page
    assert "NCBI Entrez Gene, Ensembl GRCh37, and UCSC hg19" in page
    assert "UCSC now uses the assembly-wide" in page
    assert "SIRT6, where the valid standard region is" in page
    assert "How are the local databases built from literature, and how does a probe get mapped to a locus or variant?" in page
    assert "the current whitelist probes are usually not stored as" in page
    assert "SIRT6 on the reverse strand" in page
    assert "That nearby-locus column is purely manifest-derived proximity annotation" in page
    assert "How do I use the Knowledge Sources tab and dynamic variant knowledge-base preprocessing?" in page
    assert "Workflow cards sit above the database cards" in page
    assert "Core safety workflows are checked by default" in page
    assert "Use local PDF article folder" in page
    assert "rather than full article text" in page
    assert "Dynamic Workflow Summary sections" in page
    assert "Build Variant Knowledge Base" in page
    assert "results/dynamic_knowledge_bases/" in page
    assert "How do licensed source exports, API credentials, and the API/CLI dynamic KB workflow work?" in page
    assert "The app does not scrape Google Scholar, HGMD, GeneCards, VarSome, Franklin, Mastermind" in page
    assert "`source_key`, `record_id`, `gene`, `variant`, `rsid`" in page
    assert "GET /api/v1/knowledge-sources" in page
    assert "GET /api/v1/knowledge-workflows" in page
    assert "options.knowledge_workflows" in page
    assert "options.knowledge_source_imports" in page
    assert "options.article_pdf_folder" in page


def test_history_tab_lists_saved_reports_and_serves_artifacts(monkeypatch, tmp_path: Path) -> None:
    """The History tab should surface prior reports with openable links."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    report_path = results_dir / "igf1r_report.html"
    report_path.write_text("<html><body>IGF1R report</body></html>", encoding="utf-8")
    methylation_path = results_dir / "igf1r_report_methylation.csv"
    methylation_path.write_text("probe_id,beta\ncg1,0.42\n", encoding="utf-8")

    monkeypatch.setattr("src.webapp.RESULTS_DIR", results_dir)
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])

    client = app.test_client()
    response = client.get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-tab-target="history"' in page
    assert "Human Genes & Proteins" in page
    assert 'data-protein-exclude-processed' in page
    assert "Exclude already processed proteins" in page
    assert '["IGF1R"]' in page
    assert "igf1r_report.html" in page
    assert "/results/igf1r_report.html" in page
    assert "/results/igf1r_report_methylation.csv" in page

    artifact_response = client.get("/results/igf1r_report.html")
    assert artifact_response.status_code == 200
    assert "IGF1R report" in artifact_response.get_data(as_text=True)


def test_functional_map_groups_genes_and_links_processed_reports(monkeypatch, tmp_path: Path) -> None:
    """The functional map should group knowledge-base genes and link completed reports."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    report_path = results_dir / "sirt6_report.html"
    report_path.write_text("<html><body>SIRT6 report</body></html>", encoding="utf-8")

    monkeypatch.setattr("src.webapp.RESULTS_DIR", results_dir)
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])

    client = app.test_client()
    response = client.get("/functional-map")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "NophiGene Functional Map" in page
    assert "Longevity &amp; Healthy Aging" in page
    assert "Senses &amp; Sensory Signaling" in page
    assert "Asthma, Allergy &amp; Airways" in page
    assert 'value="SIRT6"' in page
    assert "/results/sirt6_report.html" in page
    assert "Open latest report" in page


def test_functional_map_selection_seeds_preprocessing(monkeypatch) -> None:
    """Selecting a map gene should load its curated interval into preprocessing."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_bam_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    client = app.test_client()
    response = client.post(
        "/",
        data={
            "workflow": "functional_map",
            "functional_gene_name": "SIRT6",
        },
    )
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-initial-tab="preprocessing"' in page
    assert "Selected SIRT6 from Functional Map." in page
    assert 'value="SIRT6"' in page

    with client.session_transaction() as session_state:
        preprocess_state = session_state["preprocess_state"]

    assert preprocess_state["gene_name"] == "SIRT6"
    assert preprocess_state["region"] == "19:4174106-4183560"
    assert preprocess_state["scope_regions"]["promoter_only"] == "19:4182561-4183560"
    assert preprocess_state["scope_regions"]["gene_only"] == "19:4174106-4182560"
    assert preprocess_state["region_ready"] is True
    assert preprocess_state["manifest_ready"] is False
    assert preprocess_state["analysis_ready"] is False


def test_functional_map_prioritizes_requested_families() -> None:
    """High-signal descriptions should land in the intended functional families."""
    assert _classify_functional_family("human longevity and centenarian healthy aging") == "longevity"
    assert _classify_functional_family("human sensory biology and visual phototransduction") == "senses"
    assert _classify_functional_family("asthma allergy airway eosinophil biology") == "asthma_allergy"


def test_central_database_tab_displays_general_database(monkeypatch, tmp_path: Path) -> None:
    """The Central Database tab should render the one-row-per-observed-variant CSV."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    database_path = results_dir / "general_gene_analysis_database.csv"
    pd.DataFrame(
        [
            {
                "gene": "HERC2",
                "variant key": "chr15:28365618:A>G",
                "observed gene variant": "rs12913832",
                "gene variant label": "rs12913832",
                "change": "A -> G",
                "chromosome": "chr15",
                "position": 28365618,
                "variant location": "chr15:28,365,618",
                "gene location": "chr15:28,356,186-28,567,325",
                "source": "VCF",
                "(VCF) quality (qual)": 88.0,
                "matched curated marker": "HERC2/OCA2 enhancer rs12913832",
                "variant interpretation scope": "Regulatory pigmentation marker",
                "curated biological significance": "Research marker for iris pigmentation biology.",
                "functional effects": "OCA2 enhancer activity",
                "associated conditions": "iris pigmentation",
                "methylation-linked probes": "cg00000001",
                "mean beta whitelist": 0.71,
                "mean beta related to gene": 0.62,
                "mean beta on found probes in the area (numerical rows)": 0.53,
            },
            {
                "gene": "HERC2",
                "variant key": "chr15:28356859:C>T",
                "observed gene variant": "rs1129038",
                "gene variant label": "None",
                "change": "C -> T",
                "chromosome": "chr15",
                "position": 28356859,
                "variant location": "chr15:28,356,859",
                "gene location": "chr15:28,356,186-28,567,325",
                "source": "VCF",
                "(VCF) quality (qual)": 74.0,
                "matched curated marker": "",
                "variant interpretation scope": "Unclassified observed variant",
                "curated biological significance": "No curated local HERC2 significance is bundled for this observed variant.",
                "functional effects": "",
                "associated conditions": "",
                "methylation-linked probes": "",
                "mean beta whitelist": 0.71,
                "mean beta related to gene": 0.62,
                "mean beta on found probes in the area (numerical rows)": 0.53,
            }
        ]
    ).to_csv(database_path, index=False)

    monkeypatch.setattr("src.webapp.RESULTS_DIR", results_dir)
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])

    client = app.test_client()
    response = client.get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'data-tab-target="central_database"' in page
    assert "Central Analysis Database" in page
    assert "HERC2" in page
    assert "rs12913832" in page
    assert "rs1129038" in page
    assert "Only labeled variants" in page
    assert 'data-variant-labeled="true"' in page
    assert 'data-variant-labeled="false"' in page
    assert "variant key" in page
    assert "curated biological significance" in page
    assert "OCA2 enhancer activity" in page
    assert "(VCF) quality (qual)" in page
    assert "/results/general_gene_analysis_database.csv" in page
    assert "No reports yet" in page

    database_response = client.get("/results/general_gene_analysis_database.csv")
    assert database_response.status_code == 200
    assert "HERC2" in database_response.get_data(as_text=True)


def test_analysis_result_keeps_full_curated_methylation_probe_preview(monkeypatch, tmp_path: Path) -> None:
    """The curated methylation probe preview should include every row used for interpretation."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    captured_context: dict[str, object] = {}

    def fake_render_template(template_name: str, **context: object) -> str:
        captured_context["template_name"] = template_name
        captured_context["context"] = context
        return "ok"

    probe_preview = pd.DataFrame(
        [
            {
                "probe_id": f"cg{i:08d}",
                "beta": 0.8 + i / 1000,
                "UCSC_RefGene_Group": "Body",
                "Relation_to_UCSC_CpG_Island": "Island",
                "UCSC_CpG_Islands_Name": "TEST_CGI",
            }
            for i in range(15)
        ]
    )

    monkeypatch.setattr("src.webapp.render_template", fake_render_template)
    captured_run_kwargs: dict[str, object] = {}
    mock_analysis_result = SimpleNamespace(
            report_path=tmp_path / "mock_report.html",
            methylation_output_path=tmp_path / "mock_report_methylation.csv",
            variants=pd.DataFrame(
                [{"chrom": "1", "id": "rs1", "pos": 1, "ref": "A", "alt": "G", "qual": 99.0, "filter_pass": True}]
            ),
            methylation=probe_preview.copy(),
            popstats=None,
            knowledge_base={"database_name": "Mock interpretation DB", "version": "test"},
            population_database={"database_name": "Mock population DB", "version": "test"},
            population_insights={"variant_population_records": [], "gene_population_patterns": []},
            variant_interpretations={"gene_name": "TEST", "matched_records": [], "sample_highlights": {"summary": ""}},
            methylation_insights={
                "gene_name": "TEST",
                "clinical_context": "",
                "summary": "",
                "mean_beta": 0.9,
                "mean_beta_label": "Whitelist mean beta",
                "mean_beta_probe_count": 15,
                "whitelist_mean_beta": 0.9,
                "whitelist_mean_beta_label": "Whitelist mean beta",
                "whitelist_mean_beta_probe_count": 15,
                "whitelist_probe_count": 15,
                "whitelist_observed_probe_count": 15,
                "whitelist_explanation": "Mock whitelist explanation",
                "whitelist_literature_context": "Mock literature context",
                "whitelist_probe_statuses": [
                    {"probe_id": probe_id, "observed_in_run": True}
                    for probe_id in probe_preview["probe_id"].tolist()
                ],
                "whitelist_probe_reference_rows": [
                    {
                        "probe_id": "cg00000000",
                        "observed_in_run": True,
                        "beta": 0.9,
                        "probe_locus": "chr1:1",
                        "linked_variants": [{"label": "rs1", "common_name": "mock variant", "locus": "chr1:1"}],
                        "nearby_manifest_variants": [{"variant": "rs1", "distance": "0"}],
                        "papers": [{"label": "Mock Paper", "url": "https://example.com/mock", "source_variant": "rs1"}],
                    }
                ],
                "whitelist_probe_reference_summary": "Mock whitelist reference summary",
                "gene_name_mean_beta": 0.9,
                "gene_name_mean_beta_label": "TEST-named row mean beta",
                "gene_name_mean_beta_probe_count": 15,
                "gene_name_row_count": 15,
                "gene_name_match_columns": ["GencodeBasicV12_NAME"],
                "gene_name_match_rule": "Mock gene-name rule",
                "raw_mean_beta": 0.9,
                "raw_mean_beta_label": "All numeric-row mean beta",
                "raw_probe_count": 15,
                "raw_mean_beta_probe_count": 15,
                "all_numeric_mean_beta": 0.9,
                "all_numeric_mean_beta_label": "All numeric-row mean beta",
                "all_numeric_mean_beta_probe_count": 15,
                "beta_band": "high",
                "beta_band_source_label": "Whitelist mean beta",
                "observed_probe_count": 15,
                "curated_probe_count": 15,
                "probe_ids": probe_preview["probe_id"].tolist(),
                "group_breakdown": {"Body": 15},
                "methylation_effects": [],
                "methylation_condition_research": [],
                "evidence": [],
                "probe_preview": probe_preview,
            },
            predictive_theses={
                "gene_name": "TEST",
                "database_version": "test",
                "variant_found_label": "Yes",
                "matched_case_count": 1,
                "case_catalog_size": 10,
                "summary": "Mock predictive summary",
                "variant_summary": "Mock variant summary",
                "matching_rule": "Mock matching rule",
                "disclaimer": "Mock disclaimer",
                "seeded_markers": ["rs1"],
                "variant_prediction_rows": [
                    {
                        "observed_signal": "rs1",
                        "source": "Gene-level thesis",
                        "prediction": "Mock variant prediction",
                        "research_focus": "Mock focus",
                    }
                ],
                "methylation_prediction_rows": [
                    {
                        "metric_label": "Whitelist mean beta",
                        "probe_count": 15,
                        "mean_beta_display": "0.9",
                        "band_display": "High",
                        "prediction": "Mock methylation prediction",
                        "matched_case_label": "Gene variant found + high whitelist mean beta",
                        "research_focus": "Mock focus",
                    }
                ],
                "matched_cases": [
                    {
                        "case_label": "Gene variant found",
                        "trigger": "Observed promoter or gene-body variant",
                        "source": "Variant-only synthesis",
                        "mean_beta_display": "n/a",
                        "band": "n/a",
                        "prediction": "Mock synthesis prediction",
                        "research_focus": "Mock focus",
                    }
                ],
            },
    )

    def fake_run_analysis(**kwargs: object) -> SimpleNamespace:
        captured_run_kwargs.update(kwargs)
        return mock_analysis_result

    monkeypatch.setattr("src.webapp.run_analysis", fake_run_analysis)

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "TEST",
            "region": "1:1-100",
            "manifest_source": "",
            "filtered_manifest": "",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": True,
            "analysis_ready": True,
            "probe_count": 15,
            "build": "hg19",
            "logs": [],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": False,
        }

    response = client.post(
        "/",
        data={
            "workflow": "analysis",
            "vcf": "data/mock.vcf.gz",
            "idat": "data/mock_sample",
            "out": "results/mock_report.html",
            "analysis_scope": "promoter_only",
            "region": "1:1-100",
            "popstats": "",
            "manifest_file": "",
            "overwrite_general_database": "1",
        },
    )

    assert response.status_code == 200
    result = captured_context["context"]["result"]
    probe_preview_html = result["methylation_insights"]["probe_preview"]
    assert probe_preview_html.count("<tr") >= 16
    assert "cg00000000" in probe_preview_html
    assert "cg00000014" in probe_preview_html
    assert result["predictive_theses"]["matched_case_count"] == 1
    assert result["predictive_theses"]["variant_prediction_rows"][0]["prediction"] == "Mock variant prediction"
    assert result["analysis_scope_label"] == "Promoter only"
    assert result["data_sources"]["total_cards"] >= 4
    assert result["data_sources"]["dynamic_status"] == "not_run"
    assert captured_run_kwargs["analysis_scope"] == "promoter_only"
    assert captured_run_kwargs["overwrite_general_database"] is True


def test_analysis_result_labels_missing_variant_ids_in_preview(monkeypatch, tmp_path: Path) -> None:
    """Variant previews should explain when the source VCF does not provide a named ID."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    captured_context: dict[str, object] = {}

    def fake_render_template(template_name: str, **context: object) -> str:
        captured_context["template_name"] = template_name
        captured_context["context"] = context
        return "ok"

    monkeypatch.setattr("src.webapp.render_template", fake_render_template)
    monkeypatch.setattr(
        "src.webapp.run_analysis",
        lambda **_: SimpleNamespace(
            report_path=tmp_path / "mock_report.html",
            methylation_output_path=tmp_path / "mock_report_methylation.csv",
            variants=pd.DataFrame(
                [{"chrom": "11", "id": None, "pos": 636689, "ref": "G", "alt": "C", "qual": 75.35, "filter_pass": True}]
            ),
            methylation=pd.DataFrame([{"probe_id": "cg1", "beta": 0.42}]),
            popstats=None,
            knowledge_base={"database_name": "Mock interpretation DB", "version": "test"},
            population_database={"database_name": "Mock population DB", "version": "test"},
            population_insights={"variant_population_records": [], "gene_population_patterns": []},
            variant_interpretations={"gene_name": "TEST", "matched_records": [], "sample_highlights": {"summary": ""}},
            methylation_insights={
                "gene_name": "TEST",
                "clinical_context": "",
                "summary": "",
                "mean_beta": 0.42,
                "mean_beta_label": "Whitelist mean beta",
                "mean_beta_probe_count": 1,
                "whitelist_mean_beta": 0.42,
                "whitelist_mean_beta_label": "Whitelist mean beta",
                "whitelist_mean_beta_probe_count": 1,
                "whitelist_probe_count": 1,
                "whitelist_observed_probe_count": 1,
                "whitelist_explanation": "",
                "whitelist_literature_context": "",
                "whitelist_probe_statuses": [{"probe_id": "cg1", "observed_in_run": True}],
                "whitelist_probe_reference_rows": [],
                "whitelist_probe_reference_summary": "",
                "gene_name_mean_beta": 0.42,
                "gene_name_mean_beta_label": "TEST-named row mean beta",
                "gene_name_mean_beta_probe_count": 1,
                "gene_name_row_count": 1,
                "gene_name_match_columns": [],
                "gene_name_match_rule": "",
                "raw_mean_beta": 0.42,
                "raw_mean_beta_label": "All numeric-row mean beta",
                "raw_probe_count": 1,
                "raw_mean_beta_probe_count": 1,
                "all_numeric_mean_beta": 0.42,
                "all_numeric_mean_beta_label": "All numeric-row mean beta",
                "all_numeric_mean_beta_probe_count": 1,
                "beta_band": "intermediate",
                "beta_band_source_label": "Whitelist mean beta",
                "observed_probe_count": 1,
                "curated_probe_count": 1,
                "probe_ids": ["cg1"],
                "group_breakdown": {},
                "methylation_effects": [],
                "methylation_condition_research": [],
                "evidence": [],
                "probe_preview": pd.DataFrame([{"probe_id": "cg1", "beta": 0.42}]),
            },
        ),
    )

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "TEST",
            "region": "11:636000-637000",
            "manifest_source": "",
            "filtered_manifest": "",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": True,
            "analysis_ready": True,
            "probe_count": 1,
            "build": "hg19",
            "logs": [],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": False,
        }

    response = client.post(
        "/",
        data={
            "workflow": "analysis",
            "vcf": "data/mock.vcf.gz",
            "idat": "data/mock_sample",
            "out": "results/mock_report.html",
            "region": "11:636000-637000",
            "popstats": "",
            "manifest_file": "",
        },
    )

    assert response.status_code == 200
    result = captured_context["context"]["result"]
    assert "Unlabeled in source VCF" in result["variant_preview"]
    assert result["variant_rows"][0]["id"] == "Unlabeled in source VCF"
    assert result["variant_raw_page_size"] == 25


def test_analysis_result_keeps_full_variant_rows_for_pagination(monkeypatch, tmp_path: Path) -> None:
    """The raw variant panel should keep all rows for client-side pagination."""
    monkeypatch.setattr("src.webapp.discover_vcf_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_idat_prefixes", lambda: [])
    monkeypatch.setattr("src.webapp.discover_population_stats_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_manifest_files", lambda: [])
    monkeypatch.setattr("src.webapp.discover_report_history", lambda: [])

    captured_context: dict[str, object] = {}

    def fake_render_template(template_name: str, **context: object) -> str:
        captured_context["template_name"] = template_name
        captured_context["context"] = context
        return "ok"

    variant_rows = pd.DataFrame(
        [
            {
                "chrom": "15",
                "id": f"rs{i:05d}",
                "pos": i,
                "ref": "A",
                "alt": "G",
                "qual": 99.0,
                "filter_pass": True,
            }
            for i in range(1, 31)
        ]
    )

    monkeypatch.setattr("src.webapp.render_template", fake_render_template)
    monkeypatch.setattr(
        "src.webapp.run_analysis",
        lambda **_: SimpleNamespace(
            report_path=tmp_path / "mock_report.html",
            methylation_output_path=tmp_path / "mock_report_methylation.csv",
            variants=variant_rows.copy(),
            methylation=pd.DataFrame([{"probe_id": "cg1", "beta": 0.42}]),
            popstats=None,
            knowledge_base={"database_name": "Mock interpretation DB", "version": "test"},
            population_database={"database_name": "Mock population DB", "version": "test"},
            population_insights={"variant_population_records": [], "gene_population_patterns": []},
            variant_interpretations={"gene_name": "TEST", "matched_records": [], "sample_highlights": {"summary": ""}},
            methylation_insights={
                "gene_name": "TEST",
                "clinical_context": "",
                "summary": "",
                "mean_beta": 0.42,
                "mean_beta_label": "Whitelist mean beta",
                "mean_beta_probe_count": 1,
                "whitelist_mean_beta": 0.42,
                "whitelist_mean_beta_label": "Whitelist mean beta",
                "whitelist_mean_beta_probe_count": 1,
                "whitelist_probe_count": 1,
                "whitelist_observed_probe_count": 1,
                "whitelist_explanation": "",
                "whitelist_literature_context": "",
                "whitelist_probe_statuses": [{"probe_id": "cg1", "observed_in_run": True}],
                "whitelist_probe_reference_rows": [],
                "whitelist_probe_reference_summary": "",
                "gene_name_mean_beta": 0.42,
                "gene_name_mean_beta_label": "TEST-named row mean beta",
                "gene_name_mean_beta_probe_count": 1,
                "gene_name_row_count": 1,
                "gene_name_match_columns": [],
                "gene_name_match_rule": "",
                "raw_mean_beta": 0.42,
                "raw_mean_beta_label": "All numeric-row mean beta",
                "raw_probe_count": 1,
                "raw_mean_beta_probe_count": 1,
                "all_numeric_mean_beta": 0.42,
                "all_numeric_mean_beta_label": "All numeric-row mean beta",
                "all_numeric_mean_beta_probe_count": 1,
                "beta_band": "intermediate",
                "beta_band_source_label": "Whitelist mean beta",
                "observed_probe_count": 1,
                "curated_probe_count": 1,
                "probe_ids": ["cg1"],
                "group_breakdown": {},
                "methylation_effects": [],
                "methylation_condition_research": [],
                "evidence": [],
                "probe_preview": pd.DataFrame([{"probe_id": "cg1", "beta": 0.42}]),
            },
        ),
    )

    client = app.test_client()
    with client.session_transaction() as session_state:
        session_state["preprocess_state"] = {
            "gene_name": "TEST",
            "region": "15:1-1000",
            "manifest_source": "",
            "filtered_manifest": "",
            "region_candidates": [],
            "selected_sources": [],
            "region_ready": True,
            "manifest_ready": True,
            "analysis_ready": True,
            "probe_count": 1,
            "build": "hg19",
            "logs": [],
            "region_recently_updated": False,
            "overwrite_filtered_manifest": False,
        }

    response = client.post(
        "/",
        data={
            "workflow": "analysis",
            "vcf": "data/mock.vcf.gz",
            "idat": "data/mock_sample",
            "out": "results/mock_report.html",
            "region": "15:1-1000",
            "popstats": "",
            "manifest_file": "",
        },
    )

    assert response.status_code == 200
    result = captured_context["context"]["result"]
    assert len(result["variant_rows"]) == 30
    assert "rs00001" in result["variant_preview"]
    assert "rs00025" in result["variant_preview"]
    assert "rs00026" not in result["variant_preview"]
