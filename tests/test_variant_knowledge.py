import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.variant_knowledge import client as request_client_module
from src.variant_knowledge.client import KnowledgeRequestError, RequestClient
from src.variant_knowledge.connectors import connector_for
from src.variant_knowledge.credentials import ResolvedCredential, credential_status_for_specs
from src.variant_knowledge.imports import parse_source_import
from src.variant_knowledge.local_articles import (
    LOCAL_ARTICLE_SOURCE_KEY,
    LOCAL_ARTICLE_WORKFLOW_KEY,
    extract_local_article_evidence,
)
from src.variant_knowledge.merger import merge_dynamic_knowledge_base
from src.variant_knowledge.models import KnowledgeQuery, QueryVariant
from src.variant_knowledge.orchestrator import build_dynamic_knowledge_base
from src.variant_knowledge.registry import RESOURCES_PATH, get_source_spec, list_source_cards, list_source_specs
from src.variant_knowledge.workflows import CORE_SAFETY_WORKFLOW_KEYS, list_workflow_specs


class FakeClient:
    def get_json(self, url, *, params=None, headers=None, rate_limit_per_second=None):
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": ["123"]}}
        if "esummary.fcgi" in url:
            return {
                "result": {
                    "123": {
                        "uid": "123",
                        "title": "ClinVar rs123 pathogenic record",
                        "description": "Pathogenic clinical assertion for rs123",
                    }
                }
            }
        raise AssertionError(f"Unexpected fake GET URL: {url}")

    def post_json(self, url, *, json_payload=None, headers=None, rate_limit_per_second=None):
        raise AssertionError(f"Unexpected fake POST URL: {url}")


class CountingFakeClient(FakeClient):
    def __init__(self) -> None:
        self.get_counts: dict[str, int] = {}

    def get_json(self, url, *, params=None, headers=None, rate_limit_per_second=None):
        if "esearch.fcgi" in url:
            term = str((params or {}).get("term", ""))
            self.get_counts[term] = self.get_counts.get(term, 0) + 1
        return super().get_json(
            url,
            params=params,
            headers=headers,
            rate_limit_per_second=rate_limit_per_second,
        )


class RecordingClinVarClient:
    def __init__(
        self,
        *,
        ids_by_term: dict[str, list[str]] | None = None,
        summaries: dict[str, dict[str, object]] | None = None,
        clinical_tables_payload: list[object] | None = None,
    ) -> None:
        self.ids_by_term = ids_by_term or {}
        self.summaries = summaries or {}
        self.clinical_tables_payload = clinical_tables_payload or [0, [], {}, []]
        self.calls: list[dict[str, object]] = []

    def get_json(self, url, *, params=None, headers=None, rate_limit_per_second=None):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params})
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": self.ids_by_term.get(str(params.get("term", "")), [])}}
        if "esummary.fcgi" in url:
            return {
                "result": {
                    item_id: self.summaries[item_id]
                    for item_id in str(params.get("id", "")).split(",")
                    if item_id in self.summaries
                }
            }
        if "clinicaltables.nlm.nih.gov" in url:
            return self.clinical_tables_payload
        raise AssertionError(f"Unexpected ClinVar fake GET URL: {url}")

    def post_json(self, url, *, json_payload=None, headers=None, rate_limit_per_second=None):
        raise AssertionError(f"Unexpected fake POST URL: {url}")


class RecordingClinGenClient:
    VALIDITY_CSV = (
        '"CLINGEN GENE DISEASE VALIDITY CURATIONS","",""\n'
        '"GENE SYMBOL","GENE ID (HGNC)","DISEASE LABEL","DISEASE ID (MONDO)","MOI","SOP","CLASSIFICATION","ONLINE REPORT","CLASSIFICATION DATE","GCEP"\n'
        '"+++++++++++","++++","++++","++++","++++","++++","++++","++++","++++","++++"\n'
        '"GENE1","HGNC:1","Example syndrome","MONDO:0000001","AD","SOP10","Definitive","https://search.clinicalgenome.org/kb/gene-validity/GENE1","2026-01-01T00:00:00.000Z","Example GCEP"\n'
    )
    DOSAGE_CSV = (
        '"CLINGEN DOSAGE SENSITIVITY CURATIONS","",""\n'
        '"GENE SYMBOL","HGNC ID","HAPLOINSUFFICIENCY","TRIPLOSENSITIVITY","ONLINE REPORT","DATE"\n'
        '"+++++++++++","++++","++++","++++","++++","++++"\n'
        '"GENE1","HGNC:1","Sufficient Evidence for Haploinsufficiency","No Evidence for Triplosensitivity","https://search.clinicalgenome.org/kb/gene-dosage/HGNC:1","2026-01-02T00:00:00+00:00"\n'
    )
    SUMMARY_CSV = (
        '"README","",""\n'
        '"gene_symbol","hgnc_id","gene_url","disease_label","mondo_id","disease_url","mode_of_inheritance","dosage_haploinsufficiency_assertion","dosage_triplosensitivity_assertion","dosage_report","dosage_group","gene_disease_validity_assertion_classifications","gene_disease_validity_assertion_reports","gene_disease_validity_gceps","actionability_assertion_classifications","actionability_assertion_reports","actionability_groups"\n'
        '"GENE1","HGNC:1","https://search.clinicalgenome.org/kb/genes/HGNC:1","Example syndrome","MONDO:0000001","https://search.clinicalgenome.org/kb/conditions/MONDO:0000001","Autosomal dominant inheritance","Sufficient Evidence for Haploinsufficiency","No Evidence for Triplosensitivity","https://search.clinicalgenome.org/kb/gene-dosage/HGNC:1","Dosage Working Group","Definitive","https://search.clinicalgenome.org/kb/gene-validity/GENE1","Example GCEP","Actionable","https://actionability.clinicalgenome.org/ac/Adult/ui/summ","Example Actionability Group"\n'
    )
    ACTIONABILITY_JSON = {
        "columns": [
            "docId",
            "contextIri",
            "context",
            "releaseDate",
            "geneOrVariant",
            "disease",
            "status-overall",
            "outcome",
            "intervention",
            "severity",
            "likelihood",
            "natureOfIntervention",
            "effectiveness",
            "overall",
        ],
        "rows": [
            [
                "AC1",
                "https://actionability.clinicalgenome.org/ac/Adult/api/sepio/doc/AC1",
                "Adult",
                "Thu, 01 Jan 2026 00:00:00 -0000",
                "GENE1",
                "Example syndrome",
                "Released",
                "Preventable morbidity",
                "Surveillance",
                "2",
                "3C",
                "3",
                "2N",
                "10CN",
            ]
        ],
    }

    def __init__(self, *, empty: bool = False, malformed_validity: bool = False, fail_all: bool = False) -> None:
        self.empty = empty
        self.malformed_validity = malformed_validity
        self.fail_all = fail_all
        self.calls: list[str] = []

    def get_text(self, url, *, params=None, headers=None, rate_limit_per_second=None, timeout=None):
        self.calls.append(str(url))
        if self.fail_all:
            raise KnowledgeRequestError(
                "GET ClinGen failed: TLS certificate verification failed.",
                code="tls_certificate_verification_failed",
                remediation="Configure NOPHIGENE_CA_BUNDLE.",
            )
        if "gene-validity/download" in url:
            return "not,a,valid,clingen,file\n" if self.malformed_validity else self._maybe_empty(self.VALIDITY_CSV)
        if "gene-dosage/download" in url:
            return self._maybe_empty(self.DOSAGE_CSV)
        if "curation-activity-summary-report" in url:
            return self._maybe_empty(self.SUMMARY_CSV)
        raise AssertionError(f"Unexpected ClinGen fake text URL: {url}")

    def get_json(self, url, *, params=None, headers=None, rate_limit_per_second=None, timeout=None):
        self.calls.append(str(url))
        if self.fail_all:
            raise KnowledgeRequestError(
                "GET ClinGen failed: TLS certificate verification failed.",
                code="tls_certificate_verification_failed",
                remediation="Configure NOPHIGENE_CA_BUNDLE.",
            )
        return {"columns": self.ACTIONABILITY_JSON["columns"], "rows": []} if self.empty else self.ACTIONABILITY_JSON

    def post_json(self, url, *, json_payload=None, headers=None, rate_limit_per_second=None, timeout=None):
        raise AssertionError(f"Unexpected fake POST URL: {url}")

    def _maybe_empty(self, csv_text: str) -> str:
        if not self.empty:
            return csv_text
        line_count = 2 if csv_text.startswith('"README"') else 3
        return "\n".join(csv_text.splitlines()[:line_count]) + "\n"


def _write_simple_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
    payload = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
        f"4 0 obj << /Length {len(stream)} >> stream\n"
        f"{stream}\n"
        "endstream endobj\n"
        "trailer << /Root 1 0 R >>\n"
        "%%EOF\n"
    )
    path.write_bytes(payload.encode("latin-1", errors="ignore"))


def test_registry_has_one_card_per_resource_entry():
    active_resource_lines = [
        line
        for line in RESOURCES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    specs = list_source_specs()

    assert len(specs) == len(active_resource_lines)
    assert len({spec.key for spec in specs}) == len(active_resource_lines)
    assert all(spec.name for spec in specs)
    assert all(spec.connector_kind for spec in specs)


def test_non_open_sources_advertise_compliant_ingestion_modes_only():
    target_keys = {
        "google_scholar",
        "hgmd",
        "genecards",
        "varsome",
        "franklin",
        "mastermind",
        "embase",
        "scopus",
        "web_of_science",
    }
    specs = {spec.key: spec for spec in list_source_specs()}
    cards = {card["key"]: card for card in list_source_cards()}

    for key in target_keys:
        spec = specs[key]
        assert "scrap" not in spec.connector_kind.lower()
        assert "user_export" in spec.ingestion_modes
        assert "linkout_only" in spec.ingestion_modes
        assert set(spec.ingestion_modes) <= {"official_api", "user_export", "linkout_only"}
        assert cards[key]["import_schema"]
        assert cards[key]["accepted_import_formats"] == ["csv", "json"]


def test_workflow_registry_references_valid_sources_and_core_defaults():
    source_keys = {spec.key for spec in list_source_specs()}
    synthetic_local_sources = {LOCAL_ARTICLE_SOURCE_KEY}
    workflows = list_workflow_specs()

    assert [workflow.key for workflow in workflows if workflow.default_selected] == list(CORE_SAFETY_WORKFLOW_KEYS)
    assert workflows
    for workflow in workflows:
        assert workflow.label
        assert workflow.purpose
        assert workflow.report_section
        assert set(workflow.ordered_source_keys) <= source_keys | synthetic_local_sources
        if workflow.key == LOCAL_ARTICLE_WORKFLOW_KEY:
            assert workflow.ordered_source_keys == (LOCAL_ARTICLE_SOURCE_KEY,)
            assert not workflow.default_selected
        if workflow.key == "licensed_aggregator_review":
            joined_notes = " ".join(workflow.licensed_notes).lower()
            assert "scraping" in joined_notes
            assert "captcha" in joined_notes


def test_request_client_prefers_explicit_ca_bundle(monkeypatch, tmp_path: Path):
    ca_bundle = tmp_path / "custom-ca.pem"
    ca_bundle.write_text("", encoding="ascii")
    monkeypatch.setenv("NOPHIGENE_CA_BUNDLE", str(ca_bundle))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "other-ca.pem"))

    client = RequestClient()

    assert client.verify == str(ca_bundle)


def test_request_client_uses_windows_merged_ca_bundle(monkeypatch, tmp_path: Path):
    ca_bundle = tmp_path / "windows-merged-ca.pem"
    ca_bundle.write_text("", encoding="ascii")
    for env_var in request_client_module.EXPLICIT_CA_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(request_client_module.os, "name", "nt")
    monkeypatch.setattr(request_client_module, "_windows_merged_ca_bundle", lambda: str(ca_bundle))

    client = RequestClient()

    assert client.verify == str(ca_bundle)


def test_request_client_ssl_errors_are_redacted_and_actionable():
    client = RequestClient()

    def fail_get(url, *, params=None, headers=None, timeout=None, verify=None):
        raise requests.exceptions.SSLError(
            "certificate verify failed for https://example.test/?api_key=SECRET&token=TOKEN"
        )

    client.session.get = fail_get

    with pytest.raises(KnowledgeRequestError) as exc_info:
        client.get_json("https://example.test/endpoint", params={"api_key": "SECRET"})

    message = str(exc_info.value)
    assert exc_info.value.code == "tls_certificate_verification_failed"
    assert "NOPHIGENE_CA_BUNDLE" in message
    assert "SECRET" not in message
    assert "TOKEN" not in message
    assert "api_key=[redacted]" in message
    assert "token=[redacted]" in message


def test_clinvar_connector_uses_fielded_gene_query_and_ncbi_params(monkeypatch):
    monkeypatch.setenv("NOPHIGENE_NCBI_EMAIL", "researcher@example.org")
    monkeypatch.setenv("NOPHIGENE_NCBI_API_KEY", "secret-ncbi-key")
    client = RecordingClinVarClient(
        ids_by_term={"DRD4[gene] AND single_gene[prop]": ["123"]},
        summaries={
            "123": {
                "uid": "123",
                "title": "DRD4 variant",
                "clinical_significance": {"description": "Pathogenic"},
                "trait_set": [{"trait_name": "Example condition"}],
                "variation_set": [
                    {
                        "variation_name": "DRD4 c.1A>G",
                        "variation_xrefs": [{"db_source": "dbSNP", "db_id": "1800955"}],
                    }
                ],
            }
        },
    )
    connector = connector_for(get_source_spec("clinvar"), client, ResolvedCredential("clinvar"))

    result = connector.query(KnowledgeQuery(gene="DRD4", region="11:1-10", genome_build="hg19"))

    search_calls = [call for call in client.calls if "esearch.fcgi" in str(call["url"])]
    assert search_calls[0]["params"]["term"] == "DRD4[gene] AND single_gene[prop]"
    assert search_calls[0]["params"]["tool"] == "NophiGeneDynamicKB"
    assert search_calls[0]["params"]["email"] == "researcher@example.org"
    assert search_calls[0]["params"]["api_key"] == "secret-ncbi-key"
    assert result.status == "ok"
    assert result.records[0]["clinical_significance"] == "Pathogenic"
    assert result.records[0]["rsid"] == "rs1800955"
    assert result.records[0]["url"] == "https://www.ncbi.nlm.nih.gov/clinvar/variation/123/"
    serialized = json.dumps(result.to_status(get_source_spec("clinvar")), sort_keys=True)
    assert "secret-ncbi-key" not in serialized


def test_clinvar_connector_queries_rsid_first_and_deduplicates_gene_results():
    client = RecordingClinVarClient(
        ids_by_term={
            "rs123": ["123"],
            "DRD4[gene] AND single_gene[prop]": ["123"],
        },
        summaries={
            "123": {
                "uid": "123",
                "title": "ClinVar rs123 record",
                "description": "Clinical assertion",
            }
        },
    )
    connector = connector_for(get_source_spec("clinvar"), client, ResolvedCredential("clinvar"))
    query = KnowledgeQuery(
        gene="DRD4",
        region="11:1-10",
        genome_build="hg19",
        variants=(QueryVariant(chrom="11", pos=1, rsid="rs123"),),
    )

    result = connector.query(query)

    search_terms = [call["params"]["term"] for call in client.calls if "esearch.fcgi" in str(call["url"])]
    assert search_terms[:2] == ["rs123", "DRD4[gene] AND single_gene[prop]"]
    assert len(result.records) == 1
    assert result.records[0]["variant"] == "rs123"


def test_clinvar_connector_falls_back_from_single_gene_to_gene_query():
    client = RecordingClinVarClient(
        ids_by_term={
            "DRD4[gene] AND single_gene[prop]": [],
            "DRD4[gene]": ["456"],
        },
        summaries={"456": {"uid": "456", "title": "ClinVar broad gene record"}},
    )
    connector = connector_for(get_source_spec("clinvar"), client, ResolvedCredential("clinvar"))

    result = connector.query(KnowledgeQuery(gene="DRD4", region="11:1-10", genome_build="hg19"))

    search_terms = [call["params"]["term"] for call in client.calls if "esearch.fcgi" in str(call["url"])]
    assert search_terms[:2] == ["DRD4[gene] AND single_gene[prop]", "DRD4[gene]"]
    assert result.status == "ok"
    assert result.records[0]["source_id"] == "456"


def test_clinvar_connector_returns_ok_for_zero_results():
    client = RecordingClinVarClient()
    connector = connector_for(get_source_spec("clinvar"), client, ResolvedCredential("clinvar"))

    result = connector.query(KnowledgeQuery(gene="DRD4", region="11:1-10", genome_build="hg19"))

    assert result.status == "ok"
    assert result.records == []
    assert "0 clinical variant record" in result.message


def test_clinvar_connector_reports_tls_error_without_traceback():
    class FailingClient:
        def get_json(self, url, *, params=None, headers=None, rate_limit_per_second=None):
            raise KnowledgeRequestError(
                "GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi failed: "
                "TLS certificate verification failed. Configure NOPHIGENE_CA_BUNDLE.",
                code="tls_certificate_verification_failed",
                remediation="Configure NOPHIGENE_CA_BUNDLE.",
            )

    connector = connector_for(get_source_spec("clinvar"), FailingClient(), ResolvedCredential("clinvar"))

    result = connector.query(KnowledgeQuery(gene="DRD4", region="11:1-10", genome_build="hg19"))
    status = result.to_status(get_source_spec("clinvar"))

    assert result.status == "failed"
    assert status["error_code"] == "tls_certificate_verification_failed"
    assert status["remediation"] == "Configure NOPHIGENE_CA_BUNDLE."
    assert "Traceback" not in json.dumps(status)


def test_clingen_connector_returns_gene_centered_curations():
    client = RecordingClinGenClient()
    connector = connector_for(get_source_spec("clingen"), client, ResolvedCredential("clingen"))

    result = connector.query(KnowledgeQuery(gene="GENE1", region="1:1-10", genome_build="hg19"))

    assert result.status == "ok"
    assert "5 gene-centered curation record" in result.message
    categories = {record["category"] for record in result.records}
    assert categories == {
        "gene_disease_validity",
        "dosage_sensitivity",
        "clinical_gene_curation_summary",
        "clinical_actionability",
    }
    validity = next(record for record in result.records if record["category"] == "gene_disease_validity")
    assert validity["classification"] == "Definitive"
    assert validity["disease"] == "Example syndrome"
    assert validity["mondo_id"] == "MONDO:0000001"
    dosage = next(record for record in result.records if record["category"] == "dosage_sensitivity")
    assert dosage["haploinsufficiency"] == "Sufficient Evidence for Haploinsufficiency"
    actionability_records = [record for record in result.records if record["category"] == "clinical_actionability"]
    assert len(actionability_records) == 2
    actionability = actionability_records[0]
    assert actionability["actionability_score"] == "10CN"
    assert actionability["intervention"] == "Surveillance"
    assert all(record["source"] == get_source_spec("clingen").name for record in result.records)


def test_clingen_connector_reports_empty_live_query_without_metadata_record():
    client = RecordingClinGenClient(empty=True)
    connector = connector_for(get_source_spec("clingen"), client, ResolvedCredential("clingen"))

    result = connector.query(KnowledgeQuery(gene="DRD4", region="11:1-10", genome_build="hg19"))

    assert result.status == "ok"
    assert result.records == []
    assert "no ClinGen curation records found for DRD4" in result.message


def test_clingen_connector_keeps_partial_results_when_one_feed_is_malformed():
    client = RecordingClinGenClient(malformed_validity=True)
    connector = connector_for(get_source_spec("clingen"), client, ResolvedCredential("clingen"))

    result = connector.query(KnowledgeQuery(gene="GENE1", region="1:1-10", genome_build="hg19"))

    assert result.status == "ok"
    assert any("Gene-Disease Validity response could not be parsed" in warning for warning in result.warnings)
    assert {record["category"] for record in result.records} == {
        "dosage_sensitivity",
        "clinical_gene_curation_summary",
        "clinical_actionability",
    }


def test_clingen_connector_reports_request_failure_without_traceback():
    client = RecordingClinGenClient(fail_all=True)
    connector = connector_for(get_source_spec("clingen"), client, ResolvedCredential("clingen"))

    result = connector.query(KnowledgeQuery(gene="GENE1", region="1:1-10", genome_build="hg19"))
    status = result.to_status(get_source_spec("clingen"))

    assert result.status == "failed"
    assert status["error_code"] == "tls_certificate_verification_failed"
    assert status["remediation"] == "Configure NOPHIGENE_CA_BUNDLE."
    assert "Traceback" not in json.dumps(status)


def test_local_article_extractor_finds_gene_snippets_without_full_text(tmp_path: Path):
    article_dir = tmp_path / "articles"
    article_dir.mkdir()
    _write_simple_pdf(
        article_dir / "gene1_results.pdf",
        (
            "Abstract. GENE1 rs123 showed a significant association with Example syndrome in the "
            "reported cohort. DO_NOT_SERIALIZE_PRIVATE_APPENDIX"
        ),
    )

    extraction = extract_local_article_evidence(
        gene="GENE1",
        pdf_folder=article_dir,
        generated_at="2026-06-17T00:00:00Z",
    )

    assert extraction["status"] == "ok"
    assert extraction["provenance"]["pdf_count"] == 1
    assert extraction["provenance"]["record_count"] == 1
    record = extraction["records"][0]
    assert record["source_key"] == LOCAL_ARTICLE_SOURCE_KEY
    assert record["gene"] == "GENE1"
    assert record["rsid"] == "rs123"
    assert record["claim_type"] in {"clinical_variant", "population_association"}
    serialized = json.dumps(extraction, sort_keys=True)
    assert "DO_NOT_SERIALIZE_PRIVATE_APPENDIX" not in serialized
    assert str(article_dir) not in serialized
    assert record["source_file_sha256"]
    assert record["pdf_path_hash"]


def test_dynamic_builder_adds_local_article_evidence_workflow(tmp_path: Path):
    article_dir = tmp_path / "articles"
    article_dir.mkdir()
    _write_simple_pdf(
        article_dir / "gene1_function.pdf",
        "Results. GENE1 expression was reduced after knockdown and the assay showed altered pathway activity.",
    )
    variants = pd.DataFrame([{"chrom": "1", "id": "rs123", "pos": 101, "ref": "A", "alt": "G"}])

    payload = build_dynamic_knowledge_base(
        gene="GENE1",
        region="1:90-110",
        genome_build="hg19",
        variants=variants,
        selected_sources=["clinvar"],
        use_local_article_evidence=True,
        article_pdf_folder=article_dir,
        output_dir=tmp_path,
        request_client=FakeClient(),
        generated_at="2026-06-17T00:00:00Z",
    )

    statuses = {status["source_key"]: status["status"] for status in payload["provider_statuses"]}
    assert statuses[LOCAL_ARTICLE_SOURCE_KEY] == "ok"
    assert payload["local_article_evidence"]["provenance"]["record_count"] == 1
    assert payload["local_article_evidence_artifacts"]["article_evidence_json"].endswith("article_evidence.json")
    assert any(record["source_key"] == LOCAL_ARTICLE_SOURCE_KEY for record in payload["literature_records"])
    assert payload["workflow_runs"][-1]["workflow_key"] == LOCAL_ARTICLE_WORKFLOW_KEY
    assert payload["workflow_runs"][-1]["status"] == "ok"
    assert payload["workflow_source_matrix"][LOCAL_ARTICLE_SOURCE_KEY] == [LOCAL_ARTICLE_WORKFLOW_KEY]
    assert (tmp_path / "local_article_evidence" / "article_evidence_summary.csv").is_file()


def test_dynamic_builder_reports_missing_local_article_folder_as_input_needed():
    payload = build_dynamic_knowledge_base(
        gene="GENE1",
        region="1:90-110",
        genome_build="hg19",
        selected_sources=[],
        selected_workflows=[LOCAL_ARTICLE_WORKFLOW_KEY],
        use_local_article_evidence=True,
        generated_at="2026-06-17T00:00:00Z",
    )

    statuses = {status["source_key"]: status["status"] for status in payload["provider_statuses"]}
    assert statuses[LOCAL_ARTICLE_SOURCE_KEY] == "needs_folder"
    assert payload["workflow_runs"][0]["workflow_key"] == LOCAL_ARTICLE_WORKFLOW_KEY
    assert payload["workflow_runs"][0]["status"] == "needs_input"
    assert payload["local_article_evidence"]["provenance"]["record_count"] == 0


def test_credential_status_redacts_session_secret():
    spec = get_source_spec("omim")
    statuses = credential_status_for_specs([spec], {"omim": "super-secret-token"})

    assert statuses["omim"] == "session:[redacted]"
    assert "super-secret-token" not in json.dumps(statuses)


def test_dynamic_builder_writes_partial_results_without_serializing_secret(tmp_path: Path):
    variants = pd.DataFrame(
        [
            {
                "sample": "sample-one",
                "chrom": "1",
                "id": "rs123",
                "pos": 101,
                "ref": "A",
                "alt": "G",
                "gt_raw": "0/1",
                "zygosity": "heterozygous",
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "IlmnID": "cg00000001",
                "CHR": "1",
                "MAPINFO": 99,
                "UCSC_RefGene_Name": "GENE1",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    payload = build_dynamic_knowledge_base(
        gene="GENE1",
        region="1:90-110",
        genome_build="hg19",
        variants=variants,
        manifest_subset=manifest,
        selected_sources=["clinvar", "omim", "hgmd"],
        credentials={"omim": "super-secret-token"},
        output_dir=tmp_path,
        request_client=FakeClient(),
        generated_at="2026-06-17T00:00:00Z",
    )

    artifact = tmp_path / "variant_kb.json"
    assert artifact.is_file()
    assert payload["variant_records"][0]["variant"] == "rs123"
    assert payload["epigenetic_locus_records"][0]["probe_id"] == "cg00000001"
    assert {status["source_key"]: status["status"] for status in payload["provider_statuses"]} == {
        "clinvar": "ok",
        "omim": "metadata_only",
        "hgmd": "needs_export",
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "super-secret-token" not in serialized
    assert "session:[redacted]" in serialized
    assert payload["workflow_runs"][0]["workflow_key"] == "clinical_variant_triage"
    assert payload["workflow_runs"][0]["status"] == "partial"
    assert payload["workflow_source_matrix"]["clinvar"] == ["clinical_variant_triage"]


def test_dynamic_builder_runs_workflows_sequentially_and_deduplicates_sources():
    variants = pd.DataFrame([{"chrom": "1", "id": "rs123", "pos": 101, "ref": "A", "alt": "G"}])
    client = CountingFakeClient()

    payload = build_dynamic_knowledge_base(
        gene="GENE1",
        region="1:90-110",
        genome_build="hg19",
        variants=variants,
        selected_workflows=["clinical_variant_triage", "population_frequency_association"],
        selected_sources=["dbsnp"],
        request_client=client,
        generated_at="2026-06-17T00:00:00Z",
    )

    assert len(payload["provider_statuses"]) == 1
    assert payload["provider_statuses"][0]["source_key"] == "dbsnp"
    assert client.get_counts["rs123"] == 1
    assert [run["workflow_key"] for run in payload["workflow_runs"]] == [
        "clinical_variant_triage",
        "population_frequency_association",
    ]
    assert all(run["selected_source_keys"] == ["dbsnp"] for run in payload["workflow_runs"])
    assert payload["workflow_source_matrix"]["dbsnp"] == [
        "clinical_variant_triage",
        "population_frequency_association",
    ]
    assert payload["source_records"][0]["evidence_id"] == "dbsnp:001"


def test_dynamic_builder_merges_clingen_curations_into_workflow_records():
    payload = build_dynamic_knowledge_base(
        gene="GENE1",
        region="1:1-10",
        genome_build="hg19",
        selected_sources=["clingen"],
        request_client=RecordingClinGenClient(),
        generated_at="2026-06-17T00:00:00Z",
    )

    assert payload["provider_statuses"][0]["source_key"] == "clingen"
    assert payload["provider_statuses"][0]["status"] == "ok"
    assert payload["workflow_source_matrix"]["clingen"] == ["clinical_variant_triage"]
    assert {record["category"] for record in payload["source_records"]} == {
        "gene_disease_validity",
        "dosage_sensitivity",
        "clinical_gene_curation_summary",
        "clinical_actionability",
    }
    assert any(record["classification"] == "Definitive" for record in payload["source_records"])
    assert payload["workflow_runs"][0]["record_counts"]["source_records"] == 5


def test_source_import_parser_normalizes_json_csv_and_discards_raw_columns(tmp_path: Path):
    csv_path = tmp_path / "hgmd.csv"
    csv_path.write_text(
        "gene,rsid,mutation,classification,disease,pmid,private_payload\n"
        "GENE1,rs123,c.1A>G,Pathogenic,Example syndrome,PMID:1,do not serialize\n"
        "GENE1,rs123,c.1A>G,Pathogenic,Example syndrome,PMID:1,duplicate raw\n",
        encoding="utf-8",
    )
    bundle = parse_source_import(
        "hgmd",
        csv_path,
        spec=get_source_spec("hgmd"),
        imported_at="2026-06-17T00:00:00Z",
    )

    assert bundle.row_count == 2
    assert bundle.normalized_record_count == 1
    record = bundle.records[0]
    assert record["source_key"] == "hgmd"
    assert record["clinical_significance"] == "Pathogenic"
    assert record["category"] == "clinical_variant"
    serialized = json.dumps(bundle.to_provenance()) + json.dumps(bundle.records)
    assert "do not serialize" not in serialized
    assert "duplicate raw" not in serialized

    json_path = tmp_path / "mastermind.json"
    json_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "gene": "GENE1",
                        "variant": "rs123",
                        "article_title": "GENE1 rs123 evidence",
                        "snippet": "Literature summary",
                        "pmid": "PMID:2",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    literature_bundle = parse_source_import(
        "mastermind",
        json_path,
        spec=get_source_spec("mastermind"),
        imported_at="2026-06-17T00:00:00Z",
    )
    assert literature_bundle.records[0]["category"] == "literature"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"not_records": {"x": 1}}), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_source_import("hgmd", bad_json, spec=get_source_spec("hgmd"))
    with pytest.raises(ValueError):
        parse_source_import("", csv_path, spec=get_source_spec("hgmd"))


def test_dynamic_builder_uses_user_export_for_licensed_source(tmp_path: Path):
    import_path = tmp_path / "hgmd.csv"
    import_path.write_text(
        "gene,rsid,mutation,classification,disease,pmid,raw_payload\n"
        "GENE1,rs123,c.1A>G,Pathogenic,Example syndrome,PMID:1,raw licensed text\n",
        encoding="utf-8",
    )
    variants = pd.DataFrame([{"chrom": "1", "id": "rs123", "pos": 101, "ref": "A", "alt": "G"}])

    payload = build_dynamic_knowledge_base(
        gene="GENE1",
        region="1:90-110",
        genome_build="hg19",
        variants=variants,
        selected_sources=["hgmd"],
        source_imports={"hgmd": import_path},
        request_client=FakeClient(),
        generated_at="2026-06-17T00:00:00Z",
    )

    assert payload["provider_statuses"][0]["status"] == "imported"
    assert payload["provenance"]["source_imports"][0]["row_count"] == 1
    assert payload["source_records"][0]["clinical_significance"] == "Pathogenic"
    assert payload["variant_records"][0]["evidence"][0]["source_key"] == "hgmd"
    serialized = json.dumps(payload, sort_keys=True)
    assert "raw licensed text" not in serialized


def test_dynamic_builder_is_deterministic_with_fixed_timestamp():
    variants = pd.DataFrame([{"chrom": "2", "id": "rs999", "pos": 200, "ref": "C", "alt": "T"}])

    first = build_dynamic_knowledge_base(
        gene="GENE2",
        region="2:190-210",
        genome_build="hg19",
        variants=variants,
        selected_sources=["clinvar"],
        request_client=FakeClient(),
        generated_at="2026-06-17T00:00:00Z",
    )
    second = build_dynamic_knowledge_base(
        gene="GENE2",
        region="2:190-210",
        genome_build="hg19",
        variants=variants,
        selected_sources=["clinvar"],
        request_client=FakeClient(),
        generated_at="2026-06-17T00:00:00Z",
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_dynamic_merger_creates_minimal_bundle_for_uncurated_gene():
    dynamic_payload = {
        "database_name": "Dynamic GENE3 KB",
        "generated_at": "2026-06-17T00:00:00Z",
        "genome_build": "hg19",
        "provider_statuses": [{"source_key": "clinvar", "status": "ok"}],
        "variant_records": [
            {
                "variant": "rs77",
                "display_name": "rs77",
                "lookup_keys": ["rs77", "3:77"],
            }
        ],
        "epigenetic_locus_records": [{"probe_id": "cg77"}],
    }

    merged = merge_dynamic_knowledge_base(
        None,
        dynamic_payload,
        gene_name="GENE3",
        region="3:1-100",
    )

    assert merged["gene_context"]["gene_name"] == "GENE3"
    assert merged["variant_records"][0]["variant"] == "rs77"
    assert merged["gene_context"]["dynamic_knowledge_base"]["provider_count"] == 1
