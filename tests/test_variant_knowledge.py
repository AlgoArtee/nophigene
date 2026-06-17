import json
from pathlib import Path

import pandas as pd
import pytest

from src.variant_knowledge.credentials import credential_status_for_specs
from src.variant_knowledge.imports import parse_source_import
from src.variant_knowledge.merger import merge_dynamic_knowledge_base
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
    workflows = list_workflow_specs()

    assert [workflow.key for workflow in workflows if workflow.default_selected] == list(CORE_SAFETY_WORKFLOW_KEYS)
    assert workflows
    for workflow in workflows:
        assert workflow.label
        assert workflow.purpose
        assert workflow.report_section
        assert set(workflow.ordered_source_keys) <= source_keys
        if workflow.key == "licensed_aggregator_review":
            joined_notes = " ".join(workflow.licensed_notes).lower()
            assert "scraping" in joined_notes
            assert "captcha" in joined_notes


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
