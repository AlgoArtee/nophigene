"""Regression tests for schema-v2 evidence-calibrated interpretation."""

from __future__ import annotations

import pandas as pd

from src.interpretation import (
    build_evidence_snapshot,
    build_interpretation_payload,
    build_model_assessments,
    canonical_variant_alleles,
    source_record_matches_allele,
)
from src.variant_knowledge.models import KnowledgeQuery, QueryVariant, SourceSpec
from src.variant_knowledge.orchestrator import _build_dynamic_variant_records


def _variant_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample": "sample-one",
                "chrom": "1",
                "pos": 100,
                "id": "rs100",
                "ref": "A",
                "alt": "G,T",
                "gt_raw": "1/2",
                "genotype": "G/T",
                "zygosity": "compound_heterozygous",
                "filter_pass": True,
                "filter_status": "PASS",
                "qual": 80,
                "dp": 35,
                "gq": 60,
                "confidence_score": 0.95,
                "qc_flags": [],
            }
        ]
    )


def test_multiallelic_calls_are_split_before_evidence_matching() -> None:
    alleles = canonical_variant_alleles(_variant_table(), genome_build="hg38")

    assert [(allele["alt"], allele["alt_dosage"]) for allele in alleles] == [("G", 1), ("T", 1)]
    assert alleles[0]["allele_key"] == "hg38|1|100|A|G"
    assert alleles[1]["allele_key"] == "hg38|1|100|A|T"
    assert source_record_matches_allele(
        {"chromosome": "1", "position": 100, "ref": "A", "alt": "G"}, alleles[0]
    )
    assert not source_record_matches_allele(
        {"chromosome": "1", "position": 100, "ref": "A", "alt": "G"}, alleles[1]
    )
    assert not source_record_matches_allele(
        {"chromosome": "1", "position": 100, "ref": "A", "alt": "G", "genome_build": "hg19"},
        alleles[0],
    )


def test_indels_require_reference_backed_normalization_before_external_matching() -> None:
    alleles = canonical_variant_alleles(
        pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "AT", "alt": "A", "gt_raw": "0/1"}]),
        genome_build="hg38",
    )

    assert alleles[0]["normalization"]["minimal_representation"] == "as_reported"
    assert alleles[0]["normalization"]["left_alignment"] == "not_performed_without_reference"
    assert alleles[0]["normalization"]["identity_status"] == "requires_reference_normalization"


def test_unknown_build_and_phase_do_not_create_a_clinical_haplotype_claim() -> None:
    table = _variant_table().assign(alt="G", gt_raw="0|1")
    alleles = canonical_variant_alleles(table, genome_build="unresolved")
    assert alleles[0]["genome_build"] == "unknown"
    assert alleles[0]["phase_status"] == "phased_call_present"
    assert alleles[0]["haplotype_status"] == "not_inferred_from_small_variant_vcf"

    interpretation = build_interpretation_payload(
        variants=table,
        methylation=pd.DataFrame(),
        gene_name="DRD4",
        genome_build="unresolved",
        interpretation_mode="clinical_support",
        knowledge_base={"variant_records": []},
        dynamic_payload=None,
    )
    finding = interpretation["findings"][0]
    assert "unknown_genome_build" in finding["clinical_support"]["eligibility_blockers"]
    assert interpretation["drd4_repeat_assay"]["status"] == "not_assessed"
    assert interpretation["drd4_repeat_assay"]["haplotype_status"] == "not_inferred_from_nearby_snps"


def test_unmatched_provider_context_is_not_promoted_to_variant_evidence() -> None:
    dynamic_payload = {
        "gene_name": "GENE1",
        "region": "1:90-110",
        "genome_build": "hg38",
        "generated_at": "2026-07-09T00:00:00Z",
        "source_records": [
            {
                "source_key": "clingen",
                "source": "ClinGen",
                "category": "gene_disease_validity",
                "gene": "GENE1",
                "label": "GENE1 gene validity",
            }
        ],
        "provider_statuses": [{"source_key": "clingen", "name": "ClinGen", "status": "ok"}],
    }

    interpretation = build_interpretation_payload(
        variants=_variant_table(),
        methylation=pd.DataFrame(),
        gene_name="GENE1",
        genome_build="hg38",
        interpretation_mode="research",
        knowledge_base={
            "variant_records": [
                {
                    "region_class": "dynamic_query_variant",
                    "rsid": "rs100",
                    "lookup_keys": ["rs100", "1:100:A>G"],
                    "evidence": [{"label": "Unmatched dynamic provider context"}],
                }
            ]
        },
        dynamic_payload=dynamic_payload,
    )

    assert len(interpretation["findings"]) == 2
    assert all(finding["evidence"] == [] for finding in interpretation["findings"])
    assert all(finding["evidence_strength"]["status"] == "not_assessed" for finding in interpretation["findings"])
    assert interpretation["mode_reports"]["clinical_support"]["eligible_finding_keys"] == []


def test_dynamic_builder_does_not_use_unmatched_records_as_variant_fallback() -> None:
    query = KnowledgeQuery(
        gene="GENE1",
        region="1:90-110",
        genome_build="hg38",
        variants=(QueryVariant(chrom="1", pos=100, ref="A", alt="G", rsid="rs100"),),
    )
    spec = SourceSpec(
        key="clingen",
        name="ClinGen",
        description="Gene-level curation",
        lane="clinical",
        access_type="open_api",
        connector_kind="clingen",
    )
    records = _build_dynamic_variant_records(
        query,
        [{"source_key": "clingen", "category": "gene_disease_validity", "gene": "GENE1", "label": "GENE1"}],
        (spec,),
    )

    assert records[0]["evidence"] == []
    assert records[0]["dynamic_source_count"] == 0


def test_clinical_support_downgrades_when_reference_and_transcript_are_missing() -> None:
    dynamic_payload = {
        "gene_name": "GENE1",
        "region": "1:90-110",
        "genome_build": "hg38",
        "generated_at": "2026-07-09T00:00:00Z",
        "source_records": [
            {
                "source_key": "clinvar",
                "source": "ClinVar",
                "category": "clinical_variant",
                "rsid": "rs100",
                "clinical_significance": "Pathogenic",
                "review_status": "reviewed by expert panel",
                "assertion_criteria": "https://example.test/criteria",
            }
        ],
        "provider_statuses": [{"source_key": "clinvar", "name": "ClinVar", "status": "ok"}],
    }
    interpretation = build_interpretation_payload(
        variants=_variant_table().iloc[:1, :].assign(alt="G", gt_raw="0/1"),
        methylation=pd.DataFrame(),
        gene_name="GENE1",
        genome_build="hg38",
        interpretation_mode="clinical_support",
        knowledge_base={"variant_records": []},
        dynamic_payload=dynamic_payload,
        sample_context={"phenotype_terms": ["Example phenotype"]},
    )

    finding = interpretation["findings"][0]
    assert finding["evidence_strength"]["clinical_assertion_available"] is True
    assert finding["clinical_support"]["status"] == "downgraded_to_research"
    assert "reference_allele_not_validated" in finding["clinical_support"]["eligibility_blockers"]
    assert "mane_transcript_not_annotated" in finding["clinical_support"]["eligibility_blockers"]


def test_methylation_is_descriptive_without_qc_and_reference_context() -> None:
    interpretation = build_interpretation_payload(
        variants=pd.DataFrame(),
        methylation=pd.DataFrame([{"probe_id": "cg1", "beta": 0.8, "chrom": "1", "pos": 99}]),
        gene_name="GENE1",
        genome_build="hg38",
        interpretation_mode="research",
        knowledge_base={"variant_records": []},
        dynamic_payload=None,
    )

    methylation = interpretation["methylation_assessment"]
    assert methylation["status"] == "research_context_only"
    assert methylation["raw_mean_beta_is_interpreted"] is False
    assert methylation["probe_findings"][0]["interpretation"] == "descriptive_only"


def test_methylation_masking_flags_failed_probe_qc_without_interpreting_the_mean() -> None:
    interpretation = build_interpretation_payload(
        variants=pd.DataFrame(),
        methylation=pd.DataFrame(
            [
                {
                    "probe_id": "cg1",
                    "beta": 0.8,
                    "detection_p_value": 0.02,
                    "bead_count": 2,
                    "cross_reactive": True,
                }
            ]
        ),
        gene_name="GENE1",
        genome_build="hg38",
        interpretation_mode="research",
        knowledge_base={"variant_records": []},
        dynamic_payload=None,
    )

    probe = interpretation["methylation_assessment"]["probe_findings"][0]
    assert probe["included_for_reference_comparison"] is False
    assert set(probe["mask_reasons"]) == {
        "detection_p_above_0.01",
        "bead_count_below_3",
        "cross_reactive_probe",
    }


def test_methylation_variant_link_requires_exact_same_context_and_flags_ld() -> None:
    interpretation = build_interpretation_payload(
        variants=_variant_table().assign(alt="G", gt_raw="0/1"),
        methylation=pd.DataFrame(
            [
                {
                    "probe_id": "cg1",
                    "beta": 0.6,
                    "detection_p_value": 0.001,
                    "bead_count": 5,
                    "normalization_method": "noob",
                    "probe_variant_rsid": "rs100",
                    "same_tissue_cohort": True,
                    "ld_confounded": True,
                }
            ]
        ),
        gene_name="GENE1",
        genome_build="hg38",
        interpretation_mode="research",
        knowledge_base={"variant_records": []},
        dynamic_payload=None,
        sample_context={
            "tissue": "blood",
            "batch_id": "batch-1",
            "cell_composition_method": "estimated",
            "methylation_reference_cohort_id": "cohort-1",
        },
    )

    relationship = interpretation["methylation_assessment"]["probe_findings"][0]["variant_relationship"]
    assert relationship["matched_allele"] == "rs100"
    assert relationship["status"] == "exact_relationship_with_confounders"
    assert relationship["confounding_alternatives"] == ["linkage_disequilibrium"]


def test_model_assessment_requires_harmonization_coverage_and_matching_ancestry() -> None:
    assessments = build_model_assessments(
        sample_context={"ancestry": "European"},
        requested_models=[
            {
                "model_id": "PGS000001",
                "model_version": "1.0",
                "source": "PGS Catalog",
                "evaluation_ancestry": "European",
                "performance_metric": "AUROC 0.71",
                "calibration": "external calibration v1",
                "baseline_risk": "0.08",
                "weighted_call_coverage": 0.97,
                "alleles_harmonized": True,
            },
            {
                "model_id": "PGS000002",
                "model_version": "1.0",
                "source": "PGS Catalog",
                "evaluation_ancestry": "East Asian",
                "performance_metric": "AUROC 0.71",
                "calibration": "external calibration v1",
                "baseline_risk": "0.08",
                "weighted_call_coverage": 0.90,
                "alleles_harmonized": False,
            },
        ],
    )

    assert assessments[0]["status"] == "eligible"
    assert assessments[0]["probability_emitted"] is False
    assert assessments[1]["status"] == "model_not_eligible"
    assert "ancestry_evaluation_mismatch" in assessments[1]["eligibility_blockers"]
    assert "insufficient_weighted_call_coverage" in assessments[1]["eligibility_blockers"]


def test_missing_registered_model_returns_explicit_not_eligible_state() -> None:
    assessment = build_model_assessments(sample_context={"ancestry": "European"})[0]
    assert assessment["status"] == "model_not_eligible"
    assert assessment["eligibility_blockers"] == ["no_registered_model_requested"]


def test_evidence_snapshot_is_deterministic_and_preserves_not_assessed_status() -> None:
    dynamic_payload = {
        "gene_name": "GENE1",
        "region": "1:1-10",
        "genome_build": "hg38",
        "generated_at": "2026-07-09T00:00:00Z",
        "provider_statuses": [
            {"source_key": "clinvar", "name": "ClinVar", "status": "ok", "record_count": 1},
            {"source_key": "omim", "name": "OMIM", "status": "needs_credentials", "record_count": 0},
        ],
    }

    first = build_evidence_snapshot(dynamic_payload)
    second = build_evidence_snapshot(dynamic_payload)
    changed = {
        **dynamic_payload,
        "source_records": [{"source_key": "clinvar", "source_id": "VCV1", "summary": "normalized evidence"}],
    }

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["snapshot_id"] != build_evidence_snapshot(changed)["snapshot_id"]
    assert first["providers"][0]["assessment_status"] == "assessed"
    assert first["providers"][1]["assessment_status"] == "not_assessed"
