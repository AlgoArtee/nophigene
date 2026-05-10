"""Regression coverage for the newly bundled curated gene knowledge bases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis import (
    build_methylation_insights,
    build_population_insights,
    build_predictive_theses,
    build_variant_interpretations,
    load_gene_interpretation_database,
    load_gene_population_database,
    load_gene_synthesis_database,
)


CURATED_GENES = {
    "FOXO3": 108881028,
    "GLP1R": 39016574,
    "MTOR": 11166592,
    "RPS6": 19375713,
    "SIK3": 116714118,
    "FLCN": 17115526,
    "HERC2": 28356186,
    "SIRT6": 4174106,
    "PRKAA1": 40759491,
    "NAMPT": 105888744,
    "CDKN2A": 21967751,
    "TERT": 1253282,
    "CLRN2": 17516788,
    "ARHGAP10": 148653239,
    "FAM170A": 118965253,
    "SYCE3": 50989541,
    "BLTP3B": 100430850,
    "CIROP": 23568271,
}


@pytest.mark.parametrize("gene_name,gene_start", CURATED_GENES.items())
def test_curated_gene_bundle_loads_with_manifest_subset(gene_name: str, gene_start: int) -> None:
    """Each requested curated gene should ship interpretation, population, synthesis, and probe data."""
    knowledge_base = load_gene_interpretation_database(gene_name)
    population_database = load_gene_population_database(gene_name)
    synthesis_database = load_gene_synthesis_database(gene_name)

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["gene_name"] == gene_name
    assert knowledge_base["gene_context"]["gene_region"]["start"] == gene_start
    assert knowledge_base["gene_context"]["variant_effect_overview"]
    assert knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert population_database["database_name"].startswith(f"NophiGene {gene_name} Population")
    assert population_database["gene_population_patterns"]
    assert synthesis_database["database_name"].startswith(f"NophiGene {gene_name} Predictive")
    assert synthesis_database["case_count"] == 10
    assert synthesis_database["concrete_variant_prediction"]
    if knowledge_base["variant_records"]:
        assert synthesis_database["variant_prediction_rules"]

    subset_path = Path("src/gene_data") / f"{gene_name}_epigenetics_hg19.csv"
    assert subset_path.exists()
    manifest_subset = pd.read_csv(subset_path)
    assert not manifest_subset.empty


def test_foxo3_curated_bundle_drives_interpretation_and_population_helpers() -> None:
    """FOXO3 should behave like a curated gene even with pattern-only population notes."""
    knowledge_base = load_gene_interpretation_database("FOXO3")
    population_database = load_gene_population_database("FOXO3")

    assert knowledge_base is not None
    assert population_database is not None

    variants = pd.DataFrame(
        [
            {
                "chrom": "6",
                "id": "rs2802292",
                "pos": 108900000,
                "ref": "T",
                "alt": "G",
                "qual": 61.2,
                "filter_pass": True,
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="6:108881028-109005977",
    )
    population_insights = build_population_insights(variants, knowledge_base, population_database)

    assert interpretation["gene_name"] == "FOXO3"
    assert "FOXO3 is located at" in interpretation["summary"]
    assert interpretation["matched_records"][0]["variant"] == "rs2802292"

    assert population_insights["variant_population_records"] == []
    assert population_insights["gene_population_patterns"]
    assert "No embedded allele-frequency panel is bundled for FOXO3 yet" in population_insights["summary"]
    assert population_insights["gene_population_patterns_intro"].startswith(
        "Broader population patterns curated from FOXO3"
    )


def test_glp1r_curated_bundle_drives_prediction_helpers() -> None:
    """GLP1R should provide pharmacogenetic interpretation and synthesis rules."""
    knowledge_base = load_gene_interpretation_database("GLP1R")
    population_database = load_gene_population_database("GLP1R")
    synthesis_database = load_gene_synthesis_database("GLP1R")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "6:39015574-39055519"
    assert any(record["variant"] == "rs6923761" for record in knowledge_base["variant_records"])
    assert synthesis_database["case_count"] == 10
    assert "incretin-receptor response thesis" in synthesis_database["concrete_variant_prediction"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "6",
                "id": ".",
                "pos": 39034072,
                "ref": "G",
                "alt": "A",
                "gt_raw": "0/1",
                "ad": [8, 7],
                "dp": 15,
                "gq": 43,
                "qual": 77.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg15041550",
                "beta": 0.74,
                "chrom": "6",
                "pos": 39016590,
                "GencodeBasicV12_NAME": "GLP1R",
                "UCSC_RefGene_Group": "TSS1500",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="6:39015574-39055519",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "rs6923761 (Gly168Ser)"
    assert methylation_insights["whitelist_mean_beta"] == 0.74
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "reduced gliptin-response thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_cdkn2a_curated_bundle_distinguishes_common_and_pathogenic_markers() -> None:
    """CDKN2A should separate benign/common markers from exact pathogenic G101W matching."""
    knowledge_base = load_gene_interpretation_database("CDKN2A")
    population_database = load_gene_population_database("CDKN2A")
    synthesis_database = load_gene_synthesis_database("CDKN2A")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None

    variant_records = knowledge_base["variant_records"]
    variant_ids = {record["variant"] for record in variant_records}
    assert {"rs11515", "rs3088440", "rs3731249", "CDKN2A p.Gly101Trp"} <= variant_ids

    g101w_record = next(record for record in variant_records if record["variant"] == "CDKN2A p.Gly101Trp")
    assert g101w_record["position"] == 21971057
    assert "9:21971057:C>A" in g101w_record["lookup_keys"]
    assert "rs104894094" not in g101w_record["lookup_keys"]
    assert "Pathogenic ClinVar germline variant" in g101w_record["clinical_significance"]

    a148t_record = next(record for record in variant_records if record["variant"] == "rs3731249")
    assert "Benign germline ClinVar" in a148t_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "9",
                "id": ".",
                "pos": 21971057,
                "ref": "C",
                "alt": "A",
                "gt_raw": "0/1",
                "ad": [11, 9],
                "dp": 20,
                "gq": 55,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": knowledge_base["gene_context"]["relevant_methylation_probe_ids"][0],
                "beta": 0.81,
                "chrom": "9",
                "pos": 21994765,
                "GencodeBasicV12_NAME": "CDKN2A",
                "UCSC_RefGene_Group": "TSS200",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="9:21967751-21996323",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "CDKN2A p.Gly101Trp (c.301G>T / G101W)"
    assert "Pathogenic ClinVar germline variant" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.81
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "rare pathogenic G101W signal" in row["prediction"]
        and row["source"] == "GT-confirmed allele-dosage thesis"
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_cirop_bundle_covers_heterotaxy_markers_and_tss_probes() -> None:
    """CIROP should load as a developmental heterotaxy bundle with curated TSS-proximal probes."""
    knowledge_base = load_gene_interpretation_database("CIROP")
    population_database = load_gene_population_database("CIROP")
    synthesis_database = load_gene_synthesis_database("CIROP")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "14"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 23568271
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 23574198
    assert knowledge_base["gene_context"]["relevant_methylation_probe_ids"] == [
        "cg19577365",
        "cg11790074",
    ]
    assert "heterotaxy thesis" in synthesis_database["concrete_variant_prediction"]

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert {
        "CIROP c.92C>T",
        "CIROP c.571C>T",
        "CIROP c.1037G>A",
        "CIROP c.1151C>T",
        "CIROP c.1166G>T",
        "CIROP c.1364TCT[1]",
    } <= variant_ids

    r191_record = next(record for record in knowledge_base["variant_records"] if record["variant"] == "CIROP c.571C>T")
    assert r191_record["position"] == 23572916
    assert "14:23572916:G>A" in r191_record["lookup_keys"]
    assert "Pathogenic ClinVar germline variant" in r191_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "14",
                "id": ".",
                "pos": 23572916,
                "ref": "G",
                "alt": "A",
                "gt_raw": "0/1",
                "ad": [12, 10],
                "dp": 22,
                "gq": 58,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg19577365",
                "beta": 0.68,
                "chrom": "14",
                "pos": 23574175,
                "GencodeBasicV12_NAME": "",
                "UCSC_RefGene_Group": "",
                "Relation_to_UCSC_CpG_Island": "",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="14:23568271-23575198",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "CIROP c.571C>T / p.Arg191Ter"
    assert "heterotaxy" in interpretation["matched_records"][0]["clinical_significance"].lower()
    assert methylation_insights["whitelist_mean_beta"] == 0.68
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "CIROP loss-of-function thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_clrn2_bundle_covers_dfnb117_marker_and_tss_probes() -> None:
    """CLRN2 should load as a hearing-loss bundle with pathogenic and VUS marker tiers."""
    knowledge_base = load_gene_interpretation_database("CLRN2")
    population_database = load_gene_population_database("CLRN2")
    synthesis_database = load_gene_synthesis_database("CLRN2")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "4"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 17516788
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 17528727
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "4:17515788-17528727"
    assert "DFNB117 hearing-loss thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg02953545",
        "cg06791107",
        "cg00389446",
        "cg09099893",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert {"CLRN2 c.494C>A", "CLRN2 c.236G>T"} <= variant_ids

    pathogenic_record = next(
        record for record in knowledge_base["variant_records"] if record["variant"] == "CLRN2 c.494C>A"
    )
    assert pathogenic_record["position"] == 17528500
    assert "4:17528500:C>A" in pathogenic_record["lookup_keys"]
    assert "Pathogenic" in pathogenic_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "4",
                "id": ".",
                "pos": 17528500,
                "ref": "C",
                "alt": "A",
                "gt_raw": "1/1",
                "ad": [1, 21],
                "dp": 22,
                "gq": 62,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg02953545",
                "beta": 0.66,
                "chrom": "4",
                "pos": 17516819,
                "GencodeBasicV12_NAME": "CLRN2",
                "UCSC_RefGene_Group": "1stExon;5'UTR",
                "Relation_to_UCSC_CpG_Island": "S_Shelf",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="4:17515788-17528727",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "CLRN2 c.494C>A / p.Thr165Lys"
    assert "DFNB117" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.66
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "DFNB117 autosomal recessive nonsyndromic hearing-loss review thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_arhgap10_bundle_covers_schizophrenia_rhogap_and_tss_probes() -> None:
    """ARHGAP10 should load as a RhoGAP research bundle with CNV and S490P marker context."""
    knowledge_base = load_gene_interpretation_database("ARHGAP10")
    population_database = load_gene_population_database("ARHGAP10")
    synthesis_database = load_gene_synthesis_database("ARHGAP10")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "4"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 148653239
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 148993927
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "4:148652239-148993927"
    assert "RhoGAP and neuronal-morphology thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg09777578",
        "cg17876581",
        "cg06802374",
        "cg24243429",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert {"ARHGAP10 exonic CNV", "ARHGAP10 p.Ser490Pro"} <= variant_ids

    s490p_record = next(
        record for record in knowledge_base["variant_records"] if record["variant"] == "ARHGAP10 p.Ser490Pro"
    )
    assert "rs483352828" in s490p_record["lookup_keys"]
    assert "NM_024605.4(ARHGAP10):c.1468T>C" in s490p_record["lookup_keys"]
    assert "Research-level rare missense marker" in s490p_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "4",
                "id": "rs483352828",
                "pos": 148800000,
                "ref": "T",
                "alt": "C",
                "gt_raw": "0/1",
                "ad": [13, 11],
                "dp": 24,
                "gq": 60,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg09777578",
                "beta": 0.72,
                "chrom": "4",
                "pos": 148653439,
                "GencodeBasicV12_NAME": "ARHGAP10",
                "UCSC_RefGene_Group": "5'UTR;1stExon",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="4:148652239-148993927",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "ARHGAP10 c.1468T>C / p.Ser490Pro"
    assert "Research-level rare missense marker" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.72
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "rare ARHGAP10 RhoGAP-domain missense review thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_fam170a_bundle_covers_spermiogenesis_transcription_and_tss_probes() -> None:
    """FAM170A should load as a male-fertility transcription-factor research bundle."""
    knowledge_base = load_gene_interpretation_database("FAM170A")
    population_database = load_gene_population_database("FAM170A")
    synthesis_database = load_gene_synthesis_database("FAM170A")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "5"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 118965253
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 118971517
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "5:118964253-118971517"
    assert "nuclear zinc-finger and spermiogenesis thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg01229506",
        "cg24537512",
        "cg23314948",
        "cg14144850",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "FAM170A loss-of-function/deletion model" in variant_ids

    lof_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "FAM170A loss-of-function/deletion model"
    )
    assert "FAM170A deletion" in lof_record["lookup_keys"]
    assert "Mouse model and expression-supported male-fertility research marker" in lof_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "5",
                "id": "FAM170A deletion",
                "pos": 118965253,
                "ref": "N",
                "alt": "<DEL>",
                "gt_raw": "0/1",
                "ad": [12, 10],
                "dp": 22,
                "gq": 55,
                "qual": 86.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg01229506",
                "beta": 0.69,
                "chrom": "5",
                "pos": 118965233,
                "GencodeBasicV12_NAME": "FAM170A",
                "UCSC_RefGene_Group": "TSS200",
                "Relation_to_UCSC_CpG_Island": "",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="5:118964253-118971517",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "FAM170A loss-of-function or deletion model"
    assert "male-fertility research marker" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.69
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "sperm chromatin-remodeling thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_syce3_bundle_covers_meiotic_synapsis_and_tss_probes() -> None:
    """SYCE3 should load as a synaptonemal-complex meiotic-synapsis research bundle."""
    knowledge_base = load_gene_interpretation_database("SYCE3")
    population_database = load_gene_population_database("SYCE3")
    synthesis_database = load_gene_synthesis_database("SYCE3")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "22"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 50989541
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 51001348
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "22:50989541-51002348"
    assert "synaptonemal-complex central-element thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg12119715",
        "cg05722611",
        "cg00349050",
        "cg25309564",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "SYCE3 loss-of-function/deletion model" in variant_ids

    lof_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "SYCE3 loss-of-function/deletion model"
    )
    assert "SYCE3 deletion" in lof_record["lookup_keys"]
    assert "Mouse model-supported meiotic-arrest" in lof_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "22",
                "id": "SYCE3 deletion",
                "pos": 51001348,
                "ref": "N",
                "alt": "<DEL>",
                "gt_raw": "0/1",
                "ad": [11, 9],
                "dp": 20,
                "gq": 57,
                "qual": 88.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg12119715",
                "beta": 0.71,
                "chrom": "22",
                "pos": 51001351,
                "GencodeBasicV12_NAME": "SYCE3",
                "UCSC_RefGene_Group": "TSS200",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="22:50989541-51002348",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "SYCE3 loss-of-function or deletion model"
    assert "meiotic-arrest" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.71
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "synaptonemal-complex central-element thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_poteb3_bundle_is_grch38_paralog_context_without_epic_probes() -> None:
    """POTEB3 should load as a GRCh38 POTE-family structural-region bundle without EPIC probes."""
    knowledge_base = load_gene_interpretation_database("POTEB3")
    population_database = load_gene_population_database("POTEB3")
    synthesis_database = load_gene_synthesis_database("POTEB3")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["assembly"] == "GRCh38 / hg38"
    assert knowledge_base["gene_context"]["chromosome"] == "15"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 21405401
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 21440499
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "15:21405401-21441499"
    assert knowledge_base["gene_context"]["relevant_methylation_probe_ids"] == []
    assert "POTEB3/POTE-family structural-region thesis" in synthesis_database["concrete_variant_prediction"]

    subset_path = Path("src/gene_data") / "POTEB3_epigenetics_hg19.csv"
    assert subset_path.exists()
    assert pd.read_csv(subset_path).empty

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "15q11.1-q11.2 CNV including POTEB3" in variant_ids

    variants = pd.DataFrame(
        [
            {
                "chrom": "15",
                "id": "nsv533590",
                "pos": 21405401,
                "ref": "N",
                "alt": "<DEL>",
                "gt_raw": "0/1",
                "ad": [10, 8],
                "dp": 18,
                "gq": 44,
                "qual": 80.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(columns=["probe_id", "beta", "UCSC_RefGene_Group"])

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="15:21405401-21441499",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "15q11.1-q11.2 copy-number region including POTEB3"
    assert "Benign ClinVar regional CNV" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] is None
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "regional structural-variation thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_bltp3b_bundle_covers_myp3_marker_and_legacy_tss_probes() -> None:
    """BLTP3B should load with UHRF1BP1L-era myopia evidence and reverse-strand TSS probes."""
    knowledge_base = load_gene_interpretation_database("BLTP3B")
    population_database = load_gene_population_database("BLTP3B")
    synthesis_database = load_gene_synthesis_database("BLTP3B")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "12"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 100430850
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 100536652
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "12:100430850-100537652"
    assert "MYP3-locus thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg02392575",
        "cg16466203",
        "cg22215783",
        "cg10568931",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "rs7134216" in variant_ids

    variants = pd.DataFrame(
        [
            {
                "chrom": "12",
                "id": "rs7134216",
                "pos": 100430850,
                "ref": "C",
                "alt": "T",
                "gt_raw": "0/1",
                "ad": [14, 13],
                "dp": 27,
                "gq": 61,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg02392575",
                "beta": 0.62,
                "chrom": "12",
                "pos": 100536662,
                "GencodeBasicV12_NAME": "UHRF1BP1L",
                "UCSC_RefGene_Name": "UHRF1BP1L;UHRF1BP1L",
                "UCSC_RefGene_Group": "TSS200;TSS200",
                "Relation_to_UCSC_CpG_Island": "S_Shore",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="12:100430850-100537652",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "rs7134216 (BLTP3B/UHRF1BP1L MYP3-region marker)"
    assert "Research association marker" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.62
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "MYP3 high-grade myopia locus thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_mt_rnr1_bundle_is_mitochondrial_pharmacogenetic_context() -> None:
    """MT-RNR1 should load as a mitochondrial pharmacogenetic bundle without EPIC probes."""
    knowledge_base = load_gene_interpretation_database("MT-RNR1")
    population_database = load_gene_population_database("MT-RNR1")
    synthesis_database = load_gene_synthesis_database("MT-RNR1")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "MT"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 648
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 1601
    assert knowledge_base["gene_context"]["promoter_review_region"]["start"] == 1
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "MT:1-1601"
    assert knowledge_base["gene_context"]["relevant_methylation_probe_ids"] == []
    assert population_database["database_name"].startswith("NophiGene MT-RNR1 Population")
    assert "aminoglycoside-induced hearing-loss risk" in synthesis_database["concrete_variant_prediction"]

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert {"m.1555A>G", "m.1494C>T", "m.1095T>C", "m.827A>G"} <= variant_ids

    variants = pd.DataFrame(
        [
            {
                "chrom": "MT",
                "id": ".",
                "pos": 64,
                "ref": "C",
                "alt": "T",
                "gt_raw": "0/1",
                "ad": [854, 4],
                "dp": 858,
                "qual": None,
                "filter_status": "PASS",
                "filter_pass": True,
            },
            {
                "chrom": "MT",
                "id": ".",
                "pos": 73,
                "ref": "A",
                "alt": "G",
                "gt_raw": "1/1",
                "ad": [0, 889],
                "dp": 889,
                "qual": None,
                "filter_status": "PASS",
                "filter_pass": True,
            },
            {
                "chrom": "MT",
                "id": ".",
                "pos": 143,
                "ref": "G",
                "alt": "A",
                "gt_raw": "0/1",
                "ad": [926, 3],
                "dp": 929,
                "qual": None,
                "filter_status": "PASS",
                "filter_pass": True,
            },
            {
                "chrom": "MT",
                "id": ".",
                "pos": 146,
                "ref": "T",
                "alt": "C",
                "gt_raw": "1/1",
                "ad": [0, 914],
                "dp": 914,
                "qual": None,
                "filter_status": "PASS",
                "filter_pass": True,
            },
            {
                "chrom": "MT",
                "id": "rs267606617",
                "pos": 1555,
                "ref": "A",
                "alt": "G",
                "gt_raw": "0/1",
                "ad": [9, 9],
                "dp": 18,
                "gq": 50,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(columns=["probe_id", "beta", "UCSC_RefGene_Group"])

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="MT:1-1601",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "MT-RNR1 m.1555A>G"
    assert interpretation["promoter_analysis"]["found_variant_count"] == 4
    assert {
        record["display"]
        for record in interpretation["promoter_analysis"]["found_variants"]
    } == {"MT:64 C>T", "MT:73 A>G", "MT:143 G>A", "MT:146 T>C"}
    assert "CPIC increased-risk" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] is None
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "CPIC increased-risk MT-RNR1 phenotype" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )
