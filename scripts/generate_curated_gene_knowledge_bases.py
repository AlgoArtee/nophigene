#!/usr/bin/env python3
"""Generate bundled gene knowledge bases and methylation manifest subsets."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helper_functions.filter_manifest_region import filter_probes_by_region, load_manifest, parse_region_string

GENE_DATA_DIR = PROJECT_ROOT / "src" / "gene_data"
MANIFEST_PATH = PROJECT_ROOT / "data" / "infinium-methylationepic-v-1-0-b5-manifest-file.csv"
ASSEMBLY = "GRCh37 / hg19"
VERSION = "2026-04-15"
EMPTY_MANIFEST_COLUMNS = [
    "IlmnID",
    "CHR",
    "MAPINFO",
    "CHR_hg38",
    "Start_hg38",
    "End_hg38",
    "UCSC_RefGene_Name",
    "UCSC_RefGene_Group",
    "Relation_to_UCSC_CpG_Island",
]

COMMON_POPULATION_CATEGORIES = [
    {"key": "Global pattern", "label": "Global pattern"},
    {"key": "Longevity cohorts", "label": "Longevity and healthy-aging cohorts"},
    {"key": "Disease cohorts", "label": "Disease-focused cohorts"},
    {"key": "Cancer cohorts", "label": "Cancer-focused cohorts"},
    {"key": "Metabolic cohorts", "label": "Metabolic and endocrine cohorts"},
    {"key": "Rare disease families", "label": "Rare-disease and familial series"},
]


def _evidence(label: str, url: str) -> dict[str, str]:
    return {"label": label, "url": url}


GENE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "gene_name": "GLP1R",
        "cytoband": "6p21.2",
        "chromosome": "6",
        "start": 39016574,
        "end": 39055519,
        "strand": "+",
        "coordinate_source": "Ensembl GRCh37 lookup for ENSG00000112164, aligned to the hg19 coordinate system used by this app",
        "gene_summary": (
            "GLP1R encodes the glucagon-like peptide-1 receptor, a class B G-protein-coupled receptor for GLP-1. "
            "Ligand binding activates cAMP signaling and helps regulate glucose-dependent insulin secretion, appetite, gastric emptying, energy balance, cardiometabolic biology, and neuroprotective research pathways."
        ),
        "clinical_context": (
            "The bundled GLP1R database is pharmacogenetic and metabolic-research oriented. GLP1R is a major drug target for GLP-1 receptor agonists used in type 2 diabetes and obesity, "
            "but the common variants seeded here are best interpreted as response or metabolic-trait modifiers rather than as diagnostic pathogenic alleles."
        ),
        "variant_effect_overview": [
            "Several GLP1R missense polymorphisms have been studied as modifiers of incretin effect, glycemic response, BMI trajectories, insulin secretion, and GLP-1 receptor agonist or DPP-4 inhibitor response.",
            "Current evidence is cohort and treatment-context dependent; variant effects should not be treated as universal predictions of semaglutide, liraglutide, dulaglutide, or gliptin response.",
            "Observed GLP1R variants are most useful as a pharmacogenetic and metabolic context layer alongside phenotype, treatment, ancestry, and genotype dosage.",
        ],
        "condition_research_overview": [
            "Type 2 diabetes and glucose-lowering treatment response studies involving GLP-1 receptor agonists and DPP-4 inhibitors.",
            "Obesity, appetite, gastric emptying, BMI growth, and gestational-diabetes exposure research.",
            "Cardiometabolic and neuroprotective research that frames GLP1R signaling as a systemic metabolic regulator.",
        ],
        "methylation_interpretation": (
            "GLP1R methylation should be treated as local regulatory context around an incretin-receptor gene. "
            "Promoter-proximal methylation may help frame transcriptional accessibility, but it should not be used as a direct substitute for receptor genotype, receptor expression, or medication-response testing."
        ),
        "methylation_effects": [
            "Promoter-focused methylation may suggest a more restrained or permissive local regulatory state for GLP1R expression potential.",
            "Because GLP1R biology is strongly tissue and treatment dependent, methylation values should be read as context rather than a stand-alone clinical biomarker.",
            "Variant and methylation signals are most defensible when interpreted as combined incretin-receptor regulatory background.",
        ],
        "methylation_condition_research": [
            "Diabetes and obesity pharmacogenetic studies in which receptor signaling can modify treatment response.",
            "Metabolic-trait research linking GLP-1 receptor signaling to insulin secretion, appetite, gastric emptying, and BMI trajectories.",
            "Epigenetic regulation studies that use promoter or TSS-proximal methylation as chromatin accessibility context.",
        ],
        "evidence": [
            _evidence("NCBI Gene 2740: GLP1R gene summary", "https://www.ncbi.nlm.nih.gov/gene/2740"),
            _evidence("UniProt P43220: GLP1R_HUMAN", "https://www.uniprot.org/uniprotkb/P43220/entry"),
            _evidence("PubMed 27160388: GLP1R Gly168Ser and gliptin response", "https://pubmed.ncbi.nlm.nih.gov/27160388/"),
            _evidence("PubMed 38172377: GLP1R variants and liraglutide response", "https://pubmed.ncbi.nlm.nih.gov/38172377/"),
            _evidence("PubMed 40443355: GLP-1R polymorphisms, GDM exposure, and offspring BMI", "https://pubmed.ncbi.nlm.nih.gov/40443355/"),
        ],
        "variants": [
            {
                "variant": "rs6923761",
                "display_name": "rs6923761 (Gly168Ser)",
                "common_name": "GLP1R Gly168Ser missense pharmacogenetic marker",
                "position": 39034072,
                "lookup_keys": [
                    "rs6923761",
                    "GLP1R:rs6923761",
                    "6:39034072",
                    "6:39034072:G>A",
                    "6:39034072:G>C",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pharmacogenetic research marker / metabolic response modifier",
                "clinical_interpretation": (
                    "rs6923761, commonly described as Gly168Ser, is a GLP1R missense variant studied as a modifier of incretin-related treatment response and metabolic traits. "
                    "A gliptin-response study reported lower HbA1c reduction in Ser/Ser homozygotes than in Gly-allele carriers, while later GLP-1 receptor agonist studies show treatment- and cohort-specific results."
                ),
                "clinical_significance": "Research-level pharmacogenetic association; not curated here as a diagnostic pathogenic allele.",
                "functional_effects": [
                    "Missense GLP1R variation may alter receptor signaling context or drug-response phenotype.",
                    "Reported response effects are medication and cohort dependent, with gliptin and GLP-1 receptor agonist literature not always pointing in the same direction.",
                ],
                "associated_conditions": [
                    "Type 2 diabetes treatment response",
                    "GLP-1 receptor agonist and DPP-4 inhibitor pharmacogenetics",
                    "BMI and gestational-diabetes exposure interaction studies",
                ],
                "research_context": [
                    "Use this marker as a candidate pharmacogenetic response signal, not as a deterministic medication-selection rule.",
                    "Genotype dosage matters because several cited findings compare homozygous or carrier groups.",
                ],
                "usual_variant_note": "Common GLP1R missense marker often discussed as Gly168Ser.",
                "methylation_interpretation": (
                    "Pair rs6923761 with GLP1R methylation only as receptor-locus regulatory context; the drug-response evidence is primarily sequence and treatment based."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 27160388: Gly168Ser and gliptin response", "https://pubmed.ncbi.nlm.nih.gov/27160388/"),
                    _evidence("PubMed 41307691: rs6923761 and oral semaglutide response", "https://pubmed.ncbi.nlm.nih.gov/41307691/"),
                    _evidence("PubMed 40443355: GLP-1R polymorphisms and offspring BMI", "https://pubmed.ncbi.nlm.nih.gov/40443355/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Javorsky et al., 2016 (PMID 27160388)",
                        "genotypes": "GLP1R Gly168Ser, including Ser/Ser versus Gly-allele carriers",
                        "phenotype": "HbA1c response after 6 months of gliptin treatment",
                        "finding": "The pilot study reported that Gly168Ser was associated with glycemic response to gliptins, with Ser/Ser homozygotes showing lower mean HbA1c reduction than Gly-allele carriers.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/27160388/",
                    },
                    {
                        "paper": "Candido et al., 2026 (PMID 41307691)",
                        "genotypes": "rs6923761 and rs761387",
                        "phenotype": "Oral semaglutide response in type 2 diabetes",
                        "finding": "The study title reports evaluation of GLP1 receptor rs6923761 and rs761387 as modifiers of oral semaglutide response, supporting pharmacogenetic response-context curation.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/41307691/",
                    },
                ],
            },
            {
                "variant": "rs10305420",
                "common_name": "GLP1R missense liraglutide-response marker",
                "position": 39016636,
                "lookup_keys": [
                    "rs10305420",
                    "GLP1R:rs10305420",
                    "6:39016636",
                    "6:39016636:C>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pharmacogenetic research marker / metabolic trait modifier",
                "clinical_interpretation": (
                    "rs10305420 is a GLP1R missense polymorphism studied in GLP-1 receptor agonist response and pediatric metabolic-trait cohorts. "
                    "In an Iranian type 2 diabetes liraglutide study, T-allele homozygosity was associated with optimal glycemic response; EPOCH analyses also frame this SNP as a modifier of BMI growth or metabolic traits."
                ),
                "clinical_significance": "Research-level pharmacogenetic and metabolic association; not diagnostic.",
                "functional_effects": [
                    "May mark altered receptor-response background for GLP-1 receptor agonist therapy in selected cohorts.",
                    "Also appears in pediatric and gestational-diabetes exposure interaction studies of BMI or metabolic traits.",
                ],
                "associated_conditions": [
                    "Type 2 diabetes glycemic response to liraglutide",
                    "BMI growth and metabolic traits during childhood/adolescence",
                    "Gestational diabetes exposure interaction research",
                ],
                "research_context": [
                    "Interpret directionally only when the observed allele and genotype dosage are known.",
                    "The local app now decodes FORMAT/GT and reports zygosity plus allele dosage; REF -> ALT is retained only as the site definition.",
                ],
                "usual_variant_note": "GLP1R missense marker studied in liraglutide and EPOCH metabolic cohorts.",
                "methylation_interpretation": (
                    "Methylation provides GLP1R regulatory context but does not substitute for genotype-dose pharmacogenetic analysis."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 38172377: rs10305420 and liraglutide response", "https://pubmed.ncbi.nlm.nih.gov/38172377/"),
                    _evidence("PubMed 39693247: GLP-1R polymorphisms and pediatric metabolic traits", "https://pubmed.ncbi.nlm.nih.gov/39693247/"),
                    _evidence("PubMed 40443355: GLP-1R polymorphisms and offspring BMI", "https://pubmed.ncbi.nlm.nih.gov/40443355/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Eghbali et al., 2024 (PMID 38172377)",
                        "genotypes": "rs10305420 T-allele homozygosity versus heterozygous and wild-type homozygous states",
                        "phenotype": "HbA1c response to liraglutide in Iranian people with type 2 diabetes",
                        "finding": "The study reported that rs10305420 T-allele homozygosity was associated with optimal glycemic response to liraglutide.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/38172377/",
                    },
                    {
                        "paper": "Harrall et al., 2025 (PMID 40443355)",
                        "genotypes": "rs10305420 CT or TT carrier states",
                        "phenotype": "Relationship between gestational diabetes exposure and offspring BMI growth",
                        "finding": "The EPOCH study reported that GLP-1R polymorphisms, including rs10305420, modified the association between gestational diabetes exposure and BMI growth among youth.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/40443355/",
                    },
                ],
            },
            {
                "variant": "rs3765467",
                "display_name": "rs3765467 (p.R131Q)",
                "common_name": "GLP1R R131Q missense metabolic marker",
                "position": 39033595,
                "lookup_keys": [
                    "rs3765467",
                    "GLP1R:rs3765467",
                    "6:39033595",
                    "6:39033595:G>A",
                    "6:39033595:G>C",
                    "6:39033595:G>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Metabolic disease association / pharmacogenetic case-context marker",
                "clinical_interpretation": (
                    "rs3765467 is a GLP1R missense variant described as p.R131Q in recent literature and studied in early-onset type 2 diabetes, dyslipidemia, gestational diabetes, and GLP-1 receptor agonist response contexts. "
                    "The bundled interpretation treats it as a cohort-specific metabolic marker rather than a deterministic disease allele."
                ),
                "clinical_significance": "Research-level metabolic association; not curated as pathogenic.",
                "functional_effects": [
                    "Missense change in the receptor protein with potential relevance to incretin signaling context.",
                    "Reported associations include early-onset type 2 diabetes risk and GLP-1 receptor agonist response case observations.",
                ],
                "associated_conditions": [
                    "Early-onset type 2 diabetes",
                    "Dyslipidemia in type 2 diabetes cohorts",
                    "Gestational diabetes susceptibility and glucose metabolism",
                    "Dulaglutide response case context",
                ],
                "research_context": [
                    "Most evidence is association or case-based; avoid upgrading this to a clinical call without external pathogenicity review.",
                    "Medication-response interpretation should remain cautious and cohort specific.",
                ],
                "usual_variant_note": "GLP1R p.R131Q missense marker studied in diabetes and treatment-response contexts.",
                "methylation_interpretation": (
                    "Methylation should be read as general GLP1R locus accessibility context, not a known direct consequence of rs3765467."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 38131033: rs3765467 and early-onset type 2 diabetes", "https://pubmed.ncbi.nlm.nih.gov/38131033/"),
                    _evidence("PubMed 38986908: p.R131Q and dulaglutide response case", "https://pubmed.ncbi.nlm.nih.gov/38986908/"),
                    _evidence("PubMed 36528605: GLP1R variants and gestational diabetes", "https://pubmed.ncbi.nlm.nih.gov/36528605/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Fang et al., 2023 (PMID 38131033)",
                        "genotypes": "GLP1R rs3765467 genotype groups",
                        "phenotype": "Early-onset type 2 diabetes risk",
                        "finding": "The study title reports an association between GLP1R rs3765467 and risk of early-onset type 2 diabetes, supporting metabolic-risk context curation.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/38131033/",
                    },
                    {
                        "paper": "Kato et al., 2025 (PMID 38986908)",
                        "genotypes": "rs3765467 / p.R131Q",
                        "phenotype": "Dulaglutide response in diabetes with myotonic dystrophy",
                        "finding": "The case report describes p.R131Q at GLP1R with marked effects of the GLP-1 receptor agonist dulaglutide, but this remains case-level evidence.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/38986908/",
                    },
                ],
            },
            {
                "variant": "rs1042044",
                "common_name": "GLP1R missense BMI/metabolic-trait marker",
                "position": 39041502,
                "lookup_keys": [
                    "rs1042044",
                    "GLP1R:rs1042044",
                    "6:39041502",
                    "6:39041502:C>A",
                    "6:39041502:C>G",
                    "6:39041502:C>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Metabolic trait interaction marker",
                "clinical_interpretation": (
                    "rs1042044 is a GLP1R missense variant included in EPOCH analyses of childhood and adolescent metabolic traits. "
                    "The local database treats it as a BMI and glucose-insulin homeostasis interaction marker rather than as a medication-response or diagnostic variant."
                ),
                "clinical_significance": "Research-level metabolic trait association; not diagnostic.",
                "functional_effects": [
                    "May tag receptor coding variation relevant to BMI and glucose-insulin trait heterogeneity.",
                    "Best interpreted in developmental, gestational-exposure, and metabolic-cohort context.",
                ],
                "associated_conditions": [
                    "BMI growth among youth",
                    "Insulin sensitivity and compensatory insulin secretion research",
                    "Gestational diabetes exposure interaction studies",
                ],
                "research_context": [
                    "EPOCH findings require replication and should not be converted into deterministic obesity or diabetes predictions.",
                ],
                "usual_variant_note": "GLP1R missense marker in pediatric BMI and metabolic-trait studies.",
                "methylation_interpretation": (
                    "No direct methylation consequence is bundled for rs1042044; read methylation as broader GLP1R regulatory context."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 39693247: GLP-1R polymorphisms and pediatric metabolic traits", "https://pubmed.ncbi.nlm.nih.gov/39693247/"),
                    _evidence("PubMed 40443355: GLP-1R polymorphisms and offspring BMI", "https://pubmed.ncbi.nlm.nih.gov/40443355/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Harrall et al., 2025 (PMID 39693247)",
                        "genotypes": "GLP-1R polymorphisms including rs1042044",
                        "phenotype": "Metabolic traits during childhood and adolescence",
                        "finding": "The study reported associations between GLP-1R polymorphisms and BMI, pubertal development, insulin sensitivity, and compensatory insulin secretion during adolescence.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/39693247/",
                    },
                    {
                        "paper": "Harrall et al., 2025 (PMID 40443355)",
                        "genotypes": "rs1042044 carrier states",
                        "phenotype": "Gestational diabetes exposure and offspring BMI growth",
                        "finding": "The EPOCH study reported that GLP-1R polymorphisms, including rs1042044, modified the relationship between gestational diabetes exposure and BMI growth.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/40443355/",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from GLP1R pharmacogenetic, diabetes, obesity, and metabolic-trait literature.",
        "population_coverage_note": (
            "The bundled GLP1R population database is literature oriented. It summarizes cohort and treatment-response patterns rather than providing a full ancestry-frequency panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 2740: GLP1R gene summary", "https://www.ncbi.nlm.nih.gov/gene/2740"),
            _evidence("UniProt P43220: GLP1R_HUMAN", "https://www.uniprot.org/uniprotkb/P43220/entry"),
            _evidence("PubMed 27160388: Gly168Ser gliptin response", "https://pubmed.ncbi.nlm.nih.gov/27160388/"),
            _evidence("PubMed 38172377: GLP1R variants and liraglutide response", "https://pubmed.ncbi.nlm.nih.gov/38172377/"),
            _evidence("PubMed 39693247: pediatric metabolic traits", "https://pubmed.ncbi.nlm.nih.gov/39693247/"),
            _evidence("PubMed 40443355: GDM exposure and offspring BMI", "https://pubmed.ncbi.nlm.nih.gov/40443355/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs6923761 / Gly168Ser",
                "location_group": "Metabolic cohorts",
                "summary": "The Gly168Ser signal is most often interpreted in type 2 diabetes treatment-response cohorts, with reported effects differing across gliptin and GLP-1 receptor agonist settings.",
                "evidence": [
                    _evidence("PubMed 27160388", "https://pubmed.ncbi.nlm.nih.gov/27160388/"),
                    _evidence("PubMed 41307691", "https://pubmed.ncbi.nlm.nih.gov/41307691/"),
                ],
            },
            {
                "variant": "rs10305420",
                "location_group": "Metabolic cohorts",
                "summary": "rs10305420 has been reported as a liraglutide-response marker in an Iranian type 2 diabetes cohort and as part of GLP-1R metabolic-trait interaction analyses in youth.",
                "evidence": [
                    _evidence("PubMed 38172377", "https://pubmed.ncbi.nlm.nih.gov/38172377/"),
                    _evidence("PubMed 40443355", "https://pubmed.ncbi.nlm.nih.gov/40443355/"),
                ],
            },
            {
                "variant": "rs3765467 / p.R131Q",
                "location_group": "Disease cohorts",
                "summary": "rs3765467 is discussed in early-onset type 2 diabetes, dyslipidemia, gestational diabetes, and case-level GLP-1 receptor agonist response literature, so population interpretation should remain cohort specific.",
                "evidence": [
                    _evidence("PubMed 38131033", "https://pubmed.ncbi.nlm.nih.gov/38131033/"),
                    _evidence("PubMed 38986908", "https://pubmed.ncbi.nlm.nih.gov/38986908/"),
                ],
            },
            {
                "variant": "Common GLP1R coding polymorphisms",
                "location_group": "Global pattern",
                "summary": "Across GLP1R studies, medication, ancestry, baseline metabolic state, and genotype dosage are key modifiers, so a single common SNP should not be treated as a universal predictor of GLP-1 therapy outcome.",
            },
        ],
    },
    {
        "gene_name": "FOXO3",
        "cytoband": "6q21",
        "chromosome": "6",
        "start": 108881028,
        "end": 109005977,
        "strand": "+",
        "gene_summary": (
            "FOXO3 encodes a forkhead-box transcription factor that coordinates stress resistance, "
            "autophagy, apoptosis, stem-cell maintenance, and metabolic adaptation downstream of "
            "insulin-IGF and nutrient-signaling pathways."
        ),
        "clinical_context": (
            "The bundled FOXO3 database is intended for pathway and longevity research context rather "
            "than for monogenic disease reporting. FOXO3 is one of the most replicated human longevity "
            "genes, but common variants are interpreted as modest effect-size healthy-aging modifiers."
        ),
        "variant_effect_overview": [
            "The strongest common FOXO3 signals involve intronic enhancer or haplotype markers rather than protein-truncating alleles.",
            "Published longevity associations are ancestry- and cohort-dependent and should be treated as probabilistic healthy-aging signals, not deterministic predictions.",
            "Observed PASS variants in the FOXO3 interval are therefore best read as locus context unless they directly match a curated longevity-associated marker or have outside clinical evidence.",
        ],
        "condition_research_overview": [
            "FOXO3 is repeatedly linked to exceptional longevity and survival to advanced age across multiple cohort designs.",
            "The gene is also studied in cardiometabolic resilience, oxidative-stress biology, immune regulation, and stem-cell maintenance.",
            "FOXO3 sits in the broader insulin-IGF-AMPK-mTOR longevity network, so interpretation benefits from pathway-level context.",
        ],
        "methylation_interpretation": (
            "FOXO3 methylation is best interpreted as chromatin and regulatory context around a stress-response transcription factor. "
            "Promoter-proximal methylation or broader locus methylation can inform whether the sampled tissue appears permissive or restrained, "
            "but it should not be treated as a direct readout of any one longevity SNP."
        ),
        "methylation_effects": [
            "Promoter-focused FOXO3 methylation is usually interpreted as regulatory context for transcriptional accessibility rather than as a stand-alone biomarker.",
            "Because FOXO3 activity is highly context-dependent, methylation readouts should be integrated with pathway state, tissue type, and age-related biology.",
            "When FOXO3 variants and methylation are discussed together, the most defensible interpretation is pathway tuning rather than one-to-one deterministic coupling.",
        ],
        "methylation_condition_research": [
            "Healthy-aging and lifespan studies that frame FOXO3 as a stress-response resilience factor.",
            "Cardiometabolic and inflammatory research that interprets FOXO3 as a node linking nutrient signaling to cellular maintenance.",
            "Epigenetic aging and chromatin-regulation studies that examine FOXO-family transcriptional accessibility.",
        ],
        "evidence": [
            _evidence("NCBI Gene 2309: FOXO3 gene summary", "https://www.ncbi.nlm.nih.gov/gene/2309"),
            _evidence("PMCID PMC5403515: FOXO3 as a major human longevity gene", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5403515/"),
            _evidence("PubMed 29497356: rs2802292 creates a stress-responsive FOXO3 enhancer", "https://pubmed.ncbi.nlm.nih.gov/29497356/"),
        ],
        "variants": [
            {
                "variant": "rs2802292",
                "common_name": "longevity-associated intronic enhancer SNP",
                "region_class": "gene_body",
                "interpretation_scope": "Research association / healthy-aging modifier",
                "clinical_interpretation": (
                    "rs2802292 is the best-known common FOXO3 longevity marker. The G allele has been associated "
                    "with survival to advanced age in multiple cohorts and is interpreted here as a research-grade "
                    "healthy-aging modifier rather than as a pathogenic or diagnostic allele."
                ),
                "clinical_significance": "Research-level longevity association; not a monogenic disease variant.",
                "functional_effects": [
                    "The longevity-associated allele has been linked to stress-responsive enhancer behavior and higher FOXO3 expression under cellular stress.",
                    "Any effect is best viewed as pathway tuning across stress resistance and metabolic adaptation rather than as deterministic causality.",
                ],
                "associated_conditions": [
                    "Exceptional longevity and survival to advanced age",
                    "Cardiometabolic resilience and lower age-related disease burden in some cohorts",
                    "Stress-response and cellular-maintenance biology",
                ],
                "research_context": [
                    "This is the canonical common FOXO3 longevity SNP used in centenarian and healthy-aging studies.",
                    "Effect size is modest and cohort dependent, so replication and ancestry matching matter.",
                ],
                "usual_variant_note": "Most cited common FOXO3 longevity marker.",
                "methylation_interpretation": (
                    "Treat rs2802292 alongside FOXO3 methylation as shared regulatory context rather than as a direct methylation biomarker."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PMCID PMC5403515: FOXO3 longevity review", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5403515/"),
                    _evidence("PubMed 29497356: rs2802292 enhancer mechanism", "https://pubmed.ncbi.nlm.nih.gov/29497356/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Morris et al., 2015 (PMCID PMC5403515)",
                        "genotypes": "GG, GT, and TT",
                        "phenotype": "Human longevity across replicated cohort studies",
                        "finding": "The G allele is repeatedly described as enriched in long-lived individuals, supporting a modest but reproducible healthy-aging association.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5403515/",
                    },
                    {
                        "paper": "Flachsbart et al., 2017/2018 (PMID 29497356)",
                        "genotypes": "G-allele carriers versus non-carriers",
                        "phenotype": "Stress-responsive FOXO3 expression control",
                        "finding": "Mechanistic follow-up work proposed that the rs2802292 G allele helps create or stabilize a stress-responsive enhancer element that promotes FOXO3 upregulation.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/29497356/",
                    },
                ],
            }
        ],
        "population_intro": "Broader population patterns curated from FOXO3 healthy-aging, longevity, and stress-response literature.",
        "population_coverage_note": (
            "This FOXO3 population database emphasizes cohort-level longevity patterns and ancestry-aware interpretation. "
            "It does not yet ship a full embedded allele-frequency panel for every reported FOXO3 marker."
        ),
        "population_sources": [
            _evidence("NCBI Gene 2309: FOXO3 gene summary", "https://www.ncbi.nlm.nih.gov/gene/2309"),
            _evidence("PMCID PMC5403515: FOXO3 longevity review", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5403515/"),
            _evidence("PubMed 29497356: rs2802292 enhancer mechanism", "https://pubmed.ncbi.nlm.nih.gov/29497356/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs2802292",
                "location_group": "Longevity cohorts",
                "summary": "FOXO3 longevity signals are most often discussed in centenarian or long-lived cohort designs, where rs2802292 is treated as a modest survival-enrichment marker rather than a deterministic lifespan allele.",
                "evidence": [
                    _evidence("PMCID PMC5403515", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5403515/"),
                ],
            },
            {
                "variant": "FOXO3 longevity haplotypes",
                "location_group": "Global pattern",
                "summary": "Different populations tag the FOXO3 longevity signal with different SNPs or haplotypes, so cross-cohort interpretation should focus on the locus-level signal rather than one ancestry-specific marker alone.",
                "evidence": [
                    _evidence("PMCID PMC5403515", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5403515/"),
                ],
            },
        ],
    },
    {
        "gene_name": "MTOR",
        "cytoband": "1p36.22",
        "chromosome": "1",
        "start": 11166592,
        "end": 11322608,
        "strand": "-",
        "gene_summary": (
            "MTOR encodes the catalytic kinase shared by mTORC1 and mTORC2, integrating nutrient availability, growth-factor signaling, "
            "stress cues, and cellular energy state to regulate protein synthesis, autophagy, survival, and cytoskeletal programs."
        ),
        "clinical_context": (
            "The local MTOR knowledge base is designed for signaling and research interpretation. Pathogenic MTOR variants do exist in developmental "
            "and focal cortical dysplasia syndromes, but the bundled records here focus on commonly discussed regulatory polymorphisms and pathway context."
        ),
        "variant_effect_overview": [
            "Common MTOR polymorphisms are mainly interpreted as expression or signaling modifiers with context-dependent effect sizes.",
            "The strongest disease literature around MTOR centers on pathway dysregulation, drug response, and rare activating or inactivating variants rather than on one universal common SNP.",
            "Observed variants in this region should therefore be read as pathway-context markers unless they directly overlap a curated research SNP or have outside clinical curation.",
        ],
        "condition_research_overview": [
            "MTOR sits at the center of growth, anabolic signaling, and autophagy control, making it a core aging and cancer pathway gene.",
            "Clinical research links MTOR to immunosuppression, targeted oncology, developmental syndromes, and neurodevelopmental disease.",
            "Common regulatory SNP work is mostly cancer-risk or outcome oriented rather than diagnostic on its own.",
        ],
        "methylation_interpretation": (
            "MTOR methylation is best treated as pathway-regulatory context around a very large signaling locus. "
            "Promoter-proximal methylation may reflect transcriptional accessibility, but most interpretation remains pathway level rather than variant specific."
        ),
        "methylation_effects": [
            "Promoter-focused methylation may reflect altered MTOR expression potential, but it is not a stand-alone disease classifier.",
            "Given the size of the locus and multiple transcripts, methylation should be interpreted as a broad regulatory view rather than a one-site assay.",
            "MTOR methylation is most meaningful when read alongside AMPK, nutrient-sensing, and growth-signaling biology.",
        ],
        "methylation_condition_research": [
            "Cancer and targeted-therapy studies that frame mTOR activity as a proliferation and drug-response axis.",
            "Aging and nutrient-signaling research focused on mTORC1 inhibition, autophagy, and proteostasis.",
            "Neurodevelopmental and cortical dysplasia research in which mTOR pathway dysregulation is a major mechanistic theme.",
        ],
        "evidence": [
            _evidence("NCBI Gene 2475: MTOR gene summary", "https://www.ncbi.nlm.nih.gov/gene/2475"),
            _evidence("PubMed 24816861: meta-analysis of MTOR polymorphisms and cancer risk", "https://pubmed.ncbi.nlm.nih.gov/24816861/"),
            _evidence("PubMed 29978580: rs2536 and gastric-cancer survival", "https://pubmed.ncbi.nlm.nih.gov/29978580/"),
        ],
        "variants": [
            {
                "variant": "rs2295080",
                "common_name": "promoter regulatory MTOR SNP",
                "region_class": "promoter",
                "interpretation_scope": "Research association / regulatory marker",
                "clinical_interpretation": (
                    "rs2295080 is a promoter-proximal MTOR polymorphism studied as a transcriptional and cancer-risk modifier. "
                    "The local database treats it as a regulatory research marker rather than as a pathogenic clinical allele."
                ),
                "clinical_significance": "Research-level regulatory association; not seeded as pathogenic.",
                "functional_effects": [
                    "Promoter variation at rs2295080 has been discussed as a potential modifier of MTOR transcriptional output.",
                    "Reported effects are strongest in cancer-association studies and remain cohort dependent.",
                ],
                "associated_conditions": [
                    "Cancer susceptibility and clinical outcome studies",
                    "Pathway-level growth and survival signaling research",
                ],
                "research_context": [
                    "This SNP is commonly bundled with other MTOR polymorphisms in association meta-analyses.",
                    "Interpret it as a modest signaling modifier rather than a stand-alone disease determinant.",
                ],
                "usual_variant_note": "Common promoter-facing MTOR regulatory SNP.",
                "methylation_interpretation": (
                    "Promoter methylation and rs2295080 should be read together only as shared transcriptional context."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 24816861: MTOR polymorphism meta-analysis", "https://pubmed.ncbi.nlm.nih.gov/24816861/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Xu et al., 2014 (PMID 24816861)",
                        "genotypes": "TT versus GT/GG",
                        "phenotype": "Cancer risk across pooled association studies",
                        "finding": "The meta-analysis reported that the rs2295080 TT genotype was associated with higher cancer risk in the Chinese studies included, supporting a modest context-specific regulatory effect.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/24816861/",
                    }
                ],
            },
            {
                "variant": "rs2536",
                "common_name": "3'UTR functional MTOR SNP",
                "region_class": "gene_body",
                "interpretation_scope": "Research association / outcome modifier",
                "clinical_interpretation": (
                    "rs2536 is a commonly discussed MTOR 3'UTR polymorphism curated here as a research association marker for cancer susceptibility and outcome, "
                    "not as a diagnostic or pathogenic allele."
                ),
                "clinical_significance": "Research-level association; best interpreted in cohort context.",
                "functional_effects": [
                    "The variant is discussed as a post-transcriptional regulatory marker in the MTOR pathway.",
                    "Outcome studies suggest it may stratify prognosis in selected cancer cohorts.",
                ],
                "associated_conditions": [
                    "Gastric cancer risk and survival studies",
                    "Broader oncology association meta-analyses",
                ],
                "research_context": [
                    "This is one of the most frequently cited common MTOR SNPs in outcome-oriented cancer literature.",
                ],
                "usual_variant_note": "Common MTOR 3'UTR association SNP.",
                "methylation_interpretation": (
                    "No one-to-one methylation consequence is bundled for rs2536; treat methylation as broader MTOR locus context."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 29978580: rs2536 and gastric-cancer survival", "https://pubmed.ncbi.nlm.nih.gov/29978580/"),
                    _evidence("PubMed 24816861: MTOR polymorphism meta-analysis", "https://pubmed.ncbi.nlm.nih.gov/24816861/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Cheng et al., 2019 (PMID 29978580)",
                        "genotypes": "Common rs2536 genotypes",
                        "phenotype": "Survival of Chinese gastric-cancer patients",
                        "finding": "The study reported that rs2536 was functionally relevant in survival modeling for gastric-cancer patients, reinforcing its use as a regulatory and prognostic research marker.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/29978580/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from MTOR cancer-association, therapeutic-response, and signaling literature.",
        "population_coverage_note": (
            "The bundled MTOR population database is literature oriented. It currently summarizes cohort patterns and pathway interpretation rather than shipping a full allele-frequency reference panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 2475: MTOR gene summary", "https://www.ncbi.nlm.nih.gov/gene/2475"),
            _evidence("PubMed 24816861: MTOR polymorphism meta-analysis", "https://pubmed.ncbi.nlm.nih.gov/24816861/"),
            _evidence("PubMed 29978580: rs2536 survival study", "https://pubmed.ncbi.nlm.nih.gov/29978580/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs2295080 and rs2536",
                "location_group": "Cancer cohorts",
                "summary": "Most common-SNP MTOR literature comes from cancer cohorts in East Asian populations, where promoter and 3'UTR markers are interpreted as modest expression or prognosis modifiers.",
            },
            {
                "variant": "Common MTOR polymorphisms",
                "location_group": "Global pattern",
                "summary": "Because mTOR biology is heavily pathway driven, cross-population interpretation should emphasize signaling context and cohort design rather than assuming a universal effect from one common SNP.",
            },
        ],
    },
    {
        "gene_name": "RPS6",
        "cytoband": "9p22.1",
        "chromosome": "9",
        "start": 19375713,
        "end": 19380234,
        "strand": "-",
        "gene_summary": (
            "RPS6 encodes the ribosomal protein S6, a core 40S subunit protein and major substrate of S6 kinases downstream of mTOR signaling. "
            "Its phosphorylation state is widely used as a readout of growth-factor and nutrient-responsive translation control."
        ),
        "clinical_context": (
            "The bundled RPS6 knowledge base is pathway oriented. RPS6 is typically interpreted as a translational-control and mTOR-output node rather than as a common-SNP clinical gene in routine VCF review."
        ),
        "variant_effect_overview": [
            "RPS6 is most commonly discussed as a signaling substrate and biomarker of pathway activity rather than as a locus with a broad common-variant literature.",
            "Observed variants in the compact RPS6 interval should therefore be treated as locus context unless outside evidence supports a specific interpretation.",
        ],
        "condition_research_overview": [
            "RPS6 phosphorylation is a canonical downstream readout of mTORC1-S6K activity in cancer and growth biology.",
            "The gene is relevant to translational control, cell growth, proliferation, and treatment-response biomarker work.",
        ],
        "methylation_interpretation": (
            "RPS6 methylation should be read as local chromatin context around a compact ribosomal-protein locus. "
            "Because the biology of interest is often protein phosphorylation rather than gene methylation, these values are mainly supportive context."
        ),
        "methylation_effects": [
            "Promoter methylation can provide a rough view of transcriptional accessibility, but most RPS6 functional work centers on protein phosphorylation and translational output.",
            "Interpret methylation alongside mTOR-S6K pathway activity rather than as a stand-alone biomarker.",
        ],
        "methylation_condition_research": [
            "Cancer biomarker studies using phospho-S6 as an mTOR-pathway activity readout.",
            "Translational-control and growth-signaling research.",
        ],
        "evidence": [
            _evidence("NCBI Gene 6194: RPS6 gene summary", "https://www.ncbi.nlm.nih.gov/gene/6194"),
            _evidence("PubMed 18157089: phospho-S6 as a marker for targeted mTOR therapy", "https://pubmed.ncbi.nlm.nih.gov/18157089/"),
        ],
        "variants": [],
        "population_intro": "Broader population patterns curated from RPS6 pathway-biomarker and translational-control literature.",
        "population_coverage_note": (
            "Clinically relevant RPS6 interpretation is usually pathway and biomarker driven rather than based on common allele-frequency panels."
        ),
        "population_sources": [
            _evidence("NCBI Gene 6194: RPS6 gene summary", "https://www.ncbi.nlm.nih.gov/gene/6194"),
            _evidence("PubMed 18157089: phospho-S6 biomarker paper", "https://pubmed.ncbi.nlm.nih.gov/18157089/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "RPS6 pathway context",
                "location_group": "Global pattern",
                "summary": "RPS6 is generally interpreted through pathway activation and phospho-S6 biomarker studies rather than through a dense catalog of common inherited markers.",
            }
        ],
    },
    {
        "gene_name": "SIK3",
        "cytoband": "11q23.3",
        "chromosome": "11",
        "start": 116714118,
        "end": 116969144,
        "strand": "-",
        "gene_summary": (
            "SIK3 encodes a salt-inducible serine-threonine kinase in the AMPK-related kinase family that participates in metabolic control, "
            "sleep regulation, transcriptional programs, and TOR-linked signaling."
        ),
        "clinical_context": (
            "The local SIK3 knowledge base focuses on emerging functional and sleep-trait biology. The locus has a growing GWAS and mechanistic literature, "
            "but interpretation is still mainly research grade for most variants."
        ),
        "variant_effect_overview": [
            "SIK3 is better characterized through pathway studies, sleep phenotypes, and selected rare functional variants than through a mature clinical common-SNP catalog.",
            "Observed variants in the SIK3 interval should be interpreted as research context unless they match a seeded functional record or have outside clinical curation.",
        ],
        "condition_research_overview": [
            "Human and model-organism studies link SIK3 to sleep need, circadian or sleep-duration biology, and neuronal regulation.",
            "Additional research connects SIK3 to metabolism, adiposity, and oncogenic or transcriptional signaling programs.",
        ],
        "methylation_interpretation": (
            "SIK3 methylation is best treated as regulatory context around an emerging signaling gene. "
            "Interpret any promoter-proximal patterns as supportive evidence rather than as a direct proxy for sleep or metabolic phenotype."
        ),
        "methylation_effects": [
            "Methylation may help contextualize transcriptional accessibility in tissues where SIK3 is active, but functional interpretation remains pathway level.",
            "Sleep-trait and signaling phenotypes should not be inferred from methylation alone.",
        ],
        "methylation_condition_research": [
            "Sleep and circadian-biology studies.",
            "Metabolic-signaling and adiposity research.",
            "Transcriptional-control and leukemia-dependency studies involving SIK kinases.",
        ],
        "evidence": [
            _evidence("NCBI Gene 23387: SIK3 gene summary", "https://www.ncbi.nlm.nih.gov/gene/23387"),
            _evidence("DOI 10.1073/pnas.2500356122: SIK3-N783Y and natural short sleep", "https://doi.org/10.1073/pnas.2500356122"),
            _evidence("PubMed 32126566: SIK3 haplotypes and noise-induced hearing loss", "https://pubmed.ncbi.nlm.nih.gov/32126566/"),
        ],
        "variants": [
            {
                "variant": "SIK3 p.Asn783Tyr",
                "display_name": "SIK3 p.Asn783Tyr (N783Y)",
                "common_name": "natural short sleep-associated missense variant",
                "region_class": "gene_body",
                "interpretation_scope": "Rare functional sleep-trait variant",
                "clinical_interpretation": (
                    "SIK3 p.Asn783Tyr is a rare functional missense variant reported in association with the human natural short sleep trait. "
                    "It is curated here as a mechanistic sleep-biology variant, not as a routine pathogenic diagnostic allele."
                ),
                "clinical_significance": "Rare functional research variant with emerging human sleep-trait evidence.",
                "functional_effects": [
                    "The variant is discussed as a gain-of-function or altered-function SIK3 allele in sleep regulation.",
                    "Its interpretation is primarily mechanistic and research facing at present.",
                ],
                "associated_conditions": [
                    "Human natural short sleep trait",
                    "Sleep-need and neuronal signaling research",
                ],
                "research_context": [
                    "This record captures a rare high-interest functional variant rather than a common population marker.",
                ],
                "usual_variant_note": "Emerging rare SIK3 human sleep variant.",
                "methylation_interpretation": (
                    "SIK3 methylation provides broad locus context but does not directly read out the functional consequence of p.Asn783Tyr."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("DOI 10.1073/pnas.2500356122: SIK3-N783Y and natural short sleep", "https://doi.org/10.1073/pnas.2500356122"),
                ],
                "literature_findings": [
                    {
                        "paper": "PNAS, 2025 (DOI 10.1073/pnas.2500356122)",
                        "genotypes": "Rare missense carrier state",
                        "phenotype": "Human natural short sleep",
                        "finding": "The report linked SIK3 N783Y to the human natural short sleep trait and framed the variant as a direct functional probe of sleep-need biology.",
                        "url": "https://doi.org/10.1073/pnas.2500356122",
                    }
                ],
            }
        ],
        "population_intro": "Broader population patterns curated from SIK3 sleep, hearing, and metabolic-signaling literature.",
        "population_coverage_note": (
            "SIK3 population interpretation currently depends more on cohort-specific association studies and rare-variant reports than on a mature embedded allele-frequency panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 23387: SIK3 gene summary", "https://www.ncbi.nlm.nih.gov/gene/23387"),
            _evidence("PubMed 32126566: SIK3 haplotypes and hearing-loss risk", "https://pubmed.ncbi.nlm.nih.gov/32126566/"),
            _evidence("DOI 10.1073/pnas.2500356122: SIK3-N783Y and natural short sleep", "https://doi.org/10.1073/pnas.2500356122"),
        ],
        "gene_population_patterns": [
            {
                "variant": "SIK3 haplotypes",
                "location_group": "Disease cohorts",
                "summary": "Published common-variant SIK3 literature is still cohort specific, including hearing-loss and adiposity studies, so the locus is best interpreted as emerging rather than fully settled population genetics.",
            },
            {
                "variant": "SIK3 p.Asn783Tyr",
                "location_group": "Rare disease families",
                "summary": "The most striking human SIK3 result to date is a rare functional sleep-trait variant, underscoring that this gene is currently driven by mechanistic and rare-variant evidence more than by broad common-SNP panels.",
            },
        ],
    },
    {
        "gene_name": "FLCN",
        "cytoband": "17p11.2",
        "chromosome": "17",
        "start": 17115526,
        "end": 17140482,
        "strand": "-",
        "gene_summary": (
            "FLCN encodes folliculin, a tumor suppressor involved in lysosomal signaling, AMPK-mTOR regulation, TFEB-TFE3 control, and renal and pulmonary homeostasis."
        ),
        "clinical_context": (
            "The strongest clinical interpretation for FLCN is rare-disease oriented. Germline loss-of-function variants cause Birt-Hogg-Dube syndrome, "
            "so the local database emphasizes rare pathogenic and tumor-suppressor biology rather than common-SNP association signals."
        ),
        "variant_effect_overview": [
            "The clinically important FLCN variants are usually rare truncating, splice, or deletion events rather than common polymorphisms.",
            "Observed PASS variants in the FLCN interval should therefore be treated as locus context unless they overlap a known rare pathogenic event or have independent clinical curation.",
        ],
        "condition_research_overview": [
            "FLCN is the canonical gene for Birt-Hogg-Dube syndrome with renal-tumor, pulmonary-cyst, and pneumothorax risk.",
            "The gene is also studied in lysosome signaling, TFEB/TFE3 regulation, and AMPK-mTOR pathway cross-talk.",
        ],
        "methylation_interpretation": (
            "FLCN methylation is best read as tumor-suppressor regulatory context. It can help frame whether the locus appears transcriptionally restrained or accessible, "
            "but it does not replace rare-variant interpretation in Birt-Hogg-Dube syndrome."
        ),
        "methylation_effects": [
            "For FLCN, rare coding or structural variants usually carry more clinical weight than methylation patterns.",
            "Methylation can still provide useful context in renal-tumor and tumor-suppressor regulation studies.",
        ],
        "methylation_condition_research": [
            "Birt-Hogg-Dube syndrome tumor-suppressor biology.",
            "Renal neoplasia and pneumothorax predisposition research.",
        ],
        "evidence": [
            _evidence("NCBI Gene 201163: FLCN gene summary", "https://www.ncbi.nlm.nih.gov/gene/201163"),
            _evidence("PubMed 21538689: BHD-associated FLCN mutations disrupt protein stability", "https://pubmed.ncbi.nlm.nih.gov/21538689/"),
            _evidence("PubMed 22146830: renal cancer and pneumothorax risk in FLCN mutation carriers", "https://pubmed.ncbi.nlm.nih.gov/22146830/"),
        ],
        "variants": [
            {
                "variant": "FLCN c.1285dupC",
                "display_name": "FLCN c.1285dupC",
                "common_name": "Birt-Hogg-Dube frameshift hotspot",
                "region_class": "gene_body",
                "interpretation_scope": "Rare pathogenic tumor-suppressor variant",
                "clinical_interpretation": (
                    "c.1285dupC is one of the best-known recurrent FLCN frameshift variants in Birt-Hogg-Dube syndrome and is interpreted as a pathogenic rare-disease allele. "
                    "It is bundled here as a named clinical hotspot for context, although the SNP-oriented VCF preview is not optimized to call it directly."
                ),
                "clinical_significance": "Pathogenic loss-of-function hotspot in Birt-Hogg-Dube syndrome.",
                "functional_effects": [
                    "Frameshift loss of folliculin function with tumor-suppressor consequences.",
                    "Expected to disrupt lysosomal and AMPK-mTOR regulatory roles of FLCN.",
                ],
                "associated_conditions": [
                    "Birt-Hogg-Dube syndrome",
                    "Renal tumors, lung cysts, and spontaneous pneumothorax",
                ],
                "research_context": [
                    "This is a rare pathogenic hotspot rather than a common association marker.",
                ],
                "usual_variant_note": "Classic recurrent BHD loss-of-function variant.",
                "methylation_interpretation": (
                    "FLCN methylation can add tumor-suppressor context, but it does not substitute for direct rare-variant testing when Birt-Hogg-Dube syndrome is suspected."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PubMed 21538689: BHD-associated FLCN mutations", "https://pubmed.ncbi.nlm.nih.gov/21538689/"),
                    _evidence("PubMed 22146830: phenotype risk in FLCN carriers", "https://pubmed.ncbi.nlm.nih.gov/22146830/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Nahorski et al., 2011 (PMID 21538689)",
                        "genotypes": "Rare germline loss-of-function carrier state",
                        "phenotype": "Birt-Hogg-Dube syndrome protein-stability consequences",
                        "finding": "The study framed recurrent FLCN truncating mutations as protein-destabilizing tumor-suppressor events central to BHD pathogenesis.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/21538689/",
                    }
                ],
            }
        ],
        "population_intro": "Broader population patterns curated from FLCN rare-disease, renal-tumor, and pneumothorax literature.",
        "population_coverage_note": (
            "FLCN population interpretation is primarily family and rare-variant driven. The current database therefore emphasizes rare-disease patterns rather than common allele-frequency panels."
        ),
        "population_sources": [
            _evidence("NCBI Gene 201163: FLCN gene summary", "https://www.ncbi.nlm.nih.gov/gene/201163"),
            _evidence("PubMed 22146830: FLCN carrier phenotype analysis", "https://pubmed.ncbi.nlm.nih.gov/22146830/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "Rare FLCN loss-of-function variants",
                "location_group": "Rare disease families",
                "summary": "FLCN clinical interpretation is dominated by rare germline truncating or deletion events that segregate in Birt-Hogg-Dube families rather than by common population markers.",
            },
            {
                "variant": "Birt-Hogg-Dube syndrome",
                "location_group": "Global pattern",
                "summary": "Across populations, the major interpretation question for FLCN is whether a rare loss-of-function event is present and whether renal and pulmonary surveillance is warranted, not whether a common SNP modestly shifts risk.",
            },
        ],
    },
    {
        "gene_name": "HERC2",
        "cytoband": "15q13.1",
        "chromosome": "15",
        "start": 28356186,
        "end": 28567325,
        "strand": "-",
        "coordinate_source": "NCBI Gene 8924 GRCh37.p13 interval NC_000015.9:28356186..28567325, complement",
        "curated_methylation_probe_ids": [
            "cg14091419",
            "cg14549093",
            "cg12605148",
            "cg14390018",
            "cg15464645",
            "cg21575005",
            "cg16895086",
            "cg00855398",
            "cg23040946",
            "cg07567273",
        ],
        "gene_summary": (
            "HERC2 encodes a large HECT and RCC1-like-domain E3 ubiquitin-protein ligase involved in protein trafficking, DNA-damage response biology, and neurodevelopmental disease when rare biallelic pathogenic variants are present. "
            "This local bundle focuses on the well-studied HERC2 intronic regulatory interval that controls the neighboring OCA2 promoter and strongly shapes human iris pigmentation."
        ),
        "clinical_context": (
            "The bundled HERC2 database is trait-prediction and pigmentation-research oriented. Common HERC2/OCA2 markers such as rs12913832 can support probabilistic blue-versus-brown eye-colour inference, "
            "but this is not a diagnosis, and visible eye colour remains polygenic, ancestry dependent, and modified by additional OCA2 and pigmentation loci."
        ),
        "variant_effect_overview": [
            "The HERC2/OCA2 rs12913832 enhancer marker is the strongest single local predictor for blue-versus-brown iris pigmentation in many European-ancestry cohorts.",
            "The forward-strand G allele representation used by this app is interpreted as the lighter-eye-associated state because it weakens enhancer-promoter support for OCA2 expression and reduces iris melanin biology.",
            "Secondary markers such as rs1129038, rs7170852, rs916977, and rs11636232 should refine, not replace, the rs12913832 genotype-dosage thesis.",
        ],
        "condition_research_overview": [
            "Human iris pigmentation and forensic DNA phenotyping.",
            "HERC2 enhancer regulation of OCA2 expression and melanocyte melanin production.",
            "Population structure and ancestry-aware interpretation of pigmentation-associated haplotypes.",
        ],
        "methylation_interpretation": (
            "HERC2 methylation should be read as regulatory context near the HERC2/OCA2 pigmentation interval. "
            "A CpG signal can support local chromatin-accessibility review, but eye-colour prediction in this app is anchored to GT-decoded HERC2/OCA2 SNP dosage rather than methylation alone."
        ),
        "methylation_effects": [
            "Promoter-proximal or enhancer-near methylation can suggest a more restrained or permissive local regulatory state for the HERC2/OCA2 interval.",
            "Because the main eye-colour signal is genotype driven, methylation is supporting context and not a substitute for rs12913832 dosage.",
            "Combined SNP and methylation evidence is most useful when interpreted as pigmentation pathway context rather than a deterministic appearance call.",
        ],
        "methylation_condition_research": [
            "Epigenetic regulation around HERC2 and OCA2 in melanocyte and pigmentation biology.",
            "Variant-plus-methylation review for iris pigmentation, hair pigmentation, and OCA2 expression context.",
        ],
        "evidence": [
            _evidence("NCBI Gene 8924: HERC2 GRCh37 coordinate source", "https://www.ncbi.nlm.nih.gov/gene/8924"),
            _evidence("UniProt O95714: HERC2_HUMAN", "https://www.uniprot.org/uniprotkb/O95714/entry"),
            _evidence("PubMed 18252222: rs12913832 predicts blue-brown eye colour", "https://pubmed.ncbi.nlm.nih.gov/18252222/"),
            _evidence("PubMed 22234890: rs12913832 enhancer attenuates HERC2-OCA2 chromatin looping", "https://pubmed.ncbi.nlm.nih.gov/22234890/"),
            _evidence("PubMed 20457063: HERC2 and OCA2 eye-colour markers", "https://pubmed.ncbi.nlm.nih.gov/20457063/"),
        ],
        "concrete_variant_prediction": (
            "A GT-confirmed HERC2/OCA2 genotype can support a pigmentation prediction rather than a HERC2 protein-disease prediction. "
            "For rs12913832, genotype dosage matters: A/A, A/G, and G/G should be interpreted separately. The forward-strand G allele used by this workbench supports a lighter or blue-eye tendency because it weakens HERC2 enhancer looping to the OCA2 promoter, lowers OCA2 expression, and reduces iris melanin, while A is the darker-eye-compatible contrast state. "
            "Linked HERC2 markers should be read as probabilistic light-versus-dark eye-colour haplotype evidence, with ancestry and additional OCA2/pigmentation variants able to modify the visible result."
        ),
        "variants": [
            {
                "variant": "rs12913832",
                "display_name": "rs12913832",
                "common_name": "HERC2/OCA2 enhancer rs12913832",
                "position": 28365618,
                "lookup_keys": [
                    "rs12913832",
                    "HERC2:rs12913832",
                    "15:28365618",
                    "15:28365618:A>G",
                    "15:28365618:G>A",
                ],
                "region_class": "intronic_enhancer",
                "interpretation_scope": "Regulatory pigmentation marker",
                "clinical_interpretation": (
                    "rs12913832 lies in a conserved HERC2 intronic enhancer that regulates OCA2 expression. "
                    "In the app's forward-strand A/G representation, G dosage supports a lighter or blue-eye tendency and A dosage supports a darker-eye-compatible tendency, with heterozygotes kept explicitly probabilistic."
                ),
                "clinical_significance": "Research marker for iris pigmentation biology; not a disease-pathogenicity call.",
                "functional_effects": [
                    "Changes enhancer support for HERC2-OCA2 chromatin looping and OCA2 promoter activity.",
                    "Lower OCA2 expression reduces iris melanin biology and supports lighter-eye pigmentation in appropriate ancestry context.",
                ],
                "associated_conditions": [
                    "Blue-versus-brown iris pigmentation",
                    "Forensic eye-colour prediction",
                    "OCA2 expression and melanocyte melanin biology",
                ],
                "research_context": [
                    "Use decoded GT dosage rather than ALT presence alone.",
                    "Interpret with ancestry, eye-colour phenotype, and additional OCA2/pigmentation markers.",
                ],
                "usual_variant_note": "Best-known HERC2/OCA2 eye-colour enhancer marker.",
                "methylation_interpretation": (
                    "Pair rs12913832 with HERC2/OCA2 methylation as regulatory context only; the main prediction is genotype-dosage driven."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 18252222: rs12913832 predicts blue-brown eye colour", "https://pubmed.ncbi.nlm.nih.gov/18252222/"),
                    _evidence("PubMed 22234890: enhancer looping and OCA2 expression", "https://pubmed.ncbi.nlm.nih.gov/22234890/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Sturm et al., 2008 (PMID 18252222)",
                        "genotypes": "HERC2 rs12913832 genotype classes",
                        "phenotype": "Blue-brown eye-colour prediction",
                        "finding": "The study identified rs12913832 as the strongest tested HERC2/OCA2 predictor of blue-brown eye colour in European-ancestry cohorts.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/18252222/",
                    },
                    {
                        "paper": "Visser et al., 2012 (PMID 22234890)",
                        "genotypes": "rs12913832 enhancer alleles",
                        "phenotype": "OCA2 expression and melanocyte pigmentation",
                        "finding": "The functional study showed allele-specific enhancer looping to the OCA2 promoter and reduced OCA2 expression in lightly pigmented melanocytes.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/22234890/",
                    },
                ],
            },
            {
                "variant": "rs1129038",
                "display_name": "rs1129038",
                "common_name": "HERC2/OCA2 linked pigmentation haplotype marker",
                "position": 28356859,
                "lookup_keys": [
                    "rs1129038",
                    "HERC2:rs1129038",
                    "15:28356859",
                    "15:28356859:C>T",
                    "15:28356859:T>C",
                ],
                "region_class": "intronic_regulatory",
                "interpretation_scope": "Linked pigmentation haplotype marker",
                "clinical_interpretation": (
                    "rs1129038 is strongly linked to the HERC2/OCA2 eye-colour haplotype and is useful as secondary support for the rs12913832 pigmentation thesis. "
                    "The T allele is treated as the lighter-eye-supporting state in the local genotype-dosage model."
                ),
                "clinical_significance": "Research marker for iris pigmentation biology; secondary to rs12913832.",
                "functional_effects": [
                    "Tags the HERC2/OCA2 regulatory haplotype used in eye-colour prediction models.",
                    "Adds secondary genotype-dosage support for light-versus-dark pigmentation inference.",
                ],
                "associated_conditions": [
                    "Blue-versus-brown iris pigmentation",
                    "HERC2/OCA2 haplotype prediction",
                ],
                "research_context": [
                    "Use as a supporting marker, especially when rs12913832 is present or ambiguous.",
                ],
                "usual_variant_note": "Linked HERC2/OCA2 pigmentation marker.",
                "methylation_interpretation": "Use HERC2 methylation as local regulatory context, not as proof of rs1129038 functional direction.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 18252222: HERC2/OCA2 haplotypes and eye colour", "https://pubmed.ncbi.nlm.nih.gov/18252222/"),
                    _evidence("PubMed 20457063: HERC2 eye-colour marker panel", "https://pubmed.ncbi.nlm.nih.gov/20457063/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Sturm et al., 2008 (PMID 18252222)",
                        "genotypes": "rs1129038-rs12913832 haplotypes",
                        "phenotype": "Blue and nonblue eye colour",
                        "finding": "The paper reported strong eye-colour separation for linked HERC2/OCA2 haplotypes containing rs1129038 and rs12913832.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/18252222/",
                    }
                ],
            },
            {
                "variant": "rs7170852",
                "display_name": "rs7170852",
                "common_name": "Secondary HERC2 eye-colour marker",
                "position": 28427986,
                "lookup_keys": [
                    "rs7170852",
                    "HERC2:rs7170852",
                    "15:28427986",
                    "15:28427986:T>A",
                    "15:28427986:A>T",
                ],
                "region_class": "intronic_regulatory",
                "interpretation_scope": "Secondary pigmentation marker",
                "clinical_interpretation": (
                    "rs7170852 is included as a secondary HERC2 marker reported in eye-colour association panels. "
                    "The A allele is treated as additional lighter-eye-supporting evidence in the local three-marker HERC2 dosage model."
                ),
                "clinical_significance": "Research marker for iris pigmentation biology; secondary and cohort dependent.",
                "functional_effects": [
                    "Adds supporting genotype context for HERC2/OCA2 pigmentation haplotypes.",
                    "Should not override rs12913832 because its independent effect is weaker and cohort dependent.",
                ],
                "associated_conditions": [
                    "Iris pigmentation",
                    "Forensic eye-colour prediction panels",
                ],
                "research_context": [
                    "Interpret alongside rs12913832 and rs1129038 rather than as a single-marker prediction.",
                ],
                "usual_variant_note": "Secondary HERC2 pigmentation marker.",
                "methylation_interpretation": "Use HERC2 methylation as broad regulatory context for the pigmentation interval.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 20457063: rs7170852 and HERC2/OCA2 eye-colour panel", "https://pubmed.ncbi.nlm.nih.gov/20457063/"),
                    _evidence("PubMed 22065085: global OCA2-HERC2 pigmentation view", "https://pubmed.ncbi.nlm.nih.gov/22065085/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Mengel-From et al., 2010 (PMID 20457063)",
                        "genotypes": "HERC2 rs7170852 sequence variation",
                        "phenotype": "Human eye colour",
                        "finding": "The study reported additional eye-colour association for rs7170852 beyond the major HERC2/OCA2 marker context.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/20457063/",
                    }
                ],
            },
            {
                "variant": "rs916977",
                "display_name": "rs916977",
                "common_name": "HERC2 iris-colour cline marker",
                "position": 28513364,
                "lookup_keys": [
                    "rs916977",
                    "HERC2:rs916977",
                    "15:28513364",
                    "15:28513364:T>C",
                    "15:28513364:C>T",
                ],
                "region_class": "intronic_regulatory",
                "interpretation_scope": "Population-stratified iris-colour marker",
                "clinical_interpretation": (
                    "rs916977 is a HERC2 iris-colour marker reported with a clinal European allele distribution. "
                    "It is useful as secondary population-aware pigmentation context but should not be treated as the main enhancer mechanism when rs12913832 is available."
                ),
                "clinical_significance": "Research marker for iris pigmentation and population-pattern review.",
                "functional_effects": [
                    "Marks a HERC2 haplotype correlated with iris-colour variation across European populations.",
                    "Provides ancestry-aware supporting evidence for light-versus-dark eye-colour models.",
                ],
                "associated_conditions": [
                    "Iris pigmentation",
                    "Population structure in eye-colour prediction",
                    "Pigmentation-linked skin-cancer context",
                ],
                "research_context": [
                    "Use with ancestry context because rs916977 has reported European geographic clines.",
                ],
                "usual_variant_note": "HERC2 iris-colour cline marker.",
                "methylation_interpretation": "Use methylation around HERC2 as a regulatory-context layer, not as a direct rs916977 effect readout.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 18252221: HERC2 rs916977 and iris-colour GWAS", "https://pubmed.ncbi.nlm.nih.gov/18252221/"),
                    _evidence("PubMed 28878084: OCA2/HERC2 variants and cSCC timing", "https://pubmed.ncbi.nlm.nih.gov/28878084/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Kayser et al., 2008 (PMID 18252221)",
                        "genotypes": "HERC2 rs916977 genotypes",
                        "phenotype": "Human iris colour across European cohorts",
                        "finding": "The study reported rs916977 as a major HERC2 iris-colour marker with a clinal allele distribution across European populations.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/18252221/",
                    }
                ],
            },
            {
                "variant": "rs11636232",
                "display_name": "rs11636232",
                "common_name": "Additional HERC2 eye-colour panel marker",
                "position": 28386626,
                "lookup_keys": [
                    "rs11636232",
                    "HERC2:rs11636232",
                    "15:28386626",
                    "15:28386626:C>T",
                    "15:28386626:T>C",
                ],
                "region_class": "intronic_or_synonymous_marker",
                "interpretation_scope": "Secondary eye-colour panel marker",
                "clinical_interpretation": (
                    "rs11636232 is included as a secondary HERC2/OCA2 eye-colour panel marker. "
                    "It can sharpen multilocus pigmentation models but remains subordinate to rs12913832 genotype dosage."
                ),
                "clinical_significance": "Research marker for iris pigmentation biology; secondary and cohort dependent.",
                "functional_effects": [
                    "Adds marker-panel context for HERC2/OCA2 eye-colour discrimination.",
                    "May improve light-versus-dark eye-colour separation in selected multilocus models.",
                ],
                "associated_conditions": [
                    "Iris pigmentation",
                    "Forensic eye-colour prediction panels",
                ],
                "research_context": [
                    "Interpret only with exact genotype, ancestry, and the stronger HERC2/OCA2 markers.",
                ],
                "usual_variant_note": "Additional HERC2/OCA2 eye-colour marker.",
                "methylation_interpretation": "Use HERC2 methylation as broad locus context rather than a variant-specific effect readout.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 20457063: rs11636232 and HERC2/OCA2 eye-colour panel", "https://pubmed.ncbi.nlm.nih.gov/20457063/"),
                    _evidence("PMCID PMC10454093: forensic eye-colour marker review", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10454093/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Mengel-From et al., 2010 (PMID 20457063)",
                        "genotypes": "HERC2 rs11636232 sequence variation",
                        "phenotype": "Human eye colour",
                        "finding": "The study included rs11636232 as an additional HERC2 marker associated with eye-colour variation beyond the strongest HERC2/OCA2 site.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/20457063/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from HERC2/OCA2 pigmentation, iris-colour, and forensic DNA phenotyping literature.",
        "population_coverage_note": (
            "The bundled HERC2 population database provides compact 1000 Genomes-style superpopulation frequency context for selected pigmentation markers. "
            "It is meant for ancestry-aware interpretation, not for deterministic eye-colour prediction."
        ),
        "population_sources": [
            _evidence("1000 Genomes Project phase 3 reference context", "https://www.internationalgenome.org/category/phase-3/"),
            _evidence("PubMed 18252222: rs12913832 and HERC2/OCA2 haplotypes", "https://pubmed.ncbi.nlm.nih.gov/18252222/"),
            _evidence("PubMed 18252221: rs916977 European cline", "https://pubmed.ncbi.nlm.nih.gov/18252221/"),
            _evidence("PubMed 22065085: global OCA2-HERC2 pigmentation view", "https://pubmed.ncbi.nlm.nih.gov/22065085/"),
        ],
        "variant_population_records": [
            {
                "variant": "rs12913832",
                "display_name": "rs12913832",
                "common_name": "HERC2/OCA2 enhancer rs12913832",
                "effect_allele": "G",
                "focus_alleles": ["A", "G"],
                "effect_summary": "G is the lighter-eye-associated allele in the app's forward-strand A/G representation.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/18252222/",
                "top_level_location_frequencies": [
                    {"population_code": "AFR", "location_group": "African ancestry", "label": "African ancestry", "granularity": "superpopulation", "allele_frequencies": {"A": 0.96, "G": 0.04}},
                    {"population_code": "AMR", "location_group": "Admixed American ancestry", "label": "Admixed American ancestry", "granularity": "superpopulation", "allele_frequencies": {"A": 0.55, "G": 0.45}},
                    {"population_code": "EAS", "location_group": "East Asian ancestry", "label": "East Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"A": 0.99, "G": 0.01}},
                    {"population_code": "EUR", "location_group": "European ancestry", "label": "European ancestry", "granularity": "superpopulation", "allele_frequencies": {"A": 0.30, "G": 0.70}},
                    {"population_code": "SAS", "location_group": "South Asian ancestry", "label": "South Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"A": 0.73, "G": 0.27}},
                ],
                "detailed_population_frequencies": [],
            },
            {
                "variant": "rs1129038",
                "display_name": "rs1129038",
                "common_name": "HERC2/OCA2 linked pigmentation haplotype marker",
                "effect_allele": "T",
                "focus_alleles": ["C", "T"],
                "effect_summary": "T tracks the lighter-eye-associated HERC2/OCA2 haplotype used as secondary support for rs12913832.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/18252222/",
                "top_level_location_frequencies": [
                    {"population_code": "AFR", "location_group": "African ancestry", "label": "African ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.95, "T": 0.05}},
                    {"population_code": "AMR", "location_group": "Admixed American ancestry", "label": "Admixed American ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.56, "T": 0.44}},
                    {"population_code": "EAS", "location_group": "East Asian ancestry", "label": "East Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.99, "T": 0.01}},
                    {"population_code": "EUR", "location_group": "European ancestry", "label": "European ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.31, "T": 0.69}},
                    {"population_code": "SAS", "location_group": "South Asian ancestry", "label": "South Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.74, "T": 0.26}},
                ],
                "detailed_population_frequencies": [],
            },
            {
                "variant": "rs7170852",
                "display_name": "rs7170852",
                "common_name": "Secondary HERC2 eye-colour marker",
                "effect_allele": "A",
                "focus_alleles": ["T", "A"],
                "effect_summary": "A is modeled as a secondary lighter-eye-supporting allele in the HERC2 marker panel.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/20457063/",
                "top_level_location_frequencies": [
                    {"population_code": "AFR", "location_group": "African ancestry", "label": "African ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.94, "A": 0.06}},
                    {"population_code": "AMR", "location_group": "Admixed American ancestry", "label": "Admixed American ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.58, "A": 0.42}},
                    {"population_code": "EAS", "location_group": "East Asian ancestry", "label": "East Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.99, "A": 0.01}},
                    {"population_code": "EUR", "location_group": "European ancestry", "label": "European ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.34, "A": 0.66}},
                    {"population_code": "SAS", "location_group": "South Asian ancestry", "label": "South Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.76, "A": 0.24}},
                ],
                "detailed_population_frequencies": [],
            },
            {
                "variant": "rs916977",
                "display_name": "rs916977",
                "common_name": "HERC2 iris-colour cline marker",
                "effect_allele": "C",
                "focus_alleles": ["T", "C"],
                "effect_summary": "C is treated as the blue-eye-associated allele for the European cline marker rs916977.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/18252221/",
                "top_level_location_frequencies": [
                    {"population_code": "AFR", "location_group": "African ancestry", "label": "African ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.97, "C": 0.03}},
                    {"population_code": "AMR", "location_group": "Admixed American ancestry", "label": "Admixed American ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.60, "C": 0.40}},
                    {"population_code": "EAS", "location_group": "East Asian ancestry", "label": "East Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.98, "C": 0.02}},
                    {"population_code": "EUR", "location_group": "European ancestry", "label": "European ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.29, "C": 0.71}},
                    {"population_code": "SAS", "location_group": "South Asian ancestry", "label": "South Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"T": 0.78, "C": 0.22}},
                ],
                "detailed_population_frequencies": [],
            },
            {
                "variant": "rs11636232",
                "display_name": "rs11636232",
                "common_name": "Additional HERC2 eye-colour panel marker",
                "effect_allele": "T",
                "focus_alleles": ["C", "T"],
                "effect_summary": "T is modeled as an additional lighter-eye-supporting marker in selected HERC2/OCA2 panels.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/20457063/",
                "top_level_location_frequencies": [
                    {"population_code": "AFR", "location_group": "African ancestry", "label": "African ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.93, "T": 0.07}},
                    {"population_code": "AMR", "location_group": "Admixed American ancestry", "label": "Admixed American ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.70, "T": 0.30}},
                    {"population_code": "EAS", "location_group": "East Asian ancestry", "label": "East Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.95, "T": 0.05}},
                    {"population_code": "EUR", "location_group": "European ancestry", "label": "European ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.52, "T": 0.48}},
                    {"population_code": "SAS", "location_group": "South Asian ancestry", "label": "South Asian ancestry", "granularity": "superpopulation", "allele_frequencies": {"C": 0.80, "T": 0.20}},
                ],
                "detailed_population_frequencies": [],
            },
        ],
        "gene_population_patterns": [
            {
                "variant": "HERC2/OCA2 rs12913832 enhancer haplotype",
                "location_group": "Global pattern",
                "summary": "The lighter-eye-associated HERC2/OCA2 haplotype is most frequent in European-ancestry reference panels and uncommon in many African and East Asian panels, so ancestry context strongly affects interpretation.",
            },
            {
                "variant": "HERC2 rs916977 cline",
                "location_group": "European ancestry",
                "summary": "rs916977 was reported with a European clinal allele distribution correlated with iris-colour variation, supporting population-aware use in forensic eye-colour prediction.",
            },
            {
                "variant": "Secondary HERC2/OCA2 panel markers",
                "location_group": "Marker and cohort pattern",
                "summary": "rs1129038, rs7170852, and rs11636232 can sharpen multilocus pigmentation models, but they should refine rather than override the GT-decoded rs12913832 result.",
            },
        ],
    },
    {
        "gene_name": "SIRT6",
        "cytoband": "19p13.3",
        "chromosome": "19",
        "start": 4174106,
        "end": 4182560,
        "strand": "-",
        "gene_summary": (
            "SIRT6 encodes a nuclear NAD-dependent deacylase and mono-ADP-ribosyltransferase that supports DNA repair, telomere maintenance, transposon suppression, inflammation control, and energy homeostasis."
        ),
        "clinical_context": (
            "The bundled SIRT6 knowledge base is centered on aging and genome-maintenance biology. Most currently relevant human variants are interpreted as research-grade longevity or regulatory signals rather than as routine pathogenic clinical alleles."
        ),
        "variant_effect_overview": [
            "SIRT6 is a core longevity and genome-stability gene, so common and rare human variants are mainly interpreted through aging and cellular-maintenance phenotypes.",
            "Many effects are subtle or cohort specific; the strongest current evidence pairs genetic variation with mechanistic functional follow-up.",
        ],
        "condition_research_overview": [
            "SIRT6 is heavily studied in aging, genome stability, inflammation, lipid and glucose metabolism, and cardiovascular disease.",
            "Rare centenarian-enriched alleles and selected common SNPs are being used to connect human genetics with SIRT6 functional biochemistry.",
        ],
        "methylation_interpretation": (
            "SIRT6 methylation should be interpreted as chromatin and transcriptional context around an NAD-dependent longevity enzyme. "
            "Because SIRT6 function is closely tied to chromatin regulation, methylation can be informative, but not as a one-to-one biomarker."
        ),
        "methylation_effects": [
            "Promoter methylation may reflect altered SIRT6 transcriptional accessibility and should be integrated with broader NAD and stress-response biology.",
            "Functional inference still depends more on enzyme activity and downstream chromatin effects than on methylation alone.",
        ],
        "methylation_condition_research": [
            "Aging and longevity research.",
            "Genome-stability and DNA-repair studies.",
            "Cardiometabolic and inflammatory disease biology.",
        ],
        "evidence": [
            _evidence("NCBI Gene 51548: SIRT6 gene summary", "https://www.ncbi.nlm.nih.gov/gene/51548"),
            _evidence("PubMed 28032059: rs350846 and human longevity", "https://pubmed.ncbi.nlm.nih.gov/28032059/"),
            _evidence("PubMed 36215696: centenarian SIRT6 variant", "https://pubmed.ncbi.nlm.nih.gov/36215696/"),
            _evidence("PubMed 36465178: SIRT6 review in aging and metabolism", "https://pubmed.ncbi.nlm.nih.gov/36465178/"),
        ],
        "variants": [
            {
                "variant": "rs350846",
                "common_name": "longevity-associated SIRT6 SNP",
                "region_class": "gene_body",
                "interpretation_scope": "Research association / longevity marker",
                "clinical_interpretation": (
                    "rs350846 is a common SIRT6 polymorphism curated here as a research-grade human longevity marker. "
                    "The local database does not treat it as a diagnostic allele, but as a cohort-dependent aging association."
                ),
                "clinical_significance": "Research-level longevity association.",
                "functional_effects": [
                    "Reported as a human longevity-associated marker at the SIRT6 locus.",
                    "Likely reflects pathway-level modulation of SIRT6 expression or linked regulatory state rather than a deterministic coding lesion.",
                ],
                "associated_conditions": [
                    "Human longevity and healthy aging",
                    "Genome-maintenance and metabolic-aging biology",
                ],
                "research_context": [
                    "This SNP is best interpreted through cohort and population context rather than as a stand-alone predictive marker.",
                ],
                "usual_variant_note": "Most cited common SIRT6 longevity SNP.",
                "methylation_interpretation": (
                    "SIRT6 methylation provides broader chromatin context and should not be treated as a direct biomarker of rs350846 status."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 28032059: rs350846 and human longevity", "https://pubmed.ncbi.nlm.nih.gov/28032059/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Li et al., 2016 (PMID 28032059)",
                        "genotypes": "CC, CG, and GG",
                        "phenotype": "Human longevity in long-lived versus control cohorts",
                        "finding": "The study identified rs350846 as a SIRT6 locus polymorphism associated with human longevity, supporting its use as a research-grade aging marker.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/28032059/",
                    }
                ],
            },
            {
                "variant": "centSIRT6 (N308K/A313S)",
                "display_name": "centSIRT6 (N308K/A313S)",
                "common_name": "centenarian-enriched SIRT6 linked missense allele",
                "region_class": "gene_body",
                "interpretation_scope": "Rare functional longevity variant",
                "clinical_interpretation": (
                    "The linked SIRT6 missense substitutions N308K and A313S define a rare centenarian-enriched allele with enhanced genome-maintenance functions in experimental follow-up. "
                    "It is curated as a rare functional longevity variant, not as a routine clinical diagnostic allele."
                ),
                "clinical_significance": "Rare functional longevity-associated research allele.",
                "functional_effects": [
                    "Enhanced mono-ADP-ribosylase activity and stronger interaction with Lamin A/C were reported in functional studies.",
                    "The allele improved LINE1 suppression and DNA double-strand break repair in experimental systems.",
                ],
                "associated_conditions": [
                    "Exceptional longevity",
                    "Genome stability and stress-resistance biology",
                ],
                "research_context": [
                    "This is a rare mechanistic allele of high interest because it links centenarian enrichment to direct SIRT6 functional changes.",
                ],
                "usual_variant_note": "Rare centenarian SIRT6 functional allele.",
                "methylation_interpretation": (
                    "Methylation provides broad chromatin context for SIRT6 but does not directly substitute for sequencing or functional assessment of the centSIRT6 allele."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PubMed 36215696: centenarian SIRT6 variant", "https://pubmed.ncbi.nlm.nih.gov/36215696/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Simon et al., 2022 (PMID 36215696)",
                        "genotypes": "Rare linked missense carrier state",
                        "phenotype": "Centenarian enrichment and genome-stability phenotypes",
                        "finding": "The centSIRT6 allele was enriched in Ashkenazi Jewish centenarians and displayed enhanced DNA repair, LINE1 suppression, and cancer-cell killing in functional assays.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/36215696/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from SIRT6 longevity, genome-stability, and metabolic-aging literature.",
        "population_coverage_note": (
            "The bundled SIRT6 population database emphasizes longevity cohorts and rare centenarian alleles instead of a full general-population allele-frequency reference panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 51548: SIRT6 gene summary", "https://www.ncbi.nlm.nih.gov/gene/51548"),
            _evidence("PubMed 28032059: rs350846 and longevity", "https://pubmed.ncbi.nlm.nih.gov/28032059/"),
            _evidence("PubMed 36215696: centenarian SIRT6 variant", "https://pubmed.ncbi.nlm.nih.gov/36215696/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs350846",
                "location_group": "Longevity cohorts",
                "summary": "Common SIRT6 longevity signals such as rs350846 are cohort dependent and are best interpreted as probabilistic aging markers rather than universally penetrant alleles.",
            },
            {
                "variant": "centSIRT6 (N308K/A313S)",
                "location_group": "Rare disease families",
                "summary": "The most compelling current SIRT6 human-genetics result is a rare centenarian-enriched functional allele, underscoring that the locus is shaped by both common regulatory and rare mechanistic variation.",
            },
        ],
    },
    {
        "gene_name": "PRKAA1",
        "cytoband": "5p13.1",
        "chromosome": "5",
        "start": 40759491,
        "end": 40798476,
        "strand": "-",
        "gene_summary": (
            "PRKAA1 encodes the alpha-1 catalytic subunit of AMPK, the major cellular energy sensor that restrains anabolic signaling, supports metabolic adaptation, and suppresses mTORC1 under low-energy conditions."
        ),
        "clinical_context": (
            "The local PRKAA1 knowledge base is built around metabolic and cancer-association literature. "
            "Common PRKAA1 variants are best interpreted as pathway modifiers rather than as highly penetrant clinical alleles."
        ),
        "variant_effect_overview": [
            "Common PRKAA1 variants are typically interpreted as low-penetrance metabolic or cancer-risk modifiers.",
            "Because PRKAA1 acts inside the AMPK-mTOR energy-stress axis, published associations often reflect pathway state and environmental context.",
        ],
        "condition_research_overview": [
            "PRKAA1 is central to AMPK signaling, nutrient sensing, mitochondrial stress responses, and mTOR restraint.",
            "The strongest common-variant literature centers on gastric-cancer susceptibility and metabolic-endocrine phenotypes.",
        ],
        "methylation_interpretation": (
            "PRKAA1 methylation should be read as energy-sensing pathway context. Promoter-proximal methylation may suggest altered catalytic-subunit expression, "
            "but interpretation remains broader than any one common SNP."
        ),
        "methylation_effects": [
            "Promoter methylation may help contextualize PRKAA1 transcriptional tone in metabolism-focused tissues.",
            "Integrate methylation with AMPK-mTOR signaling, energy stress, and nutritional context.",
        ],
        "methylation_condition_research": [
            "Energy-stress and nutrient-sensing biology.",
            "Cancer, especially gastric-cancer association work.",
            "Metabolic and endocrine signaling research.",
        ],
        "evidence": [
            _evidence("NCBI Gene 5562: PRKAA1 gene summary", "https://www.ncbi.nlm.nih.gov/gene/5562"),
            _evidence("PubMed 26485766: PRKAA1 rs13361707 and gastric-cancer risk", "https://pubmed.ncbi.nlm.nih.gov/26485766/"),
        ],
        "variants": [
            {
                "variant": "rs13361707",
                "common_name": "gastric-cancer-associated PRKAA1 SNP",
                "region_class": "gene_body",
                "interpretation_scope": "Research association / low-penetrance risk marker",
                "clinical_interpretation": (
                    "rs13361707 is the best-established common PRKAA1 marker in gastric-cancer association literature. "
                    "The local database treats it as a low-penetrance pathway-associated risk marker rather than as a pathogenic clinical allele."
                ),
                "clinical_significance": "Research-level low-penetrance association marker.",
                "functional_effects": [
                    "Likely acts through regulatory or linked-locus effects on AMPK-pathway signaling rather than as a direct protein-disrupting lesion.",
                    "Effect sizes reported in cancer cohorts are modest and ancestry specific.",
                ],
                "associated_conditions": [
                    "Gastric-cancer susceptibility",
                    "Metabolic and stress-signaling pathway context",
                ],
                "research_context": [
                    "This is the canonical common PRKAA1 association SNP in East Asian gastric-cancer studies.",
                ],
                "usual_variant_note": "Most cited common PRKAA1 association SNP.",
                "methylation_interpretation": (
                    "PRKAA1 methylation should be read as broader energy-sensing locus context rather than as a direct biomarker of rs13361707."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 26485766: PRKAA1 rs13361707 and gastric-cancer risk", "https://pubmed.ncbi.nlm.nih.gov/26485766/"),
                    _evidence("PubMed 37670284: PRKAA1 polymorphisms and gastric cancer", "https://pubmed.ncbi.nlm.nih.gov/37670284/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Chen et al., 2015 (PMID 26485766)",
                        "genotypes": "TT, CT, and CC",
                        "phenotype": "Gastric-cancer susceptibility in an eastern Chinese population",
                        "finding": "The study reported that the rs13361707 C allele increased gastric-cancer risk, consistent with its use as a low-penetrance PRKAA1 research marker.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/26485766/",
                    }
                ],
            }
        ],
        "population_intro": "Broader population patterns curated from PRKAA1 gastric-cancer, metabolic, and AMPK-pathway literature.",
        "population_coverage_note": (
            "This PRKAA1 population database emphasizes cohort-level association patterns instead of a full embedded allele-frequency panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 5562: PRKAA1 gene summary", "https://www.ncbi.nlm.nih.gov/gene/5562"),
            _evidence("PubMed 26485766: PRKAA1 rs13361707 and gastric-cancer risk", "https://pubmed.ncbi.nlm.nih.gov/26485766/"),
            _evidence("PubMed 37670284: PRKAA1 polymorphism follow-up", "https://pubmed.ncbi.nlm.nih.gov/37670284/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs13361707",
                "location_group": "Cancer cohorts",
                "summary": "PRKAA1 common-variant literature is strongest in East Asian gastric-cancer cohorts, where rs13361707 is interpreted as a low-penetrance risk marker at the AMPK pathway locus.",
            },
            {
                "variant": "PRKAA1 pathway context",
                "location_group": "Metabolic cohorts",
                "summary": "Across populations, PRKAA1 interpretation benefits from energy-stress and nutrient-signaling context rather than from assuming a uniform common-SNP effect across phenotypes.",
            },
        ],
    },
    {
        "gene_name": "NAMPT",
        "cytoband": "7q22.3",
        "chromosome": "7",
        "start": 105888744,
        "end": 105925429,
        "strand": "-",
        "gene_summary": (
            "NAMPT encodes nicotinamide phosphoribosyltransferase, the rate-limiting enzyme in the major NAD salvage pathway and a key regulator of metabolic stress, inflammation, and cellular resilience."
        ),
        "clinical_context": (
            "The bundled NAMPT database is focused on metabolism, inflammation, and disease-association research. "
            "Most current human variant interpretation at this locus is research grade and cohort dependent."
        ),
        "variant_effect_overview": [
            "Common NAMPT variants are usually interpreted as regulatory or biomarker-linked modifiers rather than as high-penetrance alleles.",
            "Because NAMPT biology spans NAD metabolism, inflammation, and secreted eNAMPT signaling, effect sizes and directions can vary substantially by phenotype.",
        ],
        "condition_research_overview": [
            "NAMPT is studied in metabolism, obesity, inflammation, cardiovascular disease, cancer, and aging.",
            "The gene also links NAD salvage to sirtuin biology, stress responses, and systemic inflammatory signaling.",
        ],
        "methylation_interpretation": (
            "NAMPT methylation is best interpreted as regulatory context around NAD-salvage capacity and inflammatory-metabolic signaling. "
            "It should be integrated with broader pathway state rather than read as a direct biomarker for any one common SNP."
        ),
        "methylation_effects": [
            "Promoter methylation may help contextualize NAMPT transcriptional potential and systemic metabolic stress responses.",
            "Most phenotype interpretation still depends more on pathway activity, circulating NAMPT, and cohort design than on methylation alone.",
        ],
        "methylation_condition_research": [
            "NAD metabolism and healthy-aging biology.",
            "Inflammation and cardiovascular-risk studies.",
            "Cancer and systemic stress-response research.",
        ],
        "evidence": [
            _evidence("NCBI Gene 10135: NAMPT gene summary", "https://www.ncbi.nlm.nih.gov/gene/10135"),
            _evidence("PubMed 25896907: NAMPT polymorphisms in esophageal carcinoma", "https://pubmed.ncbi.nlm.nih.gov/25896907/"),
            _evidence("PubMed 33446046: NAMPT rs61330082 and cardiovascular risk after HCV clearance", "https://pubmed.ncbi.nlm.nih.gov/33446046/"),
        ],
        "variants": [
            {
                "variant": "rs61330082",
                "common_name": "regulatory NAMPT SNP",
                "region_class": "promoter",
                "interpretation_scope": "Research association / metabolic-inflammatory marker",
                "clinical_interpretation": (
                    "rs61330082 is a promoter-facing NAMPT variant studied in cancer, lipid, and cardiovascular-risk cohorts. "
                    "The local database treats it as a research association marker for inflammatory-metabolic signaling rather than as a pathogenic clinical allele."
                ),
                "clinical_significance": "Research-level metabolic and inflammatory association marker.",
                "functional_effects": [
                    "Likely regulatory or linked-locus effects on NAMPT expression and pathway tone.",
                    "Phenotypic interpretation varies by inflammatory and metabolic context.",
                ],
                "associated_conditions": [
                    "Esophageal squamous-cell carcinoma susceptibility",
                    "Cardiovascular-event risk after chronic viral disease treatment",
                    "Lipid and metabolic biomarker studies",
                ],
                "research_context": [
                    "This SNP is one of the better characterized common NAMPT regulatory markers in disease-association literature.",
                ],
                "usual_variant_note": "Most cited common regulatory NAMPT SNP.",
                "methylation_interpretation": (
                    "Treat NAMPT methylation as broader regulatory context and not as a direct surrogate for rs61330082 genotype."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 25896907: NAMPT polymorphisms in esophageal carcinoma", "https://pubmed.ncbi.nlm.nih.gov/25896907/"),
                    _evidence("PubMed 33446046: NAMPT rs61330082 and cardiovascular risk", "https://pubmed.ncbi.nlm.nih.gov/33446046/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Chen et al., 2015 (PMID 25896907)",
                        "genotypes": "CC, CT, and TT",
                        "phenotype": "Esophageal squamous-cell carcinoma susceptibility",
                        "finding": "The study examined rs61330082 among the main common NAMPT polymorphisms contributing to esophageal-squamous-cell-carcinoma risk modeling.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/25896907/",
                    },
                    {
                        "paper": "Huang et al., 2021 (PMID 33446046)",
                        "genotypes": "TT versus non-TT",
                        "phenotype": "Long-term cardiovascular events after HCV clearance",
                        "finding": "The rs61330082 TT genotype was associated with a higher cumulative incidence of cardiovascular events in the treated cohort, reinforcing its value as a context-dependent metabolic-inflammatory marker.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/33446046/",
                    },
                ],
            }
        ],
        "population_intro": "Broader population patterns curated from NAMPT inflammatory, cardiovascular, cancer, and NAD-metabolism literature.",
        "population_coverage_note": (
            "This NAMPT population database emphasizes cohort-specific metabolic and inflammatory patterns instead of a full embedded allele-frequency panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 10135: NAMPT gene summary", "https://www.ncbi.nlm.nih.gov/gene/10135"),
            _evidence("PubMed 25896907: NAMPT polymorphism study", "https://pubmed.ncbi.nlm.nih.gov/25896907/"),
            _evidence("PubMed 33446046: NAMPT rs61330082 cardiovascular study", "https://pubmed.ncbi.nlm.nih.gov/33446046/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs61330082",
                "location_group": "Metabolic cohorts",
                "summary": "NAMPT common-variant interpretation is highly phenotype specific, spanning lipid traits, inflammatory tone, and long-term cardiovascular outcomes rather than a single stable population pattern.",
            },
            {
                "variant": "NAMPT pathway context",
                "location_group": "Global pattern",
                "summary": "Because NAMPT sits at the center of NAD salvage and stress signaling, cross-population interpretation should emphasize disease context and metabolic state over blanket assumptions about one common allele.",
            },
        ],
    },
    {
        "gene_name": "CDKN2A",
        "cytoband": "9p21.3",
        "chromosome": "9",
        "start": 21967751,
        "end": 21995323,
        "strand": "-",
        "coordinate_source": "NCBI Gene 1029 GRCh37 / hg19 coordinates, with ClinVar GRCh37 variant loci for seeded markers",
        "gene_summary": (
            "CDKN2A is a 9p21.3 protein-coding tumor-suppressor gene that encodes multiple products, including p16INK4A and p14ARF. "
            "These products regulate CDK4/6-RB and MDM2-p53 control points, G1 cell-cycle arrest, senescence, and oncogenic-stress response."
        ),
        "clinical_context": (
            "The bundled CDKN2A knowledge base is tumor-suppressor focused. Clinically important CDKN2A events include heterozygous germline pathogenic variants, deletions, or promoter methylation changes; common 3'UTR and missense polymorphisms are kept as research context unless external clinical curation supports a higher-impact call."
        ),
        "variant_effect_overview": [
            "For CDKN2A, rare germline and somatic loss-of-function events usually carry more weight than common SNPs.",
            "Pathogenic p16INK4A or p14ARF-disrupting variants can affect melanoma, pancreatic cancer, and melanoma-neural-system tumor predisposition biology.",
            "Promoter methylation and deletion are central recurring mechanisms in cancer biology and senescence research at this locus.",
            "Observed PASS variants in the interval should therefore be treated as locus context unless they match a seeded research SNP or have outside clinical curation.",
        ],
        "condition_research_overview": [
            "CDKN2A is a major tumor-suppressor locus in melanoma, pancreatic cancer predisposition, glioma, and many additional tumors.",
            "The locus is also central to aging and senescence research because p16INK4A expression is a canonical cellular-senescence marker.",
            "Interpretation should distinguish common benign or low-effect polymorphisms from rare pathogenic variants such as p.Gly101Trp.",
        ],
        "methylation_interpretation": (
            "CDKN2A methylation is often biologically meaningful because promoter hypermethylation is a recurrent mode of tumor-suppressor silencing. "
            "In this workbench it should still be read as regulatory context rather than as a stand-alone diagnosis."
        ),
        "methylation_effects": [
            "Promoter hypermethylation can be consistent with reduced p16INK4A pathway accessibility and is widely studied in cancer cohorts.",
            "At the same time, CDKN2A expression itself is a senescence biomarker, so methylation should be integrated with tissue and disease context.",
        ],
        "methylation_condition_research": [
            "Cancer-associated promoter methylation studies.",
            "Aging and cellular-senescence research.",
            "Tumor-suppressor silencing and biomarker development.",
        ],
        "evidence": [
            _evidence("NCBI Gene 1029: CDKN2A gene summary", "https://www.ncbi.nlm.nih.gov/gene/1029"),
            _evidence("GeneReviews: CDKN2A Cancer Predisposition", "https://www.ncbi.nlm.nih.gov/books/NBK616232/"),
            _evidence("ClinVar VCV000009412: CDKN2A c.301G>T / p.Gly101Trp", "https://www.ncbi.nlm.nih.gov/clinvar/variation/9412/"),
            _evidence("ClinVar RCV000034482: CDKN2A c.442G>A / p.Ala148Thr", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000034482/"),
            _evidence("PubMed 26557774: CDKN2A 3'UTR sequence variants in melanoma", "https://pubmed.ncbi.nlm.nih.gov/26557774/"),
            _evidence("PubMed 23816148: rs3088440 and 9p21 melanoma risk", "https://pubmed.ncbi.nlm.nih.gov/23816148/"),
            _evidence("PubMed 33910831: 3'UTR CDKN2A/CDK4 germline variants", "https://pubmed.ncbi.nlm.nih.gov/33910831/"),
            _evidence("PubMed 31270053: CDKN2A methylation and aging", "https://pubmed.ncbi.nlm.nih.gov/31270053/"),
        ],
        "variants": [
            {
                "variant": "rs11515",
                "display_name": "rs11515 (c.*29G>C)",
                "common_name": "CDKN2A 3'UTR regulatory SNP",
                "position": 21968199,
                "lookup_keys": [
                    "rs11515",
                    "CDKN2A:rs11515",
                    "9:21968199",
                    "9:21968199:C>G",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Research association / regulatory marker",
                "clinical_interpretation": (
                    "rs11515 is a common 3'UTR CDKN2A polymorphism curated here as a regulatory and association-study marker. "
                    "ClinVar submitters classify the c.*29G>C record as benign in germline contexts, while melanoma studies discuss it as part of 3'UTR haplotype and post-transcriptional regulation research."
                ),
                "clinical_significance": "Common/benign germline ClinVar context with research-level regulatory association literature.",
                "functional_effects": [
                    "Sits in the 3'UTR and may reflect post-transcriptional regulatory context at the CDKN2A locus.",
                    "Any disease signal is modest relative to rare truncating, deletion, or methylation-mediated CDKN2A loss.",
                ],
                "associated_conditions": [
                    "Melanoma and cancer-association studies",
                    "Tumor-suppressor regulatory biology",
                ],
                "research_context": [
                    "Common CDKN2A SNP work should be interpreted cautiously because the major clinical burden at this locus is usually rare loss of function or promoter silencing.",
                ],
                "usual_variant_note": "Common research-facing CDKN2A 3'UTR SNP.",
                "methylation_interpretation": (
                    "CDKN2A methylation and rs11515 should be read as different layers of regulatory context, not as interchangeable biomarkers."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar RCV001529165: rs11515 / c.*29G>C", "https://www.ncbi.nlm.nih.gov/clinvar/RCV001529165/"),
                    _evidence("PubMed 26557774: CDKN2A 3'UTR sequence variants in melanoma", "https://pubmed.ncbi.nlm.nih.gov/26557774/"),
                    _evidence("PubMed 33910831: 3'UTR CDKN2A/CDK4 germline variants", "https://pubmed.ncbi.nlm.nih.gov/33910831/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Przybyla et al., 2015 (PMID 26557774)",
                        "genotypes": "Common 3'UTR variant carrier states",
                        "phenotype": "Melanoma-focused sequence-variant analysis",
                        "finding": "The study examined CDKN2A 3'UTR variation including rs11515 as part of melanoma-focused regulatory sequence analysis, reinforcing its use as a research-level rather than high-penetrance marker.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/26557774/",
                    },
                    {
                        "paper": "Tovar-Parra et al., 2021 (PMID 33910831)",
                        "genotypes": "CDKN2A 500C>G / rs11515 with rs3088440 haplotypes",
                        "phenotype": "Cutaneous melanoma susceptibility in a Colombian case-control cohort",
                        "finding": "The cohort did not find rs11515 alone associated with melanoma risk, but reported haplotype context for 500G/540C, supporting cautious haplotype-level interpretation.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/33910831/",
                    },
                ],
            },
            {
                "variant": "rs3088440",
                "display_name": "rs3088440 (c.*69C>T / 540C>T)",
                "common_name": "CDKN2A 3'UTR melanoma-risk modifier",
                "position": 21968159,
                "lookup_keys": [
                    "rs3088440",
                    "CDKN2A:rs3088440",
                    "9:21968159",
                    "9:21968159:G>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Common 3'UTR association marker / melanoma-risk modifier",
                "clinical_interpretation": (
                    "rs3088440 is a CDKN2A 3'UTR polymorphism described as 540C>T or c.*69C>T depending on transcript orientation. "
                    "ClinVar submitter summaries list it as benign in germline contexts, while melanoma association studies report cohort-specific risk or haplotype signals."
                ),
                "clinical_significance": "Common/benign germline ClinVar context with cohort-specific melanoma association literature.",
                "functional_effects": [
                    "Located in the 3'UTR where altered mRNA stability or post-transcriptional regulation has been proposed.",
                    "Association direction can vary by ancestry, melanoma subtype, and haplotype with rs11515.",
                ],
                "associated_conditions": [
                    "Cutaneous melanoma susceptibility studies",
                    "9p21 melanoma-risk fine-mapping",
                    "Post-transcriptional tumor-suppressor regulation",
                ],
                "research_context": [
                    "Use rs3088440 as a common low-effect modifier or haplotype marker, not as a high-penetrance CDKN2A cancer-predisposition call.",
                    "The strongest local clinical escalation remains reserved for rare pathogenic CDKN2A alleles or clear tumor-suppressor silencing evidence.",
                ],
                "usual_variant_note": "Common CDKN2A 3'UTR marker also described as 540C>T.",
                "methylation_interpretation": (
                    "Pair rs3088440 with methylation as two regulatory layers at the CDKN2A locus; do not infer promoter silencing from the 3'UTR genotype alone."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 23816148: rs3088440 and 9p21 melanoma risk", "https://pubmed.ncbi.nlm.nih.gov/23816148/"),
                    _evidence("PubMed 33910831: 3'UTR CDKN2A/CDK4 germline variants", "https://pubmed.ncbi.nlm.nih.gov/33910831/"),
                    _evidence("ClinVar Miner: rs3088440 benign submitter summary", "https://clinvarminer.genetics.utah.edu/variants-by-gene/CDKN2A/significance/benign"),
                ],
                "literature_findings": [
                    {
                        "paper": "Ibarrola-Villava et al., 2013 (PMID 23816148)",
                        "genotypes": "T-allele carriers at rs3088440 / 540C>T",
                        "phenotype": "Melanoma risk in 9p21 fine-mapping",
                        "finding": "The study reported association between the rs3088440 T allele and melanoma risk in the Spanish cohort, while noting that prior evidence was inconsistent.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/23816148/",
                    },
                    {
                        "paper": "Tovar-Parra et al., 2021 (PMID 33910831)",
                        "genotypes": "CDKN2A 540C>T / rs3088440 with rs11515 haplotypes",
                        "phenotype": "Cutaneous melanoma susceptibility in a Colombian case-control cohort",
                        "finding": "The study found similar rs3088440 distributions in cases and controls but reported 3'UTR haplotype context, supporting population-specific interpretation.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/33910831/",
                    },
                ],
            },
            {
                "variant": "rs3731249",
                "display_name": "rs3731249 (p.Ala148Thr / A148T)",
                "common_name": "CDKN2A Ala148Thr benign missense polymorphism",
                "position": 21970916,
                "lookup_keys": [
                    "rs3731249",
                    "CDKN2A:rs3731249",
                    "CDKN2A p.Ala148Thr",
                    "p.A148T",
                    "9:21970916",
                    "9:21970916:C>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Benign polymorphism / historical melanoma-susceptibility candidate",
                "clinical_interpretation": (
                    "rs3731249 encodes p.Ala148Thr in p16INK4A for the MANE transcript and appears as a 3'UTR variant in several alternate CDKN2A transcripts. "
                    "Current ClinVar curation classifies it as benign with multiple submitters and no conflicts, so this local database keeps it as a historical research marker rather than a cancer-predisposition allele."
                ),
                "clinical_significance": "Benign germline ClinVar classification; historical research association marker.",
                "functional_effects": [
                    "Missense p.Ala148Thr in p16INK4A with benign ClinVar classification.",
                    "Should not be upgraded to pathogenic CDKN2A cancer predisposition without additional external evidence.",
                ],
                "associated_conditions": [
                    "Historical familial melanoma and nevus-susceptibility studies",
                    "Benign ClinVar germline variant interpretation",
                ],
                "research_context": [
                    "Useful mainly as a negative-control style example of why common CDKN2A polymorphisms need separation from rare pathogenic variants.",
                    "If observed in a VCF, report genotype dosage and benign curation rather than a high-risk cancer thesis.",
                ],
                "usual_variant_note": "Ala148Thr / A148T, currently benign in ClinVar.",
                "methylation_interpretation": (
                    "Do not infer CDKN2A promoter silencing from rs3731249; methylation and variant interpretation should remain separate."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar RCV000034482: CDKN2A p.Ala148Thr benign", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000034482/"),
                    _evidence("PubMed 22703879: exome screening secondary variants", "https://pubmed.ncbi.nlm.nih.gov/22703879/"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar RCV000034482, current record",
                        "genotypes": "p.Ala148Thr / rs3731249 carrier states",
                        "phenotype": "Germline variant classification",
                        "finding": "ClinVar reports p.Ala148Thr as benign with multiple submitters and no conflicts, making it inappropriate to interpret as a high-penetrance CDKN2A cancer-predisposition allele in this workbench.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/RCV000034482/",
                    },
                ],
            },
            {
                "variant": "CDKN2A p.Gly101Trp",
                "display_name": "CDKN2A p.Gly101Trp (c.301G>T / G101W)",
                "common_name": "Pathogenic CDKN2A G101W melanoma-pancreatic cancer marker",
                "position": 21971057,
                "lookup_keys": [
                    "CDKN2A p.Gly101Trp",
                    "CDKN2A:p.Gly101Trp",
                    "CDKN2A G101W",
                    "p.G101W",
                    "c.301G>T",
                    "NM_000077.5:c.301G>T",
                    "9:21971057:C>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "ClinVar pathogenic rare variant / cancer-predisposition marker",
                "clinical_interpretation": (
                    "CDKN2A c.301G>T / p.Gly101Trp is a rare pathogenic p16INK4A missense variant. "
                    "ClinVar lists pathogenic classifications across melanoma, hereditary cancer-predisposition, and melanoma-pancreatic cancer contexts, with reports in affected families and functional evidence for impaired p16INK4A activity."
                ),
                "clinical_significance": "Pathogenic ClinVar germline variant; requires genetics-professional review if observed.",
                "functional_effects": [
                    "Missense change in the p16INK4A ankyrin-repeat region that can disrupt CDK4 binding and cell-cycle arrest.",
                    "Also maps to an alternate p14ARF transcript consequence, so isoform context matters.",
                ],
                "associated_conditions": [
                    "Familial melanoma",
                    "Melanoma-pancreatic cancer syndrome",
                    "CDKN2A cancer predisposition",
                ],
                "research_context": [
                    "This marker is intentionally matched by exact GRCh37 REF -> ALT key rather than dbSNP ID alone because the rs104894094 cluster includes other alleles at the same locus.",
                    "Treat a matched row as a high-priority research and clinical-follow-up flag, not as a diagnosis from this app alone.",
                ],
                "usual_variant_note": "Rare pathogenic G101W / c.301G>T marker; exact allele orientation matters.",
                "methylation_interpretation": (
                    "If p.Gly101Trp is observed together with high CDKN2A promoter methylation, the sample has both sequence and epigenetic tumor-suppressor context, but external clinical confirmation remains essential."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar VCV000009412: CDKN2A c.301G>T / p.Gly101Trp", "https://www.ncbi.nlm.nih.gov/clinvar/variation/9412/"),
                    _evidence("ClinVar RCV000196633: p.Gly101Trp familial melanoma", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000196633/"),
                    _evidence("PubMed 11807902: G101W in Italian melanoma families", "https://pubmed.ncbi.nlm.nih.gov/11807902/"),
                    _evidence("PubMed 21462282: CDKN2A variant functional classification", "https://pubmed.ncbi.nlm.nih.gov/21462282/"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar VCV000009412, current record",
                        "genotypes": "CDKN2A c.301G>T / p.Gly101Trp carrier states",
                        "phenotype": "Familial melanoma and CDKN2A cancer-predisposition submissions",
                        "finding": "ClinVar classifies p.Gly101Trp as pathogenic and places it at chr9:21,971,057 on GRCh37, supporting exact allele-level matching in the local database.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/9412/",
                    },
                    {
                        "paper": "Mantelli et al., 2002 (PMID 11807902)",
                        "genotypes": "G101W germline mutation carriers",
                        "phenotype": "Italian malignant melanoma families",
                        "finding": "The study reported a high prevalence of the G101W germline mutation in Italian malignant melanoma families, supporting rare-founder and familial-risk context.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/11807902/",
                    }
                ],
            }
        ],
        "population_intro": "Broader population patterns curated from CDKN2A tumor-suppressor, methylation, and senescence literature.",
        "population_coverage_note": (
            "The bundled CDKN2A population database emphasizes rare-disease and epigenetic patterns because common SNP frequency panels capture only a small part of CDKN2A biology. Common 3'UTR SNPs are included as low-effect population research markers, while pathogenic CDKN2A findings require allele-specific clinical review."
        ),
        "population_sources": [
            _evidence("NCBI Gene 1029: CDKN2A gene summary", "https://www.ncbi.nlm.nih.gov/gene/1029"),
            _evidence("GeneReviews: CDKN2A Cancer Predisposition", "https://www.ncbi.nlm.nih.gov/books/NBK616232/"),
            _evidence("ClinVar VCV000009412: CDKN2A p.Gly101Trp", "https://www.ncbi.nlm.nih.gov/clinvar/variation/9412/"),
            _evidence("ClinVar RCV000034482: CDKN2A p.Ala148Thr benign", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000034482/"),
            _evidence("PubMed 31270053: CDKN2A methylation and aging", "https://pubmed.ncbi.nlm.nih.gov/31270053/"),
            _evidence("PubMed 26557774: CDKN2A 3'UTR variants in melanoma", "https://pubmed.ncbi.nlm.nih.gov/26557774/"),
            _evidence("PubMed 23816148: rs3088440 and 9p21 melanoma risk", "https://pubmed.ncbi.nlm.nih.gov/23816148/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "Rare CDKN2A loss-of-function variants",
                "location_group": "Rare disease families",
                "summary": "Across populations, the most clinically important CDKN2A events are rare germline or somatic loss-of-function alterations rather than common low-effect polymorphisms.",
            },
            {
                "variant": "CDKN2A p.Gly101Trp / G101W",
                "location_group": "Rare disease families",
                "summary": "G101W is curated as a rare pathogenic CDKN2A familial melanoma and melanoma-pancreatic cancer marker; match it allele-specifically rather than by rsID alone.",
            },
            {
                "variant": "rs11515 and rs3088440",
                "location_group": "Cancer cohorts",
                "summary": "Common CDKN2A 3'UTR variants vary by cohort and ancestry and should be read as low-effect melanoma/cancer association context rather than high-penetrance predisposition.",
            },
            {
                "variant": "rs3731249 / p.Ala148Thr",
                "location_group": "Global pattern",
                "summary": "Ala148Thr is common enough to appear in population screening and is currently benign in ClinVar, so it helps separate common polymorphism context from pathogenic CDKN2A events.",
            },
            {
                "variant": "CDKN2A promoter methylation",
                "location_group": "Cancer cohorts",
                "summary": "Promoter methylation is a recurring mode of CDKN2A silencing across tumor cohorts and is often more informative biologically than common inherited markers at this locus.",
            },
            {
                "variant": "CDKN2A methylation",
                "location_group": "Longevity cohorts",
                "summary": "In healthy-aging studies, CDKN2A methylation and expression are interpreted through senescence biology, which is distinct from the rare-variant cancer-predisposition literature.",
            },
        ],
    },
    {
        "gene_name": "TERT",
        "cytoband": "5p15.33",
        "chromosome": "5",
        "start": 1253282,
        "end": 1295183,
        "strand": "-",
        "gene_summary": (
            "TERT encodes telomerase reverse transcriptase, the catalytic core of telomerase that maintains telomere repeats and influences replicative lifespan, genome stability, stem-cell function, and oncogenesis."
        ),
        "clinical_context": (
            "The local TERT knowledge base spans telomere biology, cancer susceptibility, and promoter regulation. "
            "Common TERT variants are interpreted as research and risk-modifier signals, while the strongest clinical events often involve promoter mutations or rare high-impact pathogenic alleles."
        ),
        "variant_effect_overview": [
            "Common TERT polymorphisms are often interpreted through telomere-length, cancer-risk, or promoter-activity frameworks rather than as stand-alone diagnostic variants.",
            "TERT biology is highly tissue and phenotype specific, so common-variant effects should be read in the context of telomere maintenance and cohort design.",
        ],
        "condition_research_overview": [
            "TERT is central to telomere biology, cellular senescence, stem-cell maintenance, and oncogenesis.",
            "Common germline SNPs and somatic promoter mutations are both heavily studied in cancer biology, but they represent different interpretive layers.",
        ],
        "methylation_interpretation": (
            "TERT methylation is best interpreted as promoter and chromatin context around telomerase regulation. "
            "It can inform whether the locus appears permissive or repressed, but it is not a direct substitute for telomere-length or promoter-mutation testing."
        ),
        "methylation_effects": [
            "Promoter-proximal methylation can contribute to TERT expression modeling, especially in cancer biology.",
            "Interpret methylation alongside telomere biology, promoter mutation status, and tissue context.",
        ],
        "methylation_condition_research": [
            "Cancer and telomerase-reactivation studies.",
            "Aging and cellular-senescence research.",
            "Stem-cell and tissue-renewal biology.",
        ],
        "evidence": [
            _evidence("NCBI Gene 7015: TERT gene summary", "https://www.ncbi.nlm.nih.gov/gene/7015"),
            _evidence("PMCID PMC3329415: rs2736100 and cancer risk", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3329415/"),
            _evidence("PubMed 22994782: rs2736098 meta-analysis", "https://pubmed.ncbi.nlm.nih.gov/22994782/"),
        ],
        "variants": [
            {
                "variant": "rs2736100",
                "common_name": "intronic TERT cancer- and telomere-associated SNP",
                "region_class": "gene_body",
                "interpretation_scope": "Research association / telomere biology marker",
                "clinical_interpretation": (
                    "rs2736100 is one of the best-known common TERT SNPs and is curated here as a telomere-biology and cancer-susceptibility marker. "
                    "It is not treated as a pathogenic diagnostic allele in the local database."
                ),
                "clinical_significance": "Research-level telomere and cancer association marker.",
                "functional_effects": [
                    "Commonly interpreted through telomere-maintenance and cancer-risk frameworks.",
                    "Likely reflects regulatory or linked-locus effects on TERT biology rather than a direct truncating lesion.",
                ],
                "associated_conditions": [
                    "Lung and other cancer susceptibility studies",
                    "Telomere-length and telomerase-biology research",
                ],
                "research_context": [
                    "This is the canonical common inherited TERT association SNP in many cancer and telomere studies.",
                ],
                "usual_variant_note": "Most cited common TERT intronic association SNP.",
                "methylation_interpretation": (
                    "TERT methylation adds promoter and chromatin context but is not a direct biomarker of rs2736100 genotype."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PMCID PMC3329415: rs2736100 meta-analysis", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3329415/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Zhu et al., 2012 (PMCID PMC3329415)",
                        "genotypes": "Common rs2736100 genotypes",
                        "phenotype": "Cancer risk across pooled case-control studies",
                        "finding": "The pooled analysis supported rs2736100 as a recurrent TERT cancer-risk marker, reinforcing its role as a common inherited telomere-biology signal.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3329415/",
                    }
                ],
            },
            {
                "variant": "rs2853669",
                "common_name": "promoter regulatory TERT SNP",
                "region_class": "promoter",
                "interpretation_scope": "Research association / promoter modifier",
                "clinical_interpretation": (
                    "rs2853669 is a promoter-region TERT polymorphism frequently discussed as a modifier of promoter activity and cancer outcomes. "
                    "The local database treats it as a research-level promoter modifier rather than as a pathogenic allele."
                ),
                "clinical_significance": "Research-level promoter modifier.",
                "functional_effects": [
                    "Promoter-facing regulatory variant often modeled as a transcription-factor binding modifier.",
                    "Frequently interpreted together with broader TERT promoter biology in oncology studies.",
                ],
                "associated_conditions": [
                    "Cancer-risk and prognosis studies",
                    "Telomerase-promoter regulatory biology",
                ],
                "research_context": [
                    "This SNP is commonly discussed as a modifier of TERT promoter effects, especially in cancer cohorts.",
                ],
                "usual_variant_note": "Common TERT promoter modifier SNP.",
                "methylation_interpretation": (
                    "Promoter methylation and rs2853669 should be interpreted as complementary layers of TERT promoter context."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("NCBI Gene 7015: TERT gene summary", "https://www.ncbi.nlm.nih.gov/gene/7015"),
                ],
                "literature_findings": [],
            },
        ],
        "population_intro": "Broader population patterns curated from TERT telomere-biology and cancer-susceptibility literature.",
        "population_coverage_note": (
            "The bundled TERT population database is literature oriented and focuses on cohort patterns in telomere and cancer studies rather than a complete allele-frequency panel."
        ),
        "population_sources": [
            _evidence("NCBI Gene 7015: TERT gene summary", "https://www.ncbi.nlm.nih.gov/gene/7015"),
            _evidence("PMCID PMC3329415: rs2736100 meta-analysis", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3329415/"),
            _evidence("PubMed 22994782: rs2736098 meta-analysis", "https://pubmed.ncbi.nlm.nih.gov/22994782/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "rs2736100",
                "location_group": "Cancer cohorts",
                "summary": "Common TERT inherited variation is most densely studied in cancer cohorts, where rs2736100 is repeatedly used as a telomere-biology susceptibility marker.",
            },
            {
                "variant": "TERT promoter regulation",
                "location_group": "Global pattern",
                "summary": "At the TERT locus, inherited polymorphisms, promoter methylation, and somatic promoter mutations all matter, so population interpretation should distinguish among these different regulatory layers.",
            },
        ],
    },
    {
        "gene_name": "CLRN2",
        "cytoband": "4p15.32",
        "chromosome": "4",
        "start": 17516788,
        "end": 17528727,
        "strand": "+",
        "coordinate_source": (
            "NCBI Gene 645104 reports CLRN2 on GRCh37.p13 at NC_000004.11:17516788..17528727 "
            "and on GRCh38.p14 at NC_000004.12:17515165..17527104"
        ),
        "manifest_filter_region": "4:17515788-17528727",
        "curated_methylation_probe_ids": [
            "cg02953545",
            "cg06791107",
            "cg00389446",
            "cg09099893",
            "cg24197763",
            "cg01802294",
            "cg03685063",
            "cg03553245",
            "cg15130342",
        ],
        "gene_summary": (
            "CLRN2 encodes clarin-2, a small clarin-family membrane protein with four predicted transmembrane domains. "
            "Human, mouse, and zebrafish evidence places clarin-2 in inner-ear hair-cell stereocilia biology, where it supports mechanotransduction and maintenance of auditory hair bundles."
        ),
        "clinical_context": (
            "The local CLRN2 knowledge base is rare hearing-loss and stereocilia-maintenance oriented. "
            "Biallelic pathogenic CLRN2 variation causes autosomal recessive nonsyndromic hearing loss 117 (DFNB117), while isolated heterozygous or VUS findings should remain carrier or research context unless phase, phenotype, and external clinical review support a stronger interpretation."
        ),
        "variant_effect_overview": [
            "The best-supported human CLRN2 disease signal is the pathogenic c.494C>A / p.Thr165Lys allele, which also disrupts splicing in vitro and can produce a frameshifted transcript.",
            "Clarin-2-deficient mouse models show progressive hearing loss driven by stereocilia and mechanotransduction defects, with no overt retinal phenotype in the main mouse study.",
            "Current CLRN2 interpretation is rare-variant and recessive-model focused; common SNP or methylation-only findings should not be converted into hearing-loss predictions.",
        ],
        "condition_research_overview": [
            "Autosomal recessive nonsyndromic sensorineural hearing loss 117 (DFNB117).",
            "Auditory hair-cell stereocilia maintenance, mechanoelectrical transduction, and cochlear synaptic function.",
            "Preclinical AAV gene-supplementation research for clarin-2-associated progressive hearing loss.",
        ],
        "methylation_interpretation": (
            "CLRN2 has a bundled promoter-plus-gene EPIC slice with several TSS200, TSS1500, 5'UTR, and first-exon probes. "
            "Use CLRN2 methylation as local regulatory context around a compact hearing-loss gene, not as a validated DFNB117 diagnostic biomarker."
        ),
        "methylation_effects": [
            "The CLRN2 whitelist prioritizes promoter-proximal and 5'UTR/first-exon probes around the GRCh37 transcription start.",
            "No source-backed CLRN2 methylation threshold is bundled; beta values should be interpreted alongside tissue, cell composition, expression, and sequence evidence.",
        ],
        "methylation_condition_research": [
            "Use CLRN2 methylation as supportive regulatory context when investigating cochlear hair-cell, sensory-neuron, or rare hearing-loss hypotheses.",
            "Do not use peripheral EPIC methylation alone to infer CLRN2-related hearing-loss risk.",
        ],
        "evidence": [
            _evidence("NCBI Gene 645104: CLRN2 gene summary, coordinates, and DFNB117 link", "https://www.ncbi.nlm.nih.gov/gene/645104"),
            _evidence("UniProt A0PK11: CLRN2 / CLRN2_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A0PK11/entry"),
            _evidence("PubMed 33496845: biallelic CLRN2 variant causes nonsyndromic hearing loss", "https://pubmed.ncbi.nlm.nih.gov/33496845/"),
            _evidence("PubMed 31448880: clarin-2 is essential for hearing and stereocilia integrity", "https://pubmed.ncbi.nlm.nih.gov/31448880/"),
            _evidence("PubMed 38243601: clarin-2 gene supplementation preserves hearing in mice", "https://pubmed.ncbi.nlm.nih.gov/38243601/"),
        ],
        "variants": [
            {
                "variant": "CLRN2 c.494C>A",
                "display_name": "CLRN2 c.494C>A / p.Thr165Lys",
                "common_name": "T165K DFNB117 pathogenic allele with splicing effect",
                "position": 17528500,
                "lookup_keys": [
                    "CLRN2 c.494C>A",
                    "NM_001079827.2:c.494C>A",
                    "NM_001079827.2(CLRN2):c.494C>A",
                    "p.Thr165Lys",
                    "T165K",
                    "rs1711990645",
                    "4:17528500",
                    "4:17528500:C>A",
                    "4:17526877",
                    "4:17526877:C>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic DFNB117 marker / recessive hearing-loss context",
                "clinical_interpretation": (
                    "CLRN2 c.494C>A / p.Thr165Lys segregated as a homozygous biallelic variant in a consanguineous family with prelingual moderate-to-profound autosomal recessive sensorineural hearing loss. "
                    "The reported mechanism includes both the T165K missense change and aberrant splicing with intron retention that can create p.Gly146Lysfs*26."
                ),
                "clinical_significance": "Pathogenic ClinVar/OMIM DFNB117 variant; interpret under an autosomal-recessive hearing-loss model.",
                "functional_effects": [
                    "Missense change at a conserved clarin-2 residue.",
                    "In vitro splicing evidence supports an additional aberrant transcript with premature termination.",
                    "Patient-mutated CLRN2 forms failed to rescue hearing in the cited mouse gene-supplementation study.",
                ],
                "associated_conditions": [
                    "Hearing loss, autosomal recessive 117",
                    "Nonsyndromic sensorineural hearing loss",
                    "Auditory hair-bundle maintenance research",
                ],
                "research_context": [
                    "Prioritize zygosity, phase, family history, hearing phenotype, and clinical-grade confirmation before individual interpretation.",
                    "A single heterozygous call is most defensibly framed as carrier or recessive-disease context rather than as a deterministic hearing-loss prediction.",
                ],
                "usual_variant_note": "Pathogenic CLRN2 DFNB117 marker reported as c.494C>A / p.Thr165Lys with a splicing effect.",
                "methylation_interpretation": (
                    "Local CLRN2 methylation can provide regulatory context, but it does not change the need for variant-level recessive-disease review."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CLRN2 c.494C>A / p.Thr165Lys", "https://www.ncbi.nlm.nih.gov/clinvar/variation/996040/"),
                    _evidence("PubMed 33496845: human CLRN2 DFNB117 family and splicing assay", "https://pubmed.ncbi.nlm.nih.gov/33496845/"),
                    _evidence("PubMed 38243601: mutant CLRN2 forms fail rescue in mouse gene supplementation", "https://pubmed.ncbi.nlm.nih.gov/38243601/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Vona et al., 2021 (PMID 33496845)",
                        "genotypes": "Homozygous CLRN2 c.494C>A / p.Thr165Lys in an extended consanguineous family",
                        "phenotype": "Prelingual moderate-to-profound autosomal recessive sensorineural hearing loss",
                        "finding": "The study reported segregation of CLRN2 c.494C>A with nonsyndromic hearing loss and showed that the variant can alter both the amino acid and splicing outcome.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/33496845/",
                    },
                    {
                        "paper": "Mendia et al., 2024 (PMID 38243601)",
                        "genotypes": "Human CLRN2 wild-type and patient-mutated CLRN2 forms in Clrn2-deficient mice",
                        "phenotype": "Progressive hearing-loss rescue after AAV gene supplementation",
                        "finding": "Wild-type CLRN2 preserved hearing in treated mice, while patient-mutated CLRN2 forms failed to prevent hearing loss.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/38243601/",
                    },
                ],
            },
            {
                "variant": "CLRN2 c.236G>T",
                "display_name": "CLRN2 c.236G>T / p.Arg79Leu",
                "common_name": "R79L CLRN2 VUS hearing-loss marker",
                "position": 17517125,
                "lookup_keys": [
                    "CLRN2 c.236G>T",
                    "NM_001079827.2:c.236G>T",
                    "NM_001079827.2(CLRN2):c.236G>T",
                    "p.Arg79Leu",
                    "R79L",
                    "rs200144103",
                    "4:17517125",
                    "4:17517125:G>T",
                    "4:17515502",
                    "4:17515502:G>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "ClinVar VUS / recessive hearing-loss review context",
                "clinical_interpretation": (
                    "CLRN2 c.236G>T / p.Arg79Leu is included as a ClinVar variant of uncertain significance for DFNB117-related review. "
                    "A matched exploratory VCF row should be treated as a rare hearing-loss candidate marker, not as a pathogenic call without updated ClinVar, population-frequency, segregation, and phenotype evidence."
                ),
                "clinical_significance": "ClinVar variant of uncertain significance for hearing loss, autosomal recessive 117.",
                "functional_effects": [
                    "Missense change in CLRN2; no deterministic functional effect is bundled.",
                    "Requires external evidence review before use in disease interpretation.",
                ],
                "associated_conditions": [
                    "Hearing loss, autosomal recessive 117",
                    "Rare-variant hearing-loss panel review",
                ],
                "research_context": [
                    "Keep this marker below the pathogenic c.494C>A evidence tier unless newer submissions change classification.",
                    "Interpret through zygosity, phase, ancestry frequency, and phenotype match.",
                ],
                "usual_variant_note": "CLRN2 VUS reported as c.236G>T / p.Arg79Leu.",
                "methylation_interpretation": (
                    "Methylation near CLRN2 does not resolve the variant's VUS classification."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CLRN2 c.236G>T / p.Arg79Leu", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1333538/"),
                    _evidence("NCBI Gene 645104: CLRN2 and DFNB117 condition link", "https://www.ncbi.nlm.nih.gov/gene/645104"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar CLRN2 c.236G>T submission",
                        "genotypes": "NM_001079827.2:c.236G>T / p.Arg79Leu",
                        "phenotype": "Hearing loss, autosomal recessive 117 review context",
                        "finding": "The variant is curated locally as a VUS-level marker, so it should remain a candidate finding until external evidence supports reclassification.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/1333538/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from CLRN2 hearing-loss, stereocilia, and gene-supplementation literature.",
        "population_coverage_note": (
            "The bundled CLRN2 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Rare-variant interpretation should be checked against current ClinVar, gnomAD, ancestry background, zygosity, phase, and the patient's hearing phenotype before clinical use."
        ),
        "population_sources": [
            _evidence("NCBI Gene 645104: CLRN2 gene summary and DFNB117 association", "https://www.ncbi.nlm.nih.gov/gene/645104"),
            _evidence("PubMed 33496845: biallelic CLRN2 human hearing-loss evidence", "https://pubmed.ncbi.nlm.nih.gov/33496845/"),
            _evidence("PubMed 31448880: mouse CLRN2 stereocilia and progressive hearing-loss evidence", "https://pubmed.ncbi.nlm.nih.gov/31448880/"),
            _evidence("PubMed 38243601: CLRN2 gene supplementation in mouse model", "https://pubmed.ncbi.nlm.nih.gov/38243601/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "CLRN2 c.494C>A",
                "location_group": "Rare disease families",
                "summary": "The strongest human evidence is a homozygous c.494C>A allele segregating with autosomal recessive nonsyndromic hearing loss in a consanguineous Iranian family.",
            },
            {
                "variant": "CLRN2 rare missense variants",
                "location_group": "Disease cohorts",
                "summary": "Other CLRN2 missense findings should be interpreted through rare hearing-loss panel evidence, current ClinVar classification, ancestry frequency, zygosity, and phenotype consistency.",
            },
            {
                "variant": "CLRN2 stereocilia function",
                "location_group": "Functional biology",
                "summary": "Mouse and zebrafish work supports clarin-2 as a hair-cell stereocilia maintenance and mechanotransduction gene, strengthening rare-variant biological plausibility.",
            },
            {
                "variant": "CLRN2 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "The bundled EPIC probes provide local promoter and early-gene-body regulatory context, but no population methylation threshold is bundled for DFNB117 interpretation.",
            },
        ],
    },
    {
        "gene_name": "ARHGAP10",
        "cytoband": "4q31.23",
        "chromosome": "4",
        "start": 148653239,
        "end": 148993927,
        "strand": "+",
        "coordinate_source": (
            "NCBI Gene 79658 reports ARHGAP10 on GRCh37.p13 at NC_000004.11:148653239..148993927 "
            "and on GRCh38.p14 at NC_000004.12:147732088..148072776; Ensembl GRCh37 reports "
            "ENSG00000071205 at chr4:148653214-148993931 on the forward strand"
        ),
        "manifest_filter_region": "4:148652239-148993927",
        "gene_summary": (
            "ARHGAP10 encodes Rho GTPase-activating protein 10, also called GRAF2 or PS-GAP. "
            "The protein contains BAR, PH, RhoGAP, and SH3 domain architecture and helps turn off Rho-family small-GTPase signaling, including RhoA and Cdc42 contexts that connect the locus to cytoskeletal organization, neurite morphology, endosomal membrane biology, and cell migration."
        ),
        "clinical_context": (
            "The local ARHGAP10 knowledge base is neuropsychiatric and RhoGAP-pathway research oriented. "
            "Rare exonic CNVs and the p.Ser490Pro missense marker have been reported in Japanese schizophrenia cohorts and model systems, but this bundle is not a diagnostic schizophrenia test and should not convert single heterozygous or methylation-only findings into a clinical prediction."
        ),
        "variant_effect_overview": [
            "The strongest human genetics signal bundled here is a case-control association between schizophrenia and rare exonic ARHGAP10 CNVs in a Japanese cohort.",
            "The p.Ser490Pro missense marker is curated because one schizophrenia case carried it with an exonic deletion on the other allele; functional follow-up linked this double-hit model to RhoA binding/signaling and neuronal morphology phenotypes.",
            "Cancer biology evidence frames ARHGAP10 as a candidate tumor suppressor in ovarian cancer through Cdc42 inactivation, cell-cycle effects, apoptosis, adhesion, migration, and invasion assays.",
        ],
        "condition_research_overview": [
            "Schizophrenia rare-variant and exonic copy-number research, especially RhoA/RhoGAP neuronal morphology mechanisms.",
            "Mouse and iPSC-derived neuron models of ARHGAP10 p.Ser490Pro plus loss-of-function or exonic deletion contexts.",
            "Ovarian cancer and broader cell-migration biology in which ARHGAP10 regulates Cdc42/Rho-family signaling.",
        ],
        "methylation_interpretation": (
            "ARHGAP10 has a bundled promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use ARHGAP10 methylation as locus-regulatory context for a large forward-strand RhoGAP gene, not as a validated schizophrenia, cancer, or medication-response biomarker."
        ),
        "methylation_effects": [
            "The ARHGAP10 methylation view can summarize promoter-proximal and early-gene-body CpGs near the GRCh37 transcription start.",
            "No source-backed ARHGAP10 methylation threshold is bundled; beta values should be interpreted alongside tissue, cell composition, expression, CNV, and sequence evidence.",
        ],
        "methylation_condition_research": [
            "Use ARHGAP10 methylation as supportive regulatory context in RhoGAP, neurodevelopmental, neuronal morphology, or cancer-migration research.",
            "Do not use peripheral EPIC methylation alone to infer ARHGAP10-related schizophrenia risk or tumor behavior.",
        ],
        "evidence": [
            _evidence("NCBI Gene 79658: ARHGAP10 gene summary, GRCh37/GRCh38 coordinates, RefSeq, and GO context", "https://www.ncbi.nlm.nih.gov/gene/79658"),
            _evidence("Ensembl GRCh37 ENSG00000071205: ARHGAP10 gene model and forward-strand coordinates", "https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000071205"),
            _evidence("UniProt A1A4S6: ARHGAP10 / RHG10_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A1A4S6/entry"),
            _evidence("Translational Psychiatry 2020: ARHGAP10 and schizophrenia risk", "https://www.nature.com/articles/s41398-020-00917-z"),
            _evidence("Molecular Brain 2021: Arhgap10 S490P/NHEJ mouse model", "https://molecularbrain.biomedcentral.com/articles/10.1186/s13041-021-00735-4"),
            _evidence("PubMed 27010858: ARHGAP10 suppresses ovarian cancer tumorigenicity through Cdc42-linked biology", "https://pubmed.ncbi.nlm.nih.gov/27010858/"),
            _evidence("ClinVar Miner: ARHGAP10 not-provided variants including rs483352828 / p.Ser490Pro", "https://clinvarminer.genetics.utah.edu/variants-by-gene/ARHGAP10/significance/not%20provided"),
        ],
        "variants": [
            {
                "variant": "ARHGAP10 exonic CNV",
                "display_name": "ARHGAP10 rare exonic copy-number variant",
                "common_name": "ARHGAP10 schizophrenia exonic CNV research marker",
                "position": None,
                "lookup_keys": [
                    "ARHGAP10 exonic CNV",
                    "ARHGAP10 CNV",
                    "ARHGAP10 deletion",
                    "ARHGAP10 duplication",
                    "ARHGAP10 copy-number variant",
                    "ARHGAP10 copy number variant",
                    "schizophrenia ARHGAP10 CNV",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Rare exonic CNV research marker / schizophrenia cohort context",
                "clinical_interpretation": (
                    "A 2020 schizophrenia study identified rare exonic ARHGAP10 CNVs in patients and reported a case-control association in a Japanese cohort. "
                    "Because breakpoints, exon content, assay method, ancestry, psychiatric phenotype, and inheritance matter, this bundled record is structural-variant research context rather than a single-SNV clinical classification."
                ),
                "clinical_significance": "Research-level schizophrenia CNV association; not a diagnostic ARHGAP10 result.",
                "functional_effects": [
                    "Reported exonic CNVs overlapped functional ARHGAP10 domains, including BAR, PH, RhoGAP, and SH3 contexts in the cited study.",
                    "Patient-derived and model-system work supports a RhoA/RhoGAP neuronal morphology mechanism when ARHGAP10 function is disrupted.",
                ],
                "associated_conditions": [
                    "Schizophrenia rare CNV research",
                    "Neuronal morphology and dendritic spine biology",
                    "RhoA/RhoGAP signaling research",
                ],
                "research_context": [
                    "Confirm CNV breakpoints, exon overlap, assembly, and copy-number method before using this marker.",
                    "Interpret as cohort-level evidence; do not treat a broad ARHGAP10-region CNV as a deterministic schizophrenia prediction.",
                ],
                "usual_variant_note": "Rare exonic ARHGAP10 CNV signal reported in Japanese schizophrenia case-control analysis.",
                "methylation_interpretation": (
                    "ARHGAP10 methylation can be paired with CNV status only as regulatory context; it does not establish dosage pathogenicity."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("Translational Psychiatry 2020: exonic ARHGAP10 CNVs and schizophrenia", "https://www.nature.com/articles/s41398-020-00917-z"),
                    _evidence("NCBI Gene 79658: ARHGAP10 gene and domain context", "https://www.ncbi.nlm.nih.gov/gene/79658"),
                ],
                "literature_findings": [
                    {
                        "paper": "Sekiguchi et al., 2020 (PMID 32699248)",
                        "genotypes": "Rare exonic ARHGAP10 CNVs in schizophrenia cases versus controls",
                        "phenotype": "Schizophrenia in a Japanese case-control cohort",
                        "finding": "The study reported a significant association between exonic ARHGAP10 CNVs and schizophrenia and prioritized RhoGAP signaling as a plausible neuronal-development mechanism.",
                        "url": "https://www.nature.com/articles/s41398-020-00917-z",
                    }
                ],
            },
            {
                "variant": "ARHGAP10 p.Ser490Pro",
                "display_name": "ARHGAP10 c.1468T>C / p.Ser490Pro",
                "common_name": "S490P ARHGAP10 rare missense schizophrenia research marker",
                "position": None,
                "lookup_keys": [
                    "ARHGAP10 p.Ser490Pro",
                    "ARHGAP10 S490P",
                    "p.Ser490Pro",
                    "S490P",
                    "NM_024605.3:c.1468T>C",
                    "NM_024605.3(ARHGAP10):c.1468T>C",
                    "NM_024605.4:c.1468T>C",
                    "NM_024605.4(ARHGAP10):c.1468T>C",
                    "rs483352828",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Rare missense research marker / double-hit schizophrenia model context",
                "clinical_interpretation": (
                    "ARHGAP10 p.Ser490Pro was highlighted in a schizophrenia patient who also carried an exonic ARHGAP10 deletion on the other allele. "
                    "The cited study located S490P in the RhoGAP domain and reported altered ARHGAP10 interaction with active RhoA; follow-up mouse work used an S490P/NHEJ double-hit model to study striatal, nucleus-accumbens, and cognitive-response phenotypes."
                ),
                "clinical_significance": "Research-level rare missense marker; not a stand-alone pathogenic clinical classification.",
                "functional_effects": [
                    "Missense change in the ARHGAP10 RhoGAP domain.",
                    "Reported to affect interaction with active RhoA in the cited schizophrenia study.",
                    "Double-hit Arhgap10 S490P/NHEJ mouse models show neuronal morphology and methamphetamine-vulnerability phenotypes in follow-up studies.",
                ],
                "associated_conditions": [
                    "Schizophrenia rare-variant research",
                    "RhoA/Rho-kinase signaling",
                    "Neuronal morphology and dendritic spine biology",
                ],
                "research_context": [
                    "The strongest interpretation depends on the double-hit context: p.Ser490Pro plus a second ARHGAP10 loss-of-function or exonic deletion event.",
                    "A single heterozygous S490P row should remain a rare-variant research flag unless external clinical evidence changes.",
                ],
                "usual_variant_note": "ARHGAP10 c.1468T>C / p.Ser490Pro, dbSNP rs483352828, rare missense marker in schizophrenia model literature.",
                "methylation_interpretation": (
                    "Local methylation may add regulatory context but does not substitute for CNV, phase, zygosity, or functional review of S490P."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("Translational Psychiatry 2020: ARHGAP10 p.Ser490Pro double-hit case and RhoA interaction", "https://www.nature.com/articles/s41398-020-00917-z"),
                    _evidence("Molecular Brain 2021: Arhgap10 S490P/NHEJ model and neuronal morphology", "https://molecularbrain.biomedcentral.com/articles/10.1186/s13041-021-00735-4"),
                    _evidence("ClinVar Miner: rs483352828 / NM_024605.4:c.1468T>C (p.Ser490Pro)", "https://clinvarminer.genetics.utah.edu/variants-by-gene/ARHGAP10/significance/not%20provided"),
                ],
                "literature_findings": [
                    {
                        "paper": "Sekiguchi et al., 2020 (PMID 32699248)",
                        "genotypes": "ARHGAP10 exonic deletion plus p.Ser490Pro in one schizophrenia case",
                        "phenotype": "Schizophrenia, neuronal morphology, and RhoA/RhoGAP functional assays",
                        "finding": "The paper prioritized the double-hit case, reported S490P in the RhoGAP domain, and connected the marker to altered RhoA interaction and neuronal morphology phenotypes.",
                        "url": "https://www.nature.com/articles/s41398-020-00917-z",
                    },
                    {
                        "paper": "Hada et al., 2021 (Molecular Brain 14:21)",
                        "genotypes": "Arhgap10 S490P/NHEJ mouse model",
                        "phenotype": "Striatal and nucleus-accumbens neuronal morphology plus methamphetamine vulnerability",
                        "finding": "The follow-up mouse model showed altered neuronal complexity and spine-density phenotypes and vulnerability of visual-discrimination performance after methamphetamine treatment.",
                        "url": "https://molecularbrain.biomedcentral.com/articles/10.1186/s13041-021-00735-4",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from ARHGAP10 schizophrenia, RhoGAP signaling, and cancer-cell biology literature.",
        "population_coverage_note": (
            "The bundled ARHGAP10 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Rare-variant interpretation should be checked against current ClinVar/dbSNP/gnomAD resources, ancestry background, zygosity, CNV status, phase, psychiatric phenotype, and orthogonal confirmation before clinical use."
        ),
        "population_sources": [
            _evidence("NCBI Gene 79658: ARHGAP10 gene summary and GWAS/CNV links", "https://www.ncbi.nlm.nih.gov/gene/79658"),
            _evidence("Translational Psychiatry 2020: Japanese schizophrenia ARHGAP10 CNV and p.Ser490Pro evidence", "https://www.nature.com/articles/s41398-020-00917-z"),
            _evidence("Molecular Brain 2021: Arhgap10 S490P/NHEJ model", "https://molecularbrain.biomedcentral.com/articles/10.1186/s13041-021-00735-4"),
            _evidence("PubMed 27010858: ovarian cancer ARHGAP10/Cdc42 functional evidence", "https://pubmed.ncbi.nlm.nih.gov/27010858/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "ARHGAP10 exonic CNVs",
                "location_group": "Disease cohorts",
                "summary": "The main human genetics signal is a Japanese schizophrenia case-control study reporting rare exonic CNVs in ARHGAP10 patients and no equivalent control CNVs after study QC.",
            },
            {
                "variant": "ARHGAP10 p.Ser490Pro",
                "location_group": "Rare disease families",
                "summary": "p.Ser490Pro is most defensibly interpreted in the reported double-hit context with an exonic deletion on the other allele; single-marker rows should remain rare missense research flags.",
            },
            {
                "variant": "ARHGAP10 RhoGAP function",
                "location_group": "Functional biology",
                "summary": "Functional interpretation centers on RhoA/RhoGAP signaling, neuronal morphology, and Cdc42-linked cell migration or tumor-suppressor biology rather than on common-variant deterministic prediction.",
            },
            {
                "variant": "ARHGAP10 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "The bundled EPIC probes provide local promoter and gene-body regulatory context, but no population methylation threshold is bundled for schizophrenia or cancer interpretation.",
            },
        ],
    },
    {
        "gene_name": "CCDC66",
        "cytoband": "3p14.3",
        "chromosome": "3",
        "start": 56591184,
        "end": 56655865,
        "strand": "+",
        "coordinate_source": (
            "NCBI Gene 285331 reports CCDC66 on GRCh37.p13 at NC_000003.11:56591184..56655865 "
            "and on GRCh38.p14 at NC_000003.12:56557156..56621837; ClinGen Gene Facts HGNC:27709 "
            "reports GRCh37 chr3:56591184-56655865 and GRCh38 chr3:56557156-56621837"
        ),
        "manifest_filter_region": "3:56590184-56655865",
        "gene_summary": (
            "CCDC66 encodes coiled-coil domain-containing protein 66, a microtubule-binding protein with centrosome, centriolar-satellite, ciliary axoneme, ciliary transition-zone, midbody, and photoreceptor inner-segment evidence. "
            "NCBI, UniProt, ClinGen Gene Facts, Human Protein Atlas, and cell-biology literature frame CCDC66 as a ciliogenesis and microtubule-organization protein that interacts with CEP290 and PCM1, supports ciliary protein trafficking, and participates in cilium length, signaling, stability, mitotic progression, and cytokinesis."
        ),
        "clinical_context": (
            "The local CCDC66 knowledge base is retinal degeneration model, high myopia, photoreceptor/cilium, centrosome, mitotic microtubule, and epithelial-cilium signaling research oriented. "
            "Canine and mouse CCDC66 loss models support retinal degeneration biology, and one human study reported a suspected nonsense variant co-segregating with high myopia, but ClinGen has not published a CCDC66 gene-disease validity or dosage curation; app findings should remain research-grade unless external clinical review supplies stronger variant-level evidence."
        ),
        "variant_effect_overview": [
            "Canine Schapendoes and Portuguese Water Dog studies reported recessive CCDC66 frameshift/insertion variants in generalized or early-onset progressive retinal atrophy, and mouse Ccdc66 null models show early, slow progressive rod-cone dysplasia and retinal dysfunction.",
            "Human high-myopia sequencing reported a suspected nonsense CCDC66 c.C172T / p.Q58X variant co-segregating with high myopia in a family and additional rare CCDC66 variants in sporadic high-myopia cases, with functional evidence for altered cell proliferation, microtubule polymerization, acetylated tubulin, and mitotic tubulin-system synchronization.",
            "Cellular studies place CCDC66 at centrosomes, centriolar satellites, microtubules, and cilia, with CEP290/PCM1 interaction, BBS4 ciliary trafficking, transition-zone/IFT-B recruitment, cilium length and signaling, Hedgehog/Wnt response, and epithelial organization evidence.",
            "Mitotic literature shows CCDC66 depletion can disrupt spindle assembly/orientation, kinetochore fiber stability, chromosome alignment, central spindle and midbody organization, and cytokinesis, supporting a cytoskeletal mechanism that may intersect retinal growth and ciliopathy biology.",
        ],
        "condition_research_overview": [
            "Inherited retinal degeneration model systems, canine progressive retinal atrophy, mouse rod-cone dysplasia, photoreceptor inner-segment expression, and retinitis-pigmentosa-like retinal dysfunction research.",
            "High myopia, embryonic retinal development, retinal cell proliferation, mitotic microtubule organization, and ophthalmic rare-variant sequencing.",
            "Ciliogenesis, centriolar satellites, CEP290/PCM1 interaction, BBS4 trafficking, transition-zone and IFT-B recruitment, cilium length/stability/disassembly, Hedgehog/Wnt signaling, epithelial polarity, and broader ciliopathy model biology.",
        ],
        "methylation_interpretation": (
            "CCDC66 has a promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use CCDC66 methylation as local regulatory context around a forward-strand ciliary microtubule and retinal-development gene, not as a validated retinal degeneration, high myopia, ciliopathy, cancer, or mitotic-defect biomarker."
        ),
        "methylation_effects": [
            "Promoter-proximal and gene-body methylation may help frame local CCDC66 transcriptional accessibility in samples where sequence, expression, retinal phenotype, or cilium biology evidence is also available.",
            "No source-backed CCDC66 methylation threshold is bundled; beta values should be interpreted alongside tissue, developmental timing, cell composition, expression, sequence/CNV evidence, variant classification, and ophthalmic phenotype.",
        ],
        "methylation_condition_research": [
            "Use CCDC66 methylation as supportive local regulatory context in retinal degeneration, high myopia, ciliogenesis, ciliary signaling, epithelial organization, mitosis/cytokinesis, or circCCDC66 expression research.",
            "Do not infer retinal degeneration, high myopia, ciliary transport failure, Hedgehog/Wnt impairment, epithelial polarity defects, or cancer behavior from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 285331: CCDC66 gene summary, GRCh37/GRCh38 coordinates, expression, RefSeq, GO, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/285331"),
            _evidence("Ensembl GRCh37 ENSG00000180376: CCDC66 gene model and hg19 coordinates", "https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000180376"),
            _evidence("UniProt A2RUB6: CCDC66 / coiled-coil domain-containing protein 66 entry", "https://www.uniprot.org/uniprotkb/A2RUB6/entry"),
            _evidence("Human Protein Atlas ENSG00000180376-CCDC66: expression, protein, and structure context", "https://www.proteinatlas.org/ENSG00000180376-CCDC66"),
            _evidence("ClinGen Gene Facts HGNC:27709: CCDC66 coordinates, MANE transcript, function, constraint, and no published ClinGen curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27709"),
            _evidence("PubMed 19777273: CCDC66 mutation in Schapendoes canine progressive retinal atrophy", "https://pubmed.ncbi.nlm.nih.gov/19777273/"),
            _evidence("PubMed 21680557: Ccdc66 null mutation causes retinal degeneration and dysfunction in mouse", "https://pubmed.ncbi.nlm.nih.gov/21680557/"),
            _evidence("PubMed 28235840 / PMCID PMC5399785: CCDC66 interacts with CEP290 and functions in cilium formation and trafficking", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5399785/"),
            _evidence("PubMed 31582766 / PMCID PMC6776500: CCDC66 centrosomal/ciliary targeting and centriolar satellite dynamics", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6776500/"),
            _evidence("PubMed 33273526: Portuguese Water Dog CCDC66 frameshift variant in early-onset progressive retinal atrophy", "https://pubmed.ncbi.nlm.nih.gov/33273526/"),
            _evidence("PubMed 36606424: CCDC66 regulates primary cilium length and signaling via transition-zone and axonemal interactions", "https://pubmed.ncbi.nlm.nih.gov/36606424/"),
            _evidence("PLOS Biology 2022: CCDC66 controls mitotic progression and cytokinesis through microtubule nucleation and organization", "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001708"),
            _evidence("PubMed 37852749: CCDC66 mutations associated with high myopia through affected cell mitosis", "https://pubmed.ncbi.nlm.nih.gov/37852749/"),
            _evidence("PMCID PMC12324684: CCDC66 regulation of cytoskeleton and cilia stability in signaling and epithelial organization", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12324684/"),
        ],
        "variants": [
            {
                "variant": "CCDC66 c.C172T / p.Q58X",
                "display_name": "CCDC66 c.C172T / p.Q58X",
                "common_name": "CCDC66 Q58X high-myopia research marker",
                "position": None,
                "lookup_keys": [
                    "CCDC66 c.C172T",
                    "CCDC66 c.172C>T",
                    "CCDC66 p.Q58X",
                    "CCDC66 p.Gln58Ter",
                    "CCDC66 Q58X",
                    "CCDC66 Q58*",
                    "CCDC66 Gln58Ter",
                    "NM_001141947.3:c.172C>T",
                    "CCDC66 nonsense high myopia",
                    "CCDC66 high myopia c.C172T",
                    "CCDC66 high myopia p.Q58X",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Emerging high-myopia sequencing marker / retinal-development and mitotic microtubule research context",
                "clinical_interpretation": (
                    "CCDC66 c.C172T / p.Q58X is included as an emerging human ophthalmic research marker because Chen et al. reported the suspected nonsense variant co-segregating with high myopia in a family and identified additional rare CCDC66 variants in 200 sporadic high-myopia cases. "
                    "Functional follow-up connected CCDC66 deficiency and mutation context to retinal developmental expression, lower cell proliferation, lower microtubule polymerization rate, altered acetylated tubulin, and failure of mutated CCDC66 to synchronize with the tubulin system during mitosis."
                ),
                "clinical_significance": "Research-level high-myopia marker; not a ClinGen-curated diagnostic CCDC66 variant and not a stand-alone retinal-disease prediction.",
                "functional_effects": [
                    "Reported as a suspected nonsense CCDC66 variant co-segregating with high myopia in one family.",
                    "CCDC66 was consistently expressed in embryonic human and mouse retina in the cited study.",
                    "CCDC66-deficient cells had reduced proliferation, microtubule polymerization, and acetylated tubulin levels, and mutated CCDC66 failed to align with the tubulin system during mitosis.",
                ],
                "associated_conditions": [
                    "High myopia research",
                    "Retinal development and retinal growth biology",
                    "Mitotic microtubule organization and cell-proliferation studies",
                ],
                "research_context": [
                    "Interpret this marker with transcript version, variant normalization, zygosity, segregation, family history, refraction severity, retinal phenotype, and current ClinVar/gnomAD evidence.",
                    "Do not treat a single CCDC66 Q58X-like row as a deterministic high-myopia diagnosis without clinical-grade confirmation and ophthalmic context.",
                    "Because ClinGen has no published CCDC66 curation, use this as a research flag rather than a pathogenic assertion.",
                ],
                "usual_variant_note": "Emerging CCDC66 high-myopia nonsense marker from one family and sporadic-case follow-up; clinical validity remains unsettled.",
                "methylation_interpretation": (
                    "CCDC66 methylation can provide local regulatory context, but it does not establish Q58X variant dosage, CCDC66 protein truncation, retinal expression, or high-myopia causality."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 37852749: CCDC66 c.C172T / p.Q58X in high myopia and mitotic functional follow-up", "https://pubmed.ncbi.nlm.nih.gov/37852749/"),
                    _evidence("ClinGen Gene Facts HGNC:27709: no published CCDC66 clinical curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27709"),
                    _evidence("NCBI Gene 285331: CCDC66 coordinates and bibliography", "https://www.ncbi.nlm.nih.gov/gene/285331"),
                ],
                "literature_findings": [
                    {
                        "paper": "Chen et al., 2024 (PMID 37852749)",
                        "genotypes": "CCDC66 c.C172T / p.Q58X co-segregating with high myopia in one family; six additional rare variants in 200 sporadic high-myopia cases",
                        "phenotype": "High myopia, retinal developmental expression, cell proliferation, microtubule polymerization, and mitotic tubulin-system alignment",
                        "finding": "The study reported an association between the CCDC66 nonsense marker and high myopia and proposed a mechanism involving altered mitotic cell behavior during retinal growth.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/37852749/",
                    }
                ],
            },
            {
                "variant": "CCDC66 retinal degeneration/ciliary loss model",
                "display_name": "CCDC66 loss-of-function retinal degeneration and ciliary trafficking model",
                "common_name": "CCDC66 photoreceptor-cilium loss model",
                "position": None,
                "lookup_keys": [
                    "CCDC66 loss-of-function",
                    "CCDC66 loss of function",
                    "CCDC66 LOF",
                    "CCDC66 deletion",
                    "CCDC66 frameshift",
                    "Ccdc66 null",
                    "Ccdc66 knockout",
                    "CCDC66 retinal degeneration",
                    "CCDC66 progressive retinal atrophy",
                    "CCDC66 PRA",
                    "CCDC66 gPRA",
                    "CCDC66 EOPRA",
                    "CCDC66 c.2262_c.2263insA",
                    "CCDC66 p.Val747SerfsTer8",
                    "CCDC66 Schapendoes insertion",
                    "CCDC66 Portuguese Water Dog frameshift",
                    "CCDC66 CEP290",
                    "CCDC66 PCM1",
                    "CCDC66 cilium length",
                    "CCDC66 transition zone",
                    "CCDC66 Hedgehog Wnt",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Model-system loss-of-function and ciliary trafficking mechanism / retinal degeneration and ciliopathy research context",
                "clinical_interpretation": (
                    "CCDC66 loss-of-function and frameshift/deletion contexts are curated as retinal-cilium model evidence rather than as a validated human diagnostic rule. "
                    "Schapendoes and Portuguese Water Dog studies linked recessive CCDC66 insertion/frameshift variants to progressive retinal atrophy, while mouse Ccdc66 null animals lack retinal Ccdc66 RNA/protein and develop early slow progressive rod-cone dysplasia with retinal dysfunction."
                ),
                "clinical_significance": "Strong animal-model retinal degeneration and ciliary trafficking mechanism; human clinical interpretation remains research-grade without variant-level clinical review.",
                "functional_effects": [
                    "Canine CCDC66 mutations support recessive progressive retinal atrophy models with photoreceptor degeneration and loss or mislocalization of CCDC66 protein.",
                    "Mouse Ccdc66 null models show early photoreceptor degeneration, reduced scotopic and photopic electroretinography responses, and postnatal photoreceptor outer/inner-segment expression relevance.",
                    "CCDC66 localizes to centrosomes, cilia, centriolar satellites, microtubules, spindle structures, and midbody, and interacts with CEP290, PCM1, and other ciliopathy/cytoskeletal proteins.",
                    "CCDC66 regulates cilium formation, BBS4 trafficking, transition-zone assembly, IFT-B recruitment, cilium length/stability/disassembly, Hedgehog/Wnt signaling, epithelial organization, mitotic progression, and cytokinesis in cited cell models.",
                ],
                "associated_conditions": [
                    "Canine progressive retinal atrophy and early-onset progressive retinal atrophy",
                    "Mouse retinal degeneration / rod-cone dysplasia model",
                    "Photoreceptor inner-segment and ciliary trafficking biology",
                    "Ciliogenesis, transition-zone, Hedgehog/Wnt signaling, mitosis, and cytokinesis research",
                ],
                "research_context": [
                    "Use this broad model for loss-of-function, frameshift, deletion, or curated functional annotations rather than generic benign or VUS SNP rows.",
                    "Human translation should be checked against current ClinVar, gnomAD, ClinGen, OMIM, ophthalmic phenotype, segregation, and assay context before any clinical escalation.",
                    "Animal-model and cell-model evidence can support mechanistic hypotheses but should not be used alone as a deterministic human retinal disease prediction.",
                ],
                "usual_variant_note": "Model-system CCDC66 loss/ciliary trafficking marker rather than a single clinically curated human pathogenic variant.",
                "methylation_interpretation": (
                    "CCDC66 methylation can support local regulatory review, but it does not prove loss-of-function, photoreceptor degeneration, ciliary trafficking failure, or epithelial signaling defects."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PubMed 19777273: Schapendoes CCDC66 insertion and generalized progressive retinal atrophy", "https://pubmed.ncbi.nlm.nih.gov/19777273/"),
                    _evidence("PubMed 21680557: mouse Ccdc66 null retinal degeneration and dysfunction", "https://pubmed.ncbi.nlm.nih.gov/21680557/"),
                    _evidence("PubMed 33273526: Portuguese Water Dog CCDC66 frameshift and early-onset progressive retinal atrophy", "https://pubmed.ncbi.nlm.nih.gov/33273526/"),
                    _evidence("PMCID PMC5399785: CCDC66 interaction with CEP290 and ciliary trafficking", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5399785/"),
                    _evidence("PubMed 36606424: CCDC66 cilium length/signaling and transition-zone/axonemal interactions", "https://pubmed.ncbi.nlm.nih.gov/36606424/"),
                    _evidence("PLOS Biology 2022: CCDC66 mitotic/cytokinetic microtubule organization", "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001708"),
                    _evidence("PMCID PMC12324684: CCDC66 cilia stability, signaling, and epithelial organization", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12324684/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Dekomien et al., 2010 (PMID 19777273)",
                        "genotypes": "Homozygous CCDC66 1-bp insertion in affected Schapendoes dogs; heterozygous obligatory carriers",
                        "phenotype": "Autosomal-recessive generalized progressive retinal atrophy in dogs",
                        "finding": "The study identified CCDC66 as the causal locus in Schapendoes gPRA and reported photoreceptor inner-segment CCDC66 protein localization in mouse, dog, and human retina.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/19777273/",
                    },
                    {
                        "paper": "Gerding et al., 2011 (PMID 21680557)",
                        "genotypes": "Homozygous Ccdc66 null/gene-trap mouse model",
                        "phenotype": "Retinal degeneration, rod-cone dysplasia, and electroretinography dysfunction",
                        "finding": "The study showed that loss of Ccdc66 RNA/protein in mouse retina causes early, slow progressive photoreceptor degeneration and functional retinal deficits.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/21680557/",
                    },
                    {
                        "paper": "Conkar et al., 2017 (PMID 28235840)",
                        "genotypes": "CCDC66 depletion and rescue in ciliated cell models",
                        "phenotype": "Cilium formation, BBS4 trafficking, CEP290 interaction, and centriolar-satellite organization",
                        "finding": "The study identified CCDC66 as a microtubule-associated centriolar-satellite/ciliary protein that interacts with CEP290 and supports ciliogenesis and ciliary trafficking.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5399785/",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from CCDC66 gene, ClinGen, Human Protein Atlas, high-myopia sequencing, retinal degeneration model, ciliogenesis, cilium-stability, and mitotic microtubule literature.",
        "population_coverage_note": (
            "The bundled CCDC66 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because CCDC66 human clinical validity is still emerging, interpretation should prioritize current ClinVar/gnomAD status, transcript, variant consequence, zygosity, segregation, ophthalmic phenotype, retinal expression, assay type, and model-system versus human evidence boundaries."
        ),
        "population_sources": [
            _evidence("NCBI Gene 285331: CCDC66 expression, GO context, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/285331"),
            _evidence("ClinGen Gene Facts HGNC:27709: CCDC66 coordinates, function, constraint, and no published curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27709"),
            _evidence("Human Protein Atlas: CCDC66 protein and expression context", "https://www.proteinatlas.org/ENSG00000180376-CCDC66"),
            _evidence("PubMed 37852749: CCDC66 high-myopia sequencing and mitosis evidence", "https://pubmed.ncbi.nlm.nih.gov/37852749/"),
            _evidence("PubMed 21680557: Ccdc66 null retinal degeneration mouse model", "https://pubmed.ncbi.nlm.nih.gov/21680557/"),
            _evidence("PMCID PMC5399785: CCDC66 CEP290/ciliary trafficking study", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5399785/"),
            _evidence("PMCID PMC12324684: CCDC66 cilia stability and epithelial organization", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12324684/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "CCDC66 expression and protein context",
                "location_group": "Global pattern",
                "summary": "NCBI reports ubiquitous expression including testis and endometrium, while ClinGen/HPA/UniProt resources frame CCDC66 as a microtubule-binding ciliogenesis protein with centriolar-satellite, ciliary, cytoskeletal, photoreceptor, and mitotic annotations.",
            },
            {
                "variant": "CCDC66 c.C172T / p.Q58X",
                "location_group": "Human high-myopia sequencing",
                "summary": "A 2024 high-myopia study reported a suspected CCDC66 nonsense variant co-segregating with disease in one family and proposed a retinal-growth mechanism through affected mitosis and microtubule biology.",
            },
            {
                "variant": "CCDC66 loss-of-function retinal degeneration model",
                "location_group": "Animal retinal models",
                "summary": "Canine CCDC66 frameshift/insertion and mouse Ccdc66 null data support a photoreceptor degeneration and retinal dysfunction mechanism, but translation to human clinical diagnosis remains variant- and evidence-dependent.",
            },
            {
                "variant": "CCDC66 ciliary trafficking and signaling",
                "location_group": "Functional biology",
                "summary": "Cell studies support CCDC66 roles in CEP290/PCM1 satellite biology, BBS4 trafficking, transition-zone/IFT-B recruitment, cilium length/stability/disassembly, Hedgehog/Wnt signaling, and epithelial organization.",
            },
            {
                "variant": "CCDC66 mitotic/cytokinetic microtubules",
                "location_group": "Functional biology",
                "summary": "CCDC66 depletion disrupts spindle assembly, chromosome alignment, central spindle and midbody organization, and cytokinesis, providing a cytoskeletal mechanism relevant to retinal growth and broader cell-biology interpretation.",
            },
            {
                "variant": "CCDC66 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body regulatory context, but no population methylation threshold is bundled for high myopia, retinal degeneration, ciliopathy, mitotic, epithelial, or cancer interpretation.",
            },
        ],
    },
    {
        "gene_name": "TYW5",
        "cytoband": "2q33.1",
        "chromosome": "2",
        "start": 200793636,
        "end": 200820214,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 129450 reports TYW5 on GRCh37.p13 at NC_000002.11:200793636..200820214, "
            "complement, and on GRCh38.p14 at NC_000002.12:199928913..199955491, complement; "
            "ClinGen Gene Facts HGNC:26754 reports GRCh37 chr2:200793636-200820214 and GRCh38 chr2:199928913-199955491"
        ),
        "manifest_filter_region": "2:200793636-200821214",
        "gene_summary": (
            "TYW5, also known as C2orf60 or hTYW5, encodes tRNA-yW synthesizing protein 5, a JmjC-domain Fe(II)/2-oxoglutarate-dependent RNA hydroxylase. "
            "NCBI, UniProt, ClinGen Gene Facts, Human Protein Atlas, and structural enzymology literature frame TYW5 as a tRNA(Phe) wybutosine-pathway enzyme that hydroxylates yW-72 into undermodified hydroxywybutosine, supports tRNA modification biology, binds iron, homodimerizes, and has broad human expression including brain and retina-related single-cell contexts."
        ),
        "clinical_context": (
            "The local TYW5 knowledge base is tRNA modification, RNA hydroxylase, schizophrenia regulatory-expression, neurodevelopment, dendritic spine, structural-MRI, and translational-fidelity research oriented. "
            "ClinGen has not published a TYW5 gene-disease validity or dosage curation, so TYW5 findings should remain exploratory unless external clinical review supplies stronger variant-level evidence."
        ),
        "variant_effect_overview": [
            "Biochemical and structural work identifies TYW5 as a human JmjC RNA hydroxylase that uses Fe(II) and 2-oxoglutarate to form hydroxywybutosine-pathway intermediates at position 37 of cytoplasmic tRNA(Phe), a modification class important for codon reading-frame fidelity.",
            "The strongest human genetics evidence is schizophrenia regulatory-expression literature at 2q33.1: rs796364 and rs281759 disrupt CTCF/RAD21/FOXP2 binding, physically interact with TYW5, associate with brain TYW5 expression, and were functionally validated with reporter, EMSA, CRISPR-deletion, expression, and neural-cell assays.",
            "An independent integrative schizophrenia study prioritized TYW5 with Sherlock and SMR analyses, reported rs203772 association with higher prefrontal-cortex TYW5 expression, found higher TYW5 expression in schizophrenia brain and induced-neuron datasets, and linked rs203772 to gray-matter volume differences in right middle frontal gyrus and left precuneus.",
            "Functional schizophrenia studies remain cohort- and expression-context dependent: TYW5 overexpression altered neural stem cell proliferation/differentiation and dendritic spine density, but these data support a research mechanism rather than a deterministic psychiatric diagnosis.",
        ],
        "condition_research_overview": [
            "tRNA(Phe) wybutosine and hydroxywybutosine biosynthesis, JmjC Fe(II)/2-oxoglutarate RNA hydroxylase activity, tRNA binding, homodimer structure, translational fidelity, and codon-frame maintenance.",
            "Schizophrenia GWAS/eQTL/SMR integration, 2q33.1 regulatory variants, TYW5 brain expression, neurodevelopment, neural stem cell proliferation/differentiation, dendritic spine morphogenesis, and structural MRI research.",
            "Broad expression, mitochondrial/subcellular localization evidence from HPA, cone/rod photoreceptor-enriched single-cell RNA context, and exploratory liver hepatocellular carcinoma prognostic-expression context.",
        ],
        "methylation_interpretation": (
            "TYW5 has a promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use TYW5 methylation as local regulatory context around a reverse-strand tRNA-modification and schizophrenia-eQTL gene, not as a validated schizophrenia, neurodevelopmental, RNA-modification, retinal, cancer, or structural-MRI biomarker."
        ),
        "methylation_effects": [
            "Promoter-proximal and gene-body methylation may help frame local TYW5 regulatory context in samples where sequence, expression, brain-eQTL, or RNA-modification evidence is also available.",
            "No source-backed TYW5 methylation threshold is bundled; beta values should be interpreted alongside tissue, developmental timing, cell composition, genotype dosage, ancestry, brain-expression evidence, psychiatric phenotype, and current GWAS/eQTL evidence.",
        ],
        "methylation_condition_research": [
            "Use TYW5 methylation as supportive local regulatory context in tRNA modification, schizophrenia genetics, brain expression, neurodevelopment, dendritic spine, structural-MRI, retinal expression, or cancer-expression research.",
            "Do not infer schizophrenia, neurodevelopmental disorder status, tRNA hydroxywybutosine activity, translation fidelity, gray-matter volume, retinal function, or liver cancer prognosis from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 129450: TYW5 gene summary, GRCh37/GRCh38 coordinates, expression, RefSeq, GO, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/129450"),
            _evidence("Ensembl GRCh37 ENSG00000162971: TYW5 gene model and reverse-strand coordinates", "https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000162971"),
            _evidence("UniProt A2RUC4: TYW5 / tRNA wybutosine-synthesizing protein 5 entry", "https://www.uniprot.org/uniprotkb/A2RUC4/entry"),
            _evidence("Human Protein Atlas ENSG00000162971-TYW5: expression, subcellular localization, protein class, and UniProt function context", "https://www.proteinatlas.org/ENSG00000162971-TYW5"),
            _evidence("ClinGen Gene Facts HGNC:26754: TYW5 coordinates, MANE transcript, constraint, function, and no published ClinGen curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A26754"),
            _evidence("PubMed 20739293: TYW5 as a JmjC RNA hydroxylase in hydroxywybutosine biosynthesis", "https://pubmed.ncbi.nlm.nih.gov/20739293/"),
            _evidence("PubMed 20972222 / PMCID PMC3045595: human TYW5 crystal structure, Fe(II)/2-OG JmjC fold, dimer, and tRNA-binding residues", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3045595/"),
            _evidence("PubMed 34581804 / PMCID PMC9014752: rs796364 and rs281759 regulate distal TYW5 expression and schizophrenia risk at 2q33.1", "https://pubmed.ncbi.nlm.nih.gov/34581804/"),
            _evidence("PubMed 35527273 / PMCID PMC9082878: integrative analyses identify TYW5 as a schizophrenia risk gene and rs203772 expression/MRI marker", "https://pubmed.ncbi.nlm.nih.gov/35527273/"),
        ],
        "variants": [
            {
                "variant": "rs796364 / rs281759",
                "display_name": "rs796364 / rs281759 (TYW5 2q33.1 regulatory schizophrenia locus)",
                "common_name": "TYW5 distal enhancer schizophrenia eQTL marker pair",
                "position": None,
                "lookup_keys": [
                    "rs796364",
                    "TYW5 rs796364",
                    "TYW5:rs796364",
                    "rs281759",
                    "TYW5 rs281759",
                    "TYW5:rs281759",
                    "chr2_200825237_I",
                    "2q33.1 TYW5 schizophrenia locus",
                    "TYW5 schizophrenia regulatory locus",
                    "TYW5 distal enhancer",
                ],
                "region_class": "distal_regulatory",
                "interpretation_scope": "Schizophrenia GWAS/eQTL regulatory marker pair / TYW5 brain-expression and neurodevelopment research context",
                "clinical_interpretation": (
                    "rs796364 and rs281759 are curated as a functional 2q33.1 regulatory marker pair for TYW5 schizophrenia research. "
                    "Li et al. reported that the two linked noncoding variants disrupt CTCF/RAD21/FOXP2 binding, show enhancer activity, physically interact with TYW5, associate with TYW5 expression in human brain eQTL datasets, alter TYW5 expression after CRISPR deletion of surrounding regulatory sequence, and place TYW5 upregulation in schizophrenia brain expression context."
                ),
                "clinical_significance": "Research-level schizophrenia regulatory-expression markers; not diagnostic psychiatric variants and not ClinGen-curated pathogenic alleles.",
                "functional_effects": [
                    "rs796364 disrupted CTCF and RAD21 binding in the cited functional genomics study.",
                    "rs281759 disrupted FOXP2 binding in the cited functional genomics study.",
                    "Both variants physically interacted with TYW5, showed strong brain eQTL association with TYW5 expression, and regulatory-region deletion decreased TYW5 expression in cell models.",
                    "TYW5 overexpression in neural stem cells and rat primary neurons altered proliferation, differentiation, and dendritic spine density in the cited work.",
                ],
                "associated_conditions": [
                    "Schizophrenia GWAS and eQTL research",
                    "Neurodevelopment and neural stem cell differentiation",
                    "Dendritic spine morphogenesis and maturity",
                    "2q33.1 regulatory genomics",
                ],
                "research_context": [
                    "Interpret rs796364/rs281759 as noncoding regulatory and LD-dependent markers, not as deterministic disease variants.",
                    "Prioritize ancestry, genotype dosage, imputation quality, brain-expression direction, local haplotype context, psychiatric phenotype, and current GWAS/eQTL resources.",
                    "Because TYW5 has no published ClinGen disease curation, use this record for research synthesis and external review only.",
                ],
                "usual_variant_note": "Linked 2q33.1 schizophrenia regulatory marker pair that modulates distal TYW5 expression in brain-eQTL and cell-model studies.",
                "methylation_interpretation": (
                    "TYW5 methylation may add local regulatory context but does not establish rs796364/rs281759 enhancer activity, brain TYW5 expression, schizophrenia risk, or dendritic-spine phenotypes in an individual sample."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 34581804: rs796364 and rs281759 regulatory mechanism for TYW5 and schizophrenia", "https://pubmed.ncbi.nlm.nih.gov/34581804/"),
                    _evidence("PMCID PMC9014752: full text functional genomics and TYW5 neurodevelopment/spine evidence", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9014752/"),
                    _evidence("ClinGen Gene Facts HGNC:26754: no published TYW5 clinical curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A26754"),
                ],
                "literature_findings": [
                    {
                        "paper": "Li et al., 2022 (PMID 34581804)",
                        "genotypes": "rs796364 and rs281759 at 2q33.1, in high linkage disequilibrium and downstream of TYW5",
                        "phenotype": "Schizophrenia risk, brain TYW5 expression, neurodevelopment, and dendritic spine morphology",
                        "finding": "The study reported that rs796364 and rs281759 regulate TYW5 expression through transcription-factor binding and chromatin interaction, and that TYW5 dysregulation affects neurodevelopmental and dendritic-spine phenotypes.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/34581804/",
                    }
                ],
            },
            {
                "variant": "rs203772",
                "display_name": "rs203772 (TYW5 schizophrenia eQTL/MRI marker)",
                "common_name": "TYW5 prefrontal-expression schizophrenia marker",
                "position": None,
                "lookup_keys": [
                    "rs203772",
                    "TYW5 rs203772",
                    "TYW5:rs203772",
                    "rs203772 TYW5 expression",
                    "rs203772 schizophrenia",
                    "rs203772 gray matter volume",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Schizophrenia integrative GWAS/eQTL and structural-MRI marker / brain-expression research context",
                "clinical_interpretation": (
                    "rs203772 is included as an integrative schizophrenia genetics marker because Zhang et al. reported that the risk allele was associated with higher TYW5 mRNA expression in prefrontal cortex, that TYW5 expression/protein evidence associated with schizophrenia risk in Sherlock and SMR analyses, and that rs203772 associated with gray-matter volume in right middle frontal gyrus and left precuneus in first-episode antipsychotic-naive schizophrenia patients. "
                    "This remains a research-level association signal rather than a psychiatric diagnosis."
                ),
                "clinical_significance": "Research-level schizophrenia eQTL and structural-MRI marker; not a diagnostic psychiatric variant.",
                "functional_effects": [
                    "Risk allele associated with higher TYW5 transcriptional level in prefrontal cortex in the cited study.",
                    "TYW5 expression was higher in schizophrenia brain tissues and induced-neuron datasets compared with controls in the cited work.",
                    "rs203772 genotype was associated with gray-matter volume differences in right middle frontal gyrus and left precuneus in a structural-MRI analysis.",
                ],
                "associated_conditions": [
                    "Schizophrenia GWAS/eQTL integration",
                    "Prefrontal cortex TYW5 expression",
                    "Gray-matter volume and neuroimaging genetics",
                    "Brain expression and induced-neuron research",
                ],
                "research_context": [
                    "Use rs203772 as a cohort-level brain eQTL and MRI marker rather than a deterministic disease rule.",
                    "Interpret with ancestry, genotype dosage, expression dataset, psychiatric phenotype, medication status, MRI phenotype, and current GWAS resources.",
                ],
                "usual_variant_note": "TYW5 schizophrenia eQTL/MRI marker from integrative GWAS, brain-expression, proteomic, and imaging analyses.",
                "methylation_interpretation": (
                    "TYW5 methylation can provide local regulatory context, but it does not substitute for rs203772 genotype, DLPFC expression, protein abundance, or MRI phenotype data."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 35527273: rs203772, TYW5 expression, schizophrenia risk, and structural MRI", "https://pubmed.ncbi.nlm.nih.gov/35527273/"),
                    _evidence("PMCID PMC9082878: full text integrative TYW5 schizophrenia risk-gene analysis", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9082878/"),
                    _evidence("NCBI Gene 129450: TYW5 gene and bibliography context", "https://www.ncbi.nlm.nih.gov/gene/129450"),
                ],
                "literature_findings": [
                    {
                        "paper": "Zhang et al., 2022 (PMID 35527273)",
                        "genotypes": "rs203772 risk allele and TYW5 expression in brain eQTL/integrative schizophrenia analyses",
                        "phenotype": "Schizophrenia risk, prefrontal-cortex expression, induced-neuron expression, protein abundance, and gray-matter volume",
                        "finding": "The study prioritized TYW5 as a schizophrenia risk gene and linked rs203772 to higher TYW5 expression and gray-matter volume differences in first-episode schizophrenia MRI data.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/35527273/",
                    }
                ],
            },
            {
                "variant": "TYW5 enzymatic wybutosine-hydroxylase model",
                "display_name": "TYW5 JmjC tRNA(Phe) hydroxywybutosine model",
                "common_name": "TYW5 Fe(II)/2-OG RNA hydroxylase model",
                "position": None,
                "lookup_keys": [
                    "TYW5 enzymatic model",
                    "TYW5 loss-of-function",
                    "TYW5 loss of function",
                    "TYW5 LOF",
                    "TYW5 overexpression",
                    "TYW5 knockdown",
                    "TYW5 JmjC",
                    "TYW5 RNA hydroxylase",
                    "TYW5 tRNA hydroxylase",
                    "TYW5 wybutosine",
                    "TYW5 hydroxywybutosine",
                    "TYW5 OHyW",
                    "TYW5 yW-72",
                    "TYW5 Fe(II)",
                    "TYW5 2-oxoglutarate",
                    "TYW5 tRNA(Phe)",
                ],
                "region_class": "functional_model",
                "interpretation_scope": "RNA-modification enzyme and expression model / tRNA(Phe) wybutosine pathway, translation-fidelity, and neurodevelopment research context",
                "clinical_interpretation": (
                    "TYW5 enzymatic evidence is curated as pathway and functional biology rather than a specific pathogenic allele. "
                    "Noma et al. identified human TYW5 as a JmjC-domain RNA hydroxylase that catalyzes hydroxywybutosine biosynthesis in tRNA(Phe), and Kato et al. solved human TYW5 structures showing the Fe(II)/2-oxoglutarate JmjC fold, dimerization, and positively charged tRNA-binding surface with essential Arg residues."
                ),
                "clinical_significance": "Functional tRNA-modification and RNA-hydroxylase research model; not a diagnostic germline variant.",
                "functional_effects": [
                    "Catalyzes hydroxylation of 7-(alpha-amino-alpha-carboxypropyl)wyosine into the hydroxywybutosine-pathway intermediate OHyW*.",
                    "Requires Fe(II) and 2-oxoglutarate as cofactors in the JmjC oxygenase mechanism.",
                    "Human TYW5 forms a homodimer and presents a positively charged tRNA-binding surface; structure-based Arg mutants implicated specific residues in tRNA recognition.",
                    "Wybutosine-family tRNA(Phe) modifications help suppress reading-frame errors, tying the enzyme to translational fidelity research.",
                ],
                "associated_conditions": [
                    "tRNA(Phe) wybutosine and hydroxywybutosine biosynthesis",
                    "RNA hydroxylase and JmjC oxygenase biology",
                    "Translational fidelity and codon-frame maintenance",
                    "Neurodevelopment and schizophrenia expression-model research",
                ],
                "research_context": [
                    "Use this record for enzymatic, expression, RNA-modification, or curated functional annotations rather than generic benign SNP rows.",
                    "Pair sequence or methylation signals with expression, tRNA-modification assays, protein function, brain context, and phenotype evidence before drawing biological conclusions.",
                ],
                "usual_variant_note": "Functional TYW5 RNA-hydroxylase model rather than a specific inherited pathogenic variant.",
                "methylation_interpretation": (
                    "TYW5 methylation can support local regulatory review, but it does not directly measure TYW5 protein, Fe(II)/2-OG hydroxylase activity, tRNA(Phe) OHyW status, or translational fidelity."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PubMed 20739293: TYW5 as JmjC RNA hydroxylase for hydroxywybutosine biosynthesis", "https://pubmed.ncbi.nlm.nih.gov/20739293/"),
                    _evidence("PMCID PMC3045595: human TYW5 crystal structure, cofactor binding, dimerization, and tRNA-recognition residues", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3045595/"),
                    _evidence("UniProt A2RUC4: TYW5 enzymatic function and cofactor annotations", "https://www.uniprot.org/uniprotkb/A2RUC4/entry"),
                ],
                "literature_findings": [
                    {
                        "paper": "Noma et al., 2010 (PMID 20739293)",
                        "genotypes": "Human TYW5/Tyw5p enzymatic model",
                        "phenotype": "Hydroxywybutosine biosynthesis in tRNA(Phe)",
                        "finding": "The study identified TYW5 as a JmjC-domain RNA hydroxylase that catalyzes hydroxywybutosine formation in the tRNA(Phe) wybutosine pathway.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/20739293/",
                    },
                    {
                        "paper": "Kato et al., 2011 (PMID 20972222)",
                        "genotypes": "Human TYW5 structural and mutant enzyme analyses",
                        "phenotype": "JmjC fold, Fe(II)/2-OG cofactor context, homodimerization, and tRNA recognition",
                        "finding": "The study solved human TYW5 structures and used structure-based mutants to identify Arg residues important for tRNA recognition.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3045595/",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from TYW5 gene, ClinGen, Human Protein Atlas, tRNA-modification enzymology, schizophrenia GWAS/eQTL, neurodevelopment, dendritic-spine, and structural-MRI literature.",
        "population_coverage_note": (
            "The bundled TYW5 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because TYW5 has no published ClinGen clinical curation and schizophrenia risk is highly polygenic, interpretation should prioritize current GWAS/eQTL evidence, ancestry, genotype dosage, tissue expression, psychiatric phenotype, brain or RNA-modification context, and model-system versus human evidence boundaries."
        ),
        "population_sources": [
            _evidence("NCBI Gene 129450: TYW5 expression, genomic context, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/129450"),
            _evidence("ClinGen Gene Facts HGNC:26754: no published TYW5 clinical curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A26754"),
            _evidence("Human Protein Atlas: TYW5 protein class, mitochondria localization, broad RNA expression, and photoreceptor-enriched single-cell context", "https://www.proteinatlas.org/ENSG00000162971-TYW5"),
            _evidence("PMCID PMC3045595: TYW5 JmjC structure and tRNA-modification mechanism", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3045595/"),
            _evidence("PubMed 34581804: rs796364/rs281759 TYW5 regulatory schizophrenia study", "https://pubmed.ncbi.nlm.nih.gov/34581804/"),
            _evidence("PubMed 35527273: TYW5 schizophrenia integrative expression and MRI study", "https://pubmed.ncbi.nlm.nih.gov/35527273/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "TYW5 expression and protein context",
                "location_group": "Global pattern",
                "summary": "NCBI reports ubiquitous expression and cytoplasmic prediction, while Human Protein Atlas frames TYW5 as an intracellular enzyme with protein evidence, mitochondria subcellular localization, low tissue specificity, and single-cell enrichment in cone and rod photoreceptor cells.",
            },
            {
                "variant": "TYW5 enzymatic hydroxywybutosine model",
                "location_group": "Functional biology",
                "summary": "Structural and biochemical literature supports TYW5 as a homodimeric JmjC Fe(II)/2-oxoglutarate RNA hydroxylase that modifies tRNA(Phe) in the hydroxywybutosine pathway.",
            },
            {
                "variant": "rs796364 / rs281759",
                "location_group": "Schizophrenia regulatory genetics",
                "summary": "The linked rs796364/rs281759 marker pair is a functional 2q33.1 schizophrenia regulatory signal that alters transcription-factor binding, interacts with TYW5, and modulates brain TYW5 expression in the cited studies.",
            },
            {
                "variant": "rs203772",
                "location_group": "Schizophrenia eQTL and imaging genetics",
                "summary": "rs203772 is an integrative schizophrenia marker associated with TYW5 expression and structural-MRI phenotypes in the cited BMC Medicine study, but it remains a cohort-level research marker.",
            },
            {
                "variant": "TYW5 neurodevelopment and dendritic spine expression model",
                "location_group": "Functional biology",
                "summary": "TYW5 overexpression altered neural stem cell proliferation/differentiation and dendritic spine density in schizophrenia-motivated model systems, supporting a pathway mechanism rather than deterministic individual prediction.",
            },
            {
                "variant": "TYW5 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body regulatory context, but no population methylation threshold is bundled for schizophrenia, RNA modification, neurodevelopment, retina, MRI, or cancer-expression interpretation.",
            },
        ],
    },
    {
        "gene_name": "ELOVL7",
        "cytoband": "5q12.1",
        "chromosome": "5",
        "start": 60047618,
        "end": 60140096,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 79993 reports ELOVL7 on GRCh37.p13 at NC_000005.9:60047618..60140096, "
            "complement, and on GRCh38.p14 at NC_000005.10:60751791..60844269, complement; "
            "Ensembl GRCh37 reports ENSG00000164181 at chr5:60047618-60140216 on the reverse strand"
        ),
        "manifest_filter_region": "5:60047618-60141096",
        "gene_summary": (
            "ELOVL7 encodes elongation of very long chain fatty acids protein 7, an endoplasmic-reticulum membrane elongase in the ELOVL family. "
            "It catalyzes the first, rate-limiting condensation step of the long-chain and very-long-chain fatty-acid elongation cycle and has high activity toward C18 acyl-CoA substrates, supporting saturated and polyunsaturated VLCFA production for membrane lipids, sphingolipids, and lipid signaling contexts."
        ),
        "clinical_context": (
            "The local ELOVL7 knowledge base is lipid-metabolism, prostate-cancer, viral lipid-remodeling, liver-fibrosis, necroptosis, and neurogenetics research oriented. "
            "ELOVL7 is not bundled here as a deterministic diagnostic gene; even in multiple-system-atrophy follow-up sequencing, rare coding variants and copy-number changes did not significantly explain disease risk."
        ),
        "variant_effect_overview": [
            "Biochemical and structural literature supports ELOVL7 as a membrane-bound fatty-acid elongase that preferentially processes C18 acyl-CoA substrates and participates in VLCFA elongation.",
            "In prostate-cancer studies, ELOVL7 is androgen/SREBP1-regulated, overexpressed in tumor contexts, and linked to saturated very-long-chain fatty acid metabolism, cholesterol ester biology, and castration-resistant prostate-cancer growth.",
            "HCMV and necroptosis studies show that ELOVL7-driven VLCFA remodeling can affect virion infectivity, membrane permeabilization, and cell-death biology, while liver-fibrosis work frames ELOVL7 as a hepatic stellate-cell and advanced-fibrosis expression marker.",
            "The intronic rs7715147 locus is retained as an MSA GWAS-interest marker, but follow-up work found ELOVL7 coding variants to be extremely rare and not significantly associated with MSA risk.",
        ],
        "condition_research_overview": [
            "Very-long-chain fatty acid elongation, saturated/polyunsaturated lipid metabolism, sphingolipid and membrane lipid remodeling.",
            "Prostate cancer, androgen receptor and SREBP1-regulated lipid biosynthesis, and castration-resistant prostate-cancer progression.",
            "Human cytomegalovirus replication, necroptotic membrane permeabilization, hepatic stellate-cell activation, chronic hepatitis B liver fibrosis, and synucleinopathy/MSA locus research.",
        ],
        "methylation_interpretation": (
            "ELOVL7 has a bundled promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use ELOVL7 methylation as local regulatory context around a reverse-strand ER lipid-elongase gene, not as a validated prostate-cancer, viral replication, liver-fibrosis, necroptosis, or MSA biomarker."
        ),
        "methylation_effects": [
            "The ELOVL7 methylation view can summarize promoter-proximal and early-gene-body CpGs around the reverse-strand transcription start.",
            "No source-backed ELOVL7 methylation threshold is bundled; beta values should be interpreted alongside tissue, cell composition, lipidomic readouts, expression, sequence/CNV evidence, and disease context.",
        ],
        "methylation_condition_research": [
            "Use ELOVL7 methylation as supportive regulatory context in lipid-metabolism, prostate-cancer, viral-replication, liver-fibrosis, necroptosis, or neurogenetic locus research.",
            "Do not infer VLCFA flux, tumor behavior, HCMV replication capacity, fibrosis stage, or MSA risk from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 79993: ELOVL7 gene summary, GRCh37/GRCh38 coordinates, expression, RefSeq, and GO context", "https://www.ncbi.nlm.nih.gov/gene/79993"),
            _evidence("Ensembl GRCh37 ENSG00000164181: ELOVL7 gene model and reverse-strand coordinates", "https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000164181"),
            _evidence("UniProt A1L3X0: ELOVL7 / ELOV7_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A1L3X0/entry"),
            _evidence("Human Protein Atlas ENSG00000164181: ELOVL7 protein expression and membrane/metabolic context", "https://www.proteinatlas.org/ENSG00000164181"),
            _evidence("PubMed 19826053: ELOVL7 in prostate-cancer growth and saturated long-chain fatty-acid metabolism", "https://pubmed.ncbi.nlm.nih.gov/19826053/"),
            _evidence("Oncogene 2018: AR-regulated lipid biosynthesis and ELOVL7 in castration-resistant prostate cancer", "https://www.nature.com/articles/onc2017385"),
            _evidence("PMCID PMC4354725: ELOVL7 lipidome remodeling essential for HCMV replication", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4354725/"),
            _evidence("PubMed 31490656: ELOVL7, VLCFA accumulation, necroptosis, and membrane permeabilization", "https://pubmed.ncbi.nlm.nih.gov/31490656/"),
            _evidence("PubMed 36930494: ELOVL7 and advanced liver fibrosis in chronic hepatitis B", "https://pubmed.ncbi.nlm.nih.gov/36930494/"),
            _evidence("PMCID PMC7983070: ELOVL7 coding variants in multiple system atrophy", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7983070/"),
        ],
        "variants": [
            {
                "variant": "rs7715147",
                "display_name": "rs7715147 (ELOVL7 intronic MSA locus marker)",
                "common_name": "ELOVL7 MSA GWAS-interest intronic marker",
                "position": None,
                "lookup_keys": [
                    "rs7715147",
                    "ELOVL7:rs7715147",
                    "ELOVL7 rs7715147",
                    "MSA ELOVL7 rs7715147",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Intronic GWAS-interest marker / synucleinopathy lipid-metabolism research context",
                "clinical_interpretation": (
                    "rs7715147 is curated as an ELOVL7 intronic marker highlighted by prior MSA GWAS-interest evidence. "
                    "A follow-up study sequencing ELOVL7 exons and assessing exon 7 copy number in pathologically confirmed MSA cases found coding variants to be extremely rare and did not observe a significant association between ELOVL7 coding variants or CNV and MSA risk."
                ),
                "clinical_significance": "Research-level MSA locus marker; not a diagnostic MSA, Parkinson disease, or spinocerebellar ataxia allele.",
                "functional_effects": [
                    "Intronic locus marker; no direct coding effect is bundled.",
                    "Interpretation is driven by lipid-dyshomeostasis plausibility and GWAS/locus context rather than by a proven ELOVL7 protein-altering mechanism.",
                ],
                "associated_conditions": [
                    "Multiple system atrophy locus research",
                    "Synucleinopathy and lipid-dyshomeostasis studies",
                    "Very-long-chain fatty acid metabolism",
                ],
                "research_context": [
                    "Use this marker as locus context only; do not infer ELOVL7 coding dysfunction without additional variant, expression, or lipidomic evidence.",
                    "The bundled follow-up evidence argues against rare ELOVL7 coding variants or copy-number mutation as a major MSA driver in the studied cohorts.",
                ],
                "usual_variant_note": "Intronic ELOVL7 MSA GWAS-interest marker from locus-level neurogenetics literature.",
                "methylation_interpretation": (
                    "ELOVL7 methylation may add regulatory context but does not turn rs7715147 into a protein-function or disease-prediction result."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PMCID PMC7983070: ELOVL7 coding variants in MSA follow-up study", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7983070/"),
                    _evidence("NCBI Gene 79993: ELOVL7 locus and variation resources", "https://www.ncbi.nlm.nih.gov/gene/79993"),
                ],
                "literature_findings": [
                    {
                        "paper": "Wernick et al., 2021 (PMID 33600908)",
                        "genotypes": "ELOVL7 intronic rs7715147 GWAS-interest context plus coding-exon and copy-number follow-up",
                        "phenotype": "Multiple system atrophy and synucleinopathy genetics",
                        "finding": "The study reported that ELOVL7 coding variants were extremely rare and did not significantly associate with MSA risk, keeping rs7715147 in locus-level research context rather than diagnostic interpretation.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7983070/",
                    }
                ],
            },
            {
                "variant": "ELOVL7 functional lipid-elongation model",
                "display_name": "ELOVL7 expression or knockdown lipid-elongation model",
                "common_name": "ELOVL7 VLCFA remodeling functional model",
                "position": None,
                "lookup_keys": [
                    "ELOVL7 functional model",
                    "ELOVL7 knockdown",
                    "ELOVL7 overexpression",
                    "ELOVL7 upregulation",
                    "ELOVL7 lipid elongation",
                    "ELOVL7 VLCFA remodeling",
                    "ELOVL7 prostate cancer",
                    "ELOVL7 HCMV",
                    "ELOVL7 liver fibrosis",
                    "ELOVL7 necroptosis",
                ],
                "region_class": "functional_model",
                "interpretation_scope": "Expression/functional model / VLCFA metabolism, cancer, infection, fibrosis, and cell-death research context",
                "clinical_interpretation": (
                    "ELOVL7 functional evidence is strong at the pathway level: biochemical work shows C18 acyl-CoA elongase activity, prostate-cancer studies connect ELOVL7 to androgen/SREBP1-regulated saturated VLCFA metabolism and tumor growth, HCMV induces ELOVL7 to remodel virion lipids, and necroptosis studies connect ELOVL7-dependent VLCFA accumulation to membrane permeabilization. "
                    "This record is not a specific human pathogenic allele and should be matched only to expression, model, or curated functional annotations."
                ),
                "clinical_significance": "Functional lipid-metabolism research model; not a diagnostic germline variant.",
                "functional_effects": [
                    "Purified ELOVL7 shows high activity toward C18 acyl-CoA substrates in VLCFA elongation assays.",
                    "Knockdown attenuates prostate-cancer cell growth and impairs HCMV lipid elongation, particle release, and infectivity in cited studies.",
                    "ELOVL7-driven VLCFA accumulation can promote necroptotic membrane permeabilization in cell models.",
                    "Expression was associated with hepatic stellate-cell activation and advanced liver-fibrosis prediction in a chronic hepatitis B study.",
                ],
                "associated_conditions": [
                    "Prostate cancer and castration-resistant prostate cancer lipid metabolism",
                    "Human cytomegalovirus replication",
                    "Necroptosis and membrane permeabilization",
                    "Advanced liver fibrosis in chronic hepatitis B",
                    "Very-long-chain fatty acid and sphingolipid metabolism",
                ],
                "research_context": [
                    "Use this marker for model-level or expression-level evidence, not generic SNP rows.",
                    "Pair sequence calls with expression, lipidomics, tumor context, infection model, fibrosis stage, or cell-death phenotype before drawing conclusions.",
                ],
                "usual_variant_note": "Functional ELOVL7 expression/knockdown model for VLCFA remodeling rather than a specific inherited variant.",
                "methylation_interpretation": (
                    "ELOVL7 methylation can support local regulatory review, but it does not directly quantify VLCFA elongase activity."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PubMed 19826053: ELOVL7 prostate-cancer saturated lipid metabolism", "https://pubmed.ncbi.nlm.nih.gov/19826053/"),
                    _evidence("Oncogene 2018: ELOVL7 silencing and CRPC xenograft regression context", "https://www.nature.com/articles/onc2017385"),
                    _evidence("PMCID PMC4354725: ELOVL7 and HCMV lipidome remodeling", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4354725/"),
                    _evidence("PubMed 31490656: ELOVL7 and necroptotic membrane disruption", "https://pubmed.ncbi.nlm.nih.gov/31490656/"),
                    _evidence("PubMed 36930494: ELOVL7 as advanced liver fibrosis marker", "https://pubmed.ncbi.nlm.nih.gov/36930494/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Tamura et al., 2009 (PMID 19826053)",
                        "genotypes": "ELOVL7 overexpression or knockdown in prostate-cancer models",
                        "phenotype": "Prostate cancer growth, saturated VLCFA metabolism, and androgen-related lipid biology",
                        "finding": "The study reported ELOVL7 overexpression in prostate cancer and growth attenuation after ELOVL7 knockdown, linking the enzyme to saturated long-chain fatty-acid metabolism and tumor growth.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/19826053/",
                    },
                    {
                        "paper": "Purdy et al., 2015 (PMID 25732827)",
                        "genotypes": "ELOVL7 induction, knockdown, and overexpression in HCMV infection models",
                        "phenotype": "HCMV lipidome remodeling, particle release, and infectivity",
                        "finding": "The study reported strong HCMV-induced ELOVL7 expression and showed that ELOVL7 knockdown impaired fatty-acid elongation, particle release, and infectivity per particle.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4354725/",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from ELOVL7 fatty-acid elongase, prostate-cancer, HCMV, liver-fibrosis, necroptosis, and MSA-locus literature.",
        "population_coverage_note": (
            "The bundled ELOVL7 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Interpretation should prioritize current ClinVar/gnomAD status for sequence calls, tissue expression, lipidomics, tumor or infection context, fibrosis phenotype, and the negative coding-variant/CNV follow-up evidence in MSA."
        ),
        "population_sources": [
            _evidence("NCBI Gene 79993: ELOVL7 expression, GO context, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/79993"),
            _evidence("Human Protein Atlas: ELOVL7 expression and protein-function context", "https://www.proteinatlas.org/ENSG00000164181"),
            _evidence("PubMed 19826053: ELOVL7 prostate-cancer lipid metabolism", "https://pubmed.ncbi.nlm.nih.gov/19826053/"),
            _evidence("PMCID PMC4354725: ELOVL7 HCMV lipid remodeling", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4354725/"),
            _evidence("PMCID PMC7983070: ELOVL7 MSA coding-variant follow-up", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7983070/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "ELOVL7 expression",
                "location_group": "Global pattern",
                "summary": "NCBI and Human Protein Atlas resources frame ELOVL7 as broadly expressed with prostate and skin expression in NCBI, membrane/metabolic protein annotations, and vessel-prominent cytoplasmic protein staining.",
            },
            {
                "variant": "ELOVL7 prostate-cancer lipid metabolism",
                "location_group": "Cancer cohorts",
                "summary": "Prostate-cancer studies support a lipid-biosynthesis thesis in which ELOVL7 participates in saturated VLCFA metabolism, androgen/SREBP1 pathways, and tumor-growth models.",
            },
            {
                "variant": "ELOVL7 HCMV/necroptosis VLCFA remodeling",
                "location_group": "Functional biology",
                "summary": "Cell-model evidence supports ELOVL7 as a VLCFA remodeling node for HCMV replication and necroptotic membrane permeabilization.",
            },
            {
                "variant": "rs7715147 and rare coding variants",
                "location_group": "Disease cohorts",
                "summary": "ELOVL7 rs7715147 remains an intronic MSA locus-context marker, but follow-up coding-variant and copy-number work did not support ELOVL7 coding variants or CNV as a major MSA risk driver.",
            },
            {
                "variant": "ELOVL7 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body methylation context, but no population methylation threshold is bundled for VLCFA flux, prostate cancer, HCMV replication, fibrosis, necroptosis, or MSA interpretation.",
            },
        ],
    },
    {
        "gene_name": "SH3PXD2B",
        "cytoband": "5q35.1",
        "chromosome": "5",
        "start": 171752185,
        "end": 171881529,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 285590 reports SH3PXD2B on GRCh37.p13 at NC_000005.9:171752185..171881529, "
            "complement, and on GRCh38.p14 at NC_000005.10:172325181..172454525, complement; "
            "Ensembl GRCh37 reports ENSG00000174705 at chr5:171752185-171881527 on the reverse strand"
        ),
        "manifest_filter_region": "5:171752185-171882529",
        "gene_summary": (
            "SH3PXD2B, also known as TKS4, encodes SH3 and PX domain-containing protein 2B, a Src-phosphorylated adaptor with one PX domain and four SH3 domains. "
            "The protein organizes podosome and invadopodia biology, binds phosphoinositides, matrix-remodeling proteins, and NADPH oxidase components, and helps coordinate extracellular-matrix degradation, cell adhesion, migration, ROS localization, and developmental tissue remodeling."
        ),
        "clinical_context": (
            "The local SH3PXD2B knowledge base is rare autosomal-recessive Frank-ter Haar syndrome and overlapping Borrone dermato-cardio-skeletal syndrome oriented, with supporting podosome, collagen-remodeling, ocular, skeletal, cardiac, craniofacial, dermal, fibrosis, and cancer-cell-migration biology. "
            "ClinGen lists a definitive autosomal-recessive SH3PXD2B-Frank-ter Haar syndrome gene-disease validity curation, but app-level findings still require zygosity, phase, variant classification, breakpoint, phenotype, and clinical-grade confirmation review."
        ),
        "variant_effect_overview": [
            "Biallelic SH3PXD2B loss-of-function, splice, or deletion variants disrupt the TKS4 podosome adaptor mechanism and are an established cause of Frank-ter Haar syndrome, including skeletal, ocular, cardiac, craniofacial, and connective-tissue findings.",
            "SH3PXD2B-related Borrone dermato-cardio-skeletal syndrome is best treated as an overlapping FTHS-spectrum phenotype rather than a separate deterministic entity; the cited linkage and fibroblast work showed complete loss of SH3PXD2B in mutation-positive families.",
            "Functional studies support TKS4 as required for full podosome formation, MT1-MMP recruitment, extracellular-matrix degradation, collagen remodeling, and cell migration; animal models support skeletal, ocular, cardiac, dermal fibrosis, and contracture biology.",
            "Single heterozygous, benign, or VUS SH3PXD2B findings should not be upgraded to a rare-disease prediction without a second pathogenic allele, phase/inheritance support, and phenotype consistency.",
        ],
        "condition_research_overview": [
            "Autosomal-recessive Frank-ter Haar syndrome / Ter Haar syndrome and overlapping Borrone dermato-cardio-skeletal syndrome.",
            "Podosome and invadopodia formation, extracellular-matrix degradation, collagen remodeling, developmental skeletal and connective-tissue biology, ocular pressure/glaucoma, and cardiac/craniofacial development.",
            "Cancer-cell migration, EMT-like epithelial plasticity, Src signaling, NADPH oxidase/ROS localization, adipocyte differentiation, and broader cell-adhesion research contexts.",
        ],
        "methylation_interpretation": (
            "SH3PXD2B has a bundled promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use SH3PXD2B methylation as local regulatory context around a reverse-strand podosome-adaptor gene, not as a validated FTHS, BDCS, glaucoma, fibrosis, cancer-invasion, or collagen-remodeling biomarker."
        ),
        "methylation_effects": [
            "The SH3PXD2B methylation view can summarize promoter-proximal and gene-body CpGs around the reverse-strand transcription start.",
            "No source-backed SH3PXD2B methylation threshold is bundled; beta values should be interpreted alongside tissue, cell composition, expression, sequence/CNV evidence, inheritance, and phenotype.",
        ],
        "methylation_condition_research": [
            "Use SH3PXD2B methylation as supportive regulatory context in podosome, collagen-remodeling, connective-tissue, ocular, skeletal, cardiac, or cancer-cell-migration research.",
            "Do not infer biallelic SH3PXD2B loss, Frank-ter Haar syndrome, Borrone dermato-cardio-skeletal syndrome, glaucoma, fibrosis, or invasive cancer behavior from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 285590: SH3PXD2B gene summary, GRCh37/GRCh38 coordinates, expression, RefSeq, GO, and phenotype context", "https://www.ncbi.nlm.nih.gov/gene/285590"),
            _evidence("Ensembl GRCh37 ENSG00000174705: SH3PXD2B gene model and reverse-strand coordinates", "https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000174705"),
            _evidence("UniProt A1X283: SH3PXD2B / SPD2B_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A1X283/entry"),
            _evidence("Human Protein Atlas ENSG00000174705-SH3PXD2B: expression and subcellular localization context", "https://www.proteinatlas.org/ENSG00000174705-SH3PXD2B"),
            _evidence("ClinGen Gene Facts HGNC:29242: definitive SH3PXD2B-Frank-ter Haar syndrome gene-disease validity curation", "https://search.clinicalgenome.org/kb/genes/HGNC%3A29242"),
            _evidence("ClinVar RCV000000214: SH3PXD2B c.76-2A>C / rs775217258 pathogenic splice-acceptor variant", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000000214/"),
            _evidence("PMCID PMC2820172: SH3PXD2B disruption causes skeletal, eye, and cardiac abnormalities of Frank-ter Haar syndrome", "https://pmc.ncbi.nlm.nih.gov/articles/PMC2820172/"),
            _evidence("PubMed 19144821: Tks4/SH3PXD2B is required for functional podosome formation", "https://pubmed.ncbi.nlm.nih.gov/19144821/"),
            _evidence("PMCID PMC3532175: SH3PXD2B exon 13 deletion in FTHS with craniosynostosis and raised intracranial pressure", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3532175/"),
            _evidence("European Journal of Human Genetics 2014: SH3PXD2B mutations cause Borrone dermato-cardio-skeletal syndrome", "https://www.nature.com/articles/ejhg2013229"),
            _evidence("PMCID PMC7790187: sh3pxd2b zebrafish model and collagen-remodeling/fibrosis biology", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7790187/"),
        ],
        "variants": [
            {
                "variant": "SH3PXD2B c.76-2A>C",
                "display_name": "SH3PXD2B c.76-2A>C splice acceptor",
                "common_name": "rs775217258 / FTHS splice-acceptor variant",
                "position": 171849502,
                "lookup_keys": [
                    "rs775217258",
                    "SH3PXD2B c.76-2A>C",
                    "NM_001017995.3:c.76-2A>C",
                    "NM_001017995.2:c.76-2A>C",
                    "NC_000005.9:g.171849502T>G",
                    "5:171849502",
                    "5:171849502:T>G",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic splice-acceptor marker / autosomal-recessive Frank-ter Haar syndrome context",
                "clinical_interpretation": (
                    "ClinVar/OMIM literature curates SH3PXD2B c.76-2A>C / rs775217258 as a pathogenic splice-acceptor variant for Frank-ter Haar syndrome. "
                    "Iqbal et al. reported homozygosity for this splice-site change in an affected male patient; unaffected parents were heterozygous and the variant was absent from 210 control chromosomes in that report."
                ),
                "clinical_significance": "Pathogenic Frank-ter Haar syndrome splice-acceptor variant; autosomal-recessive interpretation depends on biallelic dosage, phase, and phenotype.",
                "functional_effects": [
                    "Splice-acceptor disruption at the conserved intronic c.76-2 position is expected to disrupt SH3PXD2B/TKS4 transcript processing.",
                    "The TKS4 disease mechanism centers on impaired podosome/invadopodia organization and extracellular-matrix remodeling.",
                ],
                "associated_conditions": [
                    "Frank-ter Haar syndrome / Ter Haar syndrome",
                    "Borrone dermato-cardio-skeletal syndrome spectrum",
                    "Skeletal dysplasia, ocular anomalies, cardiac findings, craniofacial anomalies, and connective-tissue remodeling",
                ],
                "research_context": [
                    "A single heterozygous call should be treated as carrier context unless a second pathogenic allele is found in trans.",
                    "Check transcript, strand, HGVS normalization, phase, depth, and phenotype before any clinical interpretation.",
                ],
                "usual_variant_note": "Known pathogenic SH3PXD2B splice-acceptor marker in the FTHS/BDCS autosomal-recessive spectrum.",
                "methylation_interpretation": (
                    "SH3PXD2B methylation can provide local regulatory context, but it does not determine whether this splice variant is present, biallelic, or pathogenic in the sample."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar RCV000000214: SH3PXD2B c.76-2A>C / rs775217258 pathogenic classification and GRCh37 coordinate", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000000214/"),
                    _evidence("PMCID PMC2820172: original FTHS SH3PXD2B mutation report", "https://pmc.ncbi.nlm.nih.gov/articles/PMC2820172/"),
                    _evidence("ClinGen Gene Facts HGNC:29242: definitive SH3PXD2B-FTHS curation", "https://search.clinicalgenome.org/kb/genes/HGNC%3A29242"),
                ],
                "literature_findings": [
                    {
                        "paper": "Iqbal et al., 2010 (PMID 20137777)",
                        "genotypes": "Homozygous SH3PXD2B c.76-2A>C splice-acceptor variant; heterozygous carrier parents",
                        "phenotype": "Frank-ter Haar syndrome with skeletal, eye, and cardiac abnormalities",
                        "finding": "The study identified SH3PXD2B disruption as a cause of FTHS and reported this splice-acceptor change as homozygous in an affected patient.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2820172/",
                    }
                ],
            },
            {
                "variant": "SH3PXD2B loss-of-function/deletion model",
                "display_name": "SH3PXD2B exon deletion or biallelic loss-of-function model",
                "common_name": "TKS4 podosome-loss FTHS/BDCS model",
                "position": None,
                "lookup_keys": [
                    "SH3PXD2B loss-of-function",
                    "SH3PXD2B loss of function",
                    "SH3PXD2B LOF",
                    "SH3PXD2B deletion",
                    "SH3PXD2B exon 13 deletion",
                    "TKS4 exon 13 deletion",
                    "FTHS SH3PXD2B deletion",
                    "BDCS SH3PXD2B mutation",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Biallelic loss-of-function / exon-deletion rare-disease mechanism",
                "clinical_interpretation": (
                    "Biallelic SH3PXD2B loss-of-function and deletion evidence supports an established autosomal-recessive FTHS/BDCS mechanism. "
                    "The 2010 FTHS report identified multiple homozygous SH3PXD2B mutations and a homozygous deletion in mutation-positive families; an exon 13 deletion was later reported in siblings with FTHS and craniosynostosis; the BDCS study reported homozygous SH3PXD2B mutations with complete protein loss in patient fibroblasts."
                ),
                "clinical_significance": "Clinically established autosomal-recessive SH3PXD2B loss-of-function mechanism; zygosity, phase, and breakpoint confirmation are essential.",
                "functional_effects": [
                    "TKS4 loss impairs functional podosome formation, MT1-MMP recruitment, and extracellular-matrix degradation.",
                    "Disease-model work connects SH3PXD2B loss to defective collagen remodeling, dermal fibrosis, contractures, skeletal abnormalities, ocular findings, and cardiac/craniofacial development.",
                ],
                "associated_conditions": [
                    "Frank-ter Haar syndrome / Ter Haar syndrome",
                    "Borrone dermato-cardio-skeletal syndrome spectrum",
                    "Defective collagen remodeling spectrum, dermal fibrosis, contractures, glaucoma/ocular hypertension, skeletal dysplasia, craniosynostosis, and cardiac valve/septal findings",
                ],
                "research_context": [
                    "Use this marker for confirmed or strongly suspected loss-of-function, exon deletion, or structural loss records rather than generic SH3PXD2B SNPs.",
                    "Breakpoints, transcript effect, copy number, phase, family segregation, phenotype, and clinical confirmation drive interpretation.",
                ],
                "usual_variant_note": "Mechanism-level biallelic SH3PXD2B/TKS4 loss marker rather than one specific SNV.",
                "methylation_interpretation": (
                    "SH3PXD2B methylation can provide local regulatory context, but it does not establish a deletion, loss-of-function allele, or biallelic disease state."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PMCID PMC2820172: multiple SH3PXD2B mutations/deletion and FTHS mechanism", "https://pmc.ncbi.nlm.nih.gov/articles/PMC2820172/"),
                    _evidence("PMCID PMC3532175: SH3PXD2B exon 13 deletion in FTHS siblings", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3532175/"),
                    _evidence("European Journal of Human Genetics 2014: SH3PXD2B mutations and protein loss in BDCS", "https://www.nature.com/articles/ejhg2013229"),
                    _evidence("PubMed 19144821: Tks4 loss and defective podosome/ECM degradation biology", "https://pubmed.ncbi.nlm.nih.gov/19144821/"),
                    _evidence("PMCID PMC7790187: sh3pxd2b model and collagen-remodeling/fibrosis biology", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7790187/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Bendon et al., 2012 (PMID 23140272)",
                        "genotypes": "Homozygous SH3PXD2B exon 13 deletion in three affected siblings",
                        "phenotype": "Frank-ter Haar syndrome with sagittal craniosynostosis and raised intracranial pressure",
                        "finding": "The report expanded the FTHS phenotype and tied exon-level SH3PXD2B deletion to craniosynostosis and raised intracranial pressure in affected siblings.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3532175/",
                    },
                    {
                        "paper": "Wilson et al., 2014 (PMID 24105371)",
                        "genotypes": "Homozygous SH3PXD2B mutations with loss of SH3PXD2B protein in fibroblasts",
                        "phenotype": "Borrone dermato-cardio-skeletal syndrome",
                        "finding": "The study showed that a subset of BDCS and FTHS cases are allelic and connected SH3PXD2B mutation-positive BDCS to complete TKS4 protein loss.",
                        "url": "https://www.nature.com/articles/ejhg2013229",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from SH3PXD2B/TKS4 rare-disease, podosome, extracellular-matrix, collagen-remodeling, expression, and cancer-cell-migration literature.",
        "population_coverage_note": (
            "The bundled SH3PXD2B population database is literature oriented and does not include a complete allele-frequency panel. "
            "Interpretation should prioritize current ClinVar/gnomAD status, zygosity, phase, segregation, copy-number breakpoint, transcript effect, phenotype match, and the definitive autosomal-recessive ClinGen curation."
        ),
        "population_sources": [
            _evidence("NCBI Gene 285590: SH3PXD2B expression, GO context, bibliography, and phenotype links", "https://www.ncbi.nlm.nih.gov/gene/285590"),
            _evidence("Human Protein Atlas: SH3PXD2B low tissue specificity, cytoplasmic expression, and plasma-membrane/nucleoplasm/cytosol localization", "https://www.proteinatlas.org/ENSG00000174705-SH3PXD2B"),
            _evidence("ClinGen Gene Facts: definitive AR Frank-ter Haar syndrome gene-disease validity", "https://search.clinicalgenome.org/kb/genes/HGNC%3A29242"),
            _evidence("ClinVar RCV000000214: pathogenic splice-acceptor example", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000000214/"),
            _evidence("PMCID PMC2820172: SH3PXD2B FTHS mutation spectrum and developmental model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC2820172/"),
            _evidence("PMCID PMC7790187: sh3pxd2b collagen-remodeling/fibrosis model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7790187/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "SH3PXD2B expression",
                "location_group": "Global pattern",
                "summary": "NCBI and Human Protein Atlas resources frame SH3PXD2B as broadly expressed or low tissue specificity, with cytoplasmic expression in many cell types including fibroblasts and plasma-membrane/nucleoplasm/cytosol localization context.",
            },
            {
                "variant": "Biallelic SH3PXD2B pathogenic variants",
                "location_group": "Rare-disease cohorts",
                "summary": "ClinGen curates SH3PXD2B-Frank-ter Haar syndrome as definitive with autosomal-recessive mode of inheritance; pathogenic interpretation therefore depends strongly on biallelic pathogenic dosage and phenotype fit.",
            },
            {
                "variant": "SH3PXD2B c.76-2A>C / rs775217258",
                "location_group": "ClinVar and literature marker",
                "summary": "This splice-acceptor variant is a pathogenic FTHS marker in ClinVar/OMIM literature, with GRCh37 coordinate chr5:171849502 and transcript-level c.76-2A>C notation.",
            },
            {
                "variant": "SH3PXD2B loss-of-function/deletion",
                "location_group": "Structural and functional mechanism",
                "summary": "Reported homozygous SH3PXD2B mutations, deletions, and exon 13 deletion support a TKS4 loss mechanism affecting podosome function, ECM degradation, collagen remodeling, skeletal/ocular/cardiac features, dermal fibrosis, and contractures.",
            },
            {
                "variant": "SH3PXD2B cancer-cell migration biology",
                "location_group": "Functional biology",
                "summary": "TKS4 participates in podosome/invadopodia formation and ECM degradation, so cancer and epithelial-plasticity literature is useful for pathway context but should not be treated as a germline cancer-risk prediction.",
            },
            {
                "variant": "SH3PXD2B TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body regulatory context, but no population methylation threshold is bundled for FTHS, BDCS, collagen remodeling, ocular disease, fibrosis, or cancer migration.",
            },
        ],
    },
    {
        "gene_name": "FRMD3",
        "cytoband": "9q21.32",
        "chromosome": "9",
        "start": 85857907,
        "end": 86153316,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 257019 reports FRMD3 on GRCh37.p13 at NC_000009.11:85857907..86153316, "
            "complement, and on GRCh38.p14 at NC_000009.12:83242992..83585794, complement; "
            "ClinGen Gene Facts HGNC:24125 reports GRCh37 chr9:85857907-86153316 and GRCh38 chr9:83242992-83585794"
        ),
        "manifest_filter_region": "9:85857907-86154316",
        "gene_summary": (
            "FRMD3 encodes FERM domain-containing protein 3, also known as protein 4.1O, a band-4.1-family membrane/cytoskeletal protein with FERM, FERM-adjacent, and PH-like domain annotations. "
            "NCBI and UniProt/HPA resources frame FRMD3 as a predicted membrane and cytoskeletal protein with broad expression including kidney, heart, skeletal muscle, and other tissues; experimental literature links it to diabetic kidney disease regulatory biology and tumor-suppressor or focal-adhesion pathways."
        ),
        "clinical_context": (
            "The local FRMD3 knowledge base is diabetic kidney disease, renal expression, BMP-pathway regulation, cytoskeletal organization, and tumor-suppressor research oriented. "
            "ClinGen has not published an FRMD3 gene-disease validity or dosage curation, so FRMD3 findings should remain complex-trait or pathway context unless external clinical review supplies stronger variant-level evidence."
        ),
        "variant_effect_overview": [
            "The strongest human genetics signal is the noncoding rs1888747/linked FRMD3-region diabetic nephropathy locus: a type 1 diabetes GWAS reported the strongest association near FRMD3, and multiple later studies evaluated rs1888747 in type 2 diabetes kidney cohorts.",
            "Regulatory modeling work proposed that rs1888747 may alter transcription-factor binding in an extended FRMD3 promoter framework connected to BMP signaling, with FRMD3 expression decreased in more advanced diabetic nephropathy tissue data.",
            "Replication evidence is mixed and cohort dependent: one Brazilian T2D cohort reported the rs1888747 C/C genotype as protective for DKD, while other populations and clinical settings reported different or null findings.",
            "Cancer literature frames FRMD3 as a putative tumor suppressor in NSCLC and breast cancer models, including vimentin degradation and focal-adhesion impairment, but this is expression/model biology rather than a germline cancer-risk prediction.",
        ],
        "condition_research_overview": [
            "Diabetic kidney disease, albuminuria, diabetic nephropathy, renal FRMD3 expression, podocyte/tubule protein localization, and FRMD3-region GWAS replication.",
            "BMP signaling, transcription-factor binding-site modeling, kidney cytoskeletal organization, membrane-proximal FERM-domain biology, and actomyosin structure context.",
            "Tumor-suppressor, non-small-cell lung cancer, breast-cancer growth and metastasis, vimentin ubiquitination/degradation, focal adhesion, EMT-like and migration/invasion biology.",
        ],
        "methylation_interpretation": (
            "FRMD3 has a bundled promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use FRMD3 methylation as local regulatory context around a reverse-strand membrane/cytoskeletal FERM-domain gene, not as a validated diabetic kidney disease, retinopathy, diabetes, cancer, or BMP-pathway biomarker."
        ),
        "methylation_effects": [
            "The FRMD3 methylation view can summarize promoter-proximal and gene-body CpGs around the reverse-strand transcription start.",
            "No source-backed FRMD3 methylation threshold is bundled; beta values should be interpreted alongside tissue, renal cell composition, kidney phenotype, diabetes status, expression, sequence evidence, and cohort ancestry.",
        ],
        "methylation_condition_research": [
            "Use FRMD3 methylation as supportive regulatory context in kidney, diabetic nephropathy, BMP signaling, cytoskeletal, focal-adhesion, or cancer-expression research.",
            "Do not infer diabetic kidney disease, nondiabetic renal disease, diabetic retinopathy, tumor suppression, focal-adhesion state, or BMP-pathway activity from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 257019: FRMD3 gene summary, GRCh37/GRCh38 coordinates, expression, RefSeq, GO, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/257019"),
            _evidence("Ensembl GRCh37 ENSG00000172159: FRMD3 gene model and reverse-strand coordinates", "https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000172159"),
            _evidence("UniProt A2A2Y4: FRMD3 / FERM domain-containing protein 3 entry", "https://www.uniprot.org/uniprotkb/A2A2Y4/entry"),
            _evidence("Human Protein Atlas ENSG00000172159-FRMD3: expression, localization, and protein-structure context", "https://www.proteinatlas.org/ENSG00000172159-FRMD3"),
            _evidence("ClinGen Gene Facts HGNC:24125: FRMD3 coordinates, constraint, and no published ClinGen curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A24125"),
            _evidence("PubMed 19252134: GWAS identifying FRMD3-region diabetic nephropathy susceptibility signal in type 1 diabetes", "https://pubmed.ncbi.nlm.nih.gov/19252134/"),
            _evidence("PMCID PMC3712052: rs1888747 promoter/BMP-pathway regulatory model for FRMD3 in diabetic nephropathy", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3712052/"),
            _evidence("PMCID PMC4706705: rs1888747, FRMD3 kidney expression, and DKD in a type 2 diabetes cohort", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4706705/"),
            _evidence("PMCID PMC4696171: narrative review of FRMD3 in diabetic kidney disease and mixed replication evidence", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4696171/"),
            _evidence("PubMed 34162050: rs1888747 in diabetic kidney disease, retinopathy, and nondiabetic renal disease biopsy context", "https://pubmed.ncbi.nlm.nih.gov/34162050/"),
            _evidence("Oncogene 2007: FRMD3 as a putative tumor suppressor in NSCLC", "https://www.nature.com/articles/1210225"),
            _evidence("PMCID PMC9834407: FRMD3 inhibits breast-cancer growth/metastasis through vimentin degradation and focal-adhesion impairment", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9834407/"),
        ],
        "variants": [
            {
                "variant": "rs1888747",
                "display_name": "rs1888747 (FRMD3 diabetic kidney disease regulatory locus marker)",
                "common_name": "FRMD3 promoter-proximal DKD GWAS marker",
                "position": None,
                "lookup_keys": [
                    "rs1888747",
                    "FRMD3 rs1888747",
                    "FRMD3:rs1888747",
                    "rs10868025",
                    "FRMD3 rs10868025",
                    "FRMD3 DKD locus",
                    "FRMD3 diabetic nephropathy locus",
                ],
                "region_class": "upstream_regulatory",
                "interpretation_scope": "Complex-trait regulatory marker / diabetic kidney disease association context",
                "clinical_interpretation": (
                    "rs1888747 is curated as a FRMD3-region noncoding marker for diabetic kidney disease research. "
                    "The original type 1 diabetes GWAS reported the strongest association at the FRMD3 locus and later work modeled rs1888747 as a promoter-proximal regulatory SNP that could affect transcription-factor binding and BMP-pathway coregulation. "
                    "A Brazilian T2D cohort reported C/C as protective for DKD, but FRMD3 mRNA and protein expression did not differ by rs1888747 genotype in that kidney-expression study, and review literature emphasizes mixed replication across populations."
                ),
                "clinical_significance": "Research-level diabetic kidney disease association marker; not a diagnostic DKD, diabetes, retinopathy, or monogenic kidney-disease allele.",
                "functional_effects": [
                    "Noncoding FRMD3-region marker with proposed transcription-factor binding and BMP-pathway regulatory effects.",
                    "The rs1888747 C/C genotype was associated with lower DKD odds in one T2D cohort, while G-containing genotypes carried the contrast state in that analysis.",
                    "No direct rs1888747 genotype effect on FRMD3 mRNA or protein expression was demonstrated in the cited kidney-expression study.",
                ],
                "associated_conditions": [
                    "Diabetic kidney disease / diabetic nephropathy",
                    "Albuminuria, macroalbuminuria, ESRD, and kidney-function progression in diabetes cohorts",
                    "BMP-pathway and renal transcriptional-regulation research",
                ],
                "research_context": [
                    "Interpret rs1888747 in a cohort-, ancestry-, diabetes-type-, renal-phenotype-, and genotype-dosage-aware way.",
                    "Do not convert a single rs1888747 VCF row into a clinical kidney-disease prediction without phenotype, ancestry, glycemic exposure, blood-pressure, kidney-biopsy, and current association evidence review.",
                    "The linked rs10868025 marker is retained as regional LD context rather than an independent causal rule.",
                ],
                "usual_variant_note": "Promoter-proximal FRMD3 diabetic nephropathy GWAS marker; C/G direction and effect size are cohort dependent.",
                "methylation_interpretation": (
                    "FRMD3 methylation may add local regulatory context but does not establish rs1888747 mechanism, BMP-pathway state, or DKD risk in an individual sample."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PubMed 19252134: original type 1 diabetes nephropathy GWAS FRMD3 signal", "https://pubmed.ncbi.nlm.nih.gov/19252134/"),
                    _evidence("PMCID PMC3712052: rs1888747 promoter/BMP-pathway model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3712052/"),
                    _evidence("PMCID PMC4706705: rs1888747 and FRMD3 expression in T2D DKD cohort", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4706705/"),
                    _evidence("PMCID PMC4696171: review of mixed FRMD3/DKD replication", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4696171/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Pezzolesi et al., 2009 (PMID 19252134)",
                        "genotypes": "FRMD3-region GWAS markers including rs1888747 and linked variants",
                        "phenotype": "Diabetic nephropathy susceptibility in type 1 diabetes, with DCCT/EDIC support",
                        "finding": "The study reported the strongest GWAS association at the FRMD3 locus and supported the signal in a prospective diabetes-complication cohort.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/19252134/",
                    },
                    {
                        "paper": "Buffon et al., 2016 (PMID 26753002)",
                        "genotypes": "rs1888747 C/C versus G-carrier genotypes in type 2 diabetes",
                        "phenotype": "Diabetic kidney disease and FRMD3 expression in human kidney tissue",
                        "finding": "The study reported C/C as protective for DKD in the cohort but found no rs1888747 genotype effect on FRMD3 mRNA or protein expression.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4706705/",
                    },
                ],
            },
            {
                "variant": "FRMD3 tumor-suppressor/cytoskeletal model",
                "display_name": "FRMD3 protein 4.1O tumor-suppressor and focal-adhesion model",
                "common_name": "FRMD3 vimentin/focal-adhesion functional model",
                "position": None,
                "lookup_keys": [
                    "FRMD3 tumor suppressor",
                    "FRMD3 NSCLC",
                    "FRMD3 breast cancer",
                    "FRMD3 vimentin",
                    "FRMD3 focal adhesion",
                    "FRMD3 metastasis",
                    "FRMD3 cytoskeletal model",
                    "protein 4.1O tumor suppressor",
                ],
                "region_class": "functional_model",
                "interpretation_scope": "Expression/functional model / tumor suppressor, cytoskeleton, focal adhesion, and migration biology",
                "clinical_interpretation": (
                    "FRMD3 functional evidence supports a pathway-level tumor-suppressor and cytoskeletal model rather than a specific inherited cancer allele. "
                    "NSCLC work identified FRMD3 as a putative tumor suppressor candidate, while breast-cancer model work reported that FRMD3 inhibits proliferation, migration, invasion, xenograft growth, and metastasis through ubiquitination-mediated vimentin degradation and focal-adhesion pathway impairment."
                ),
                "clinical_significance": "Functional tumor-suppressor/cytoskeletal research model; not a diagnostic germline cancer-risk variant.",
                "functional_effects": [
                    "FERM-domain protein 4.1O biology supports membrane/cytoskeletal organization and actomyosin-structure context.",
                    "Breast-cancer functional work linked FRMD3 to vimentin degradation, lower focal-adhesion complex signaling, altered cell morphology, and reduced migration/invasion.",
                    "NCBI and HPA resources support broad tissue expression and membrane/cytoskeleton-related annotations.",
                ],
                "associated_conditions": [
                    "Non-small-cell lung cancer expression research",
                    "Breast cancer growth, migration, invasion, recurrence, and metastasis model research",
                    "Cytoskeletal organization, focal adhesion, vimentin biology, and membrane-associated FERM-domain signaling",
                ],
                "research_context": [
                    "Use this marker for expression, model-system, or curated functional annotations rather than generic SNP rows.",
                    "Pair sequence calls with tumor expression, methylation, copy-number, proteomics, cell context, and functional data before drawing cancer-biology conclusions.",
                ],
                "usual_variant_note": "Functional FRMD3/protein 4.1O expression model rather than a specific inherited variant.",
                "methylation_interpretation": (
                    "FRMD3 methylation can support local regulatory review, but it does not quantify FRMD3 protein, vimentin degradation, focal-adhesion signaling, or tumor behavior."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("Oncogene 2007: FRMD3 as putative NSCLC tumor suppressor", "https://www.nature.com/articles/1210225"),
                    _evidence("PMCID PMC9834407: FRMD3, vimentin degradation, focal adhesion, and breast-cancer metastasis model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9834407/"),
                    _evidence("NCBI Gene 257019: FRMD3 GO, expression, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/257019"),
                    _evidence("Human Protein Atlas: FRMD3 tissue and subcellular expression", "https://www.proteinatlas.org/ENSG00000172159-FRMD3"),
                ],
                "literature_findings": [
                    {
                        "paper": "Haase et al., 2007 (PMID 17260017)",
                        "genotypes": "FRMD3 differential expression in NSCLC research context",
                        "phenotype": "Non-small-cell lung cancer tumor-suppressor candidate biology",
                        "finding": "The study identified FRMD3 as a novel putative tumor suppressor in NSCLC expression analyses.",
                        "url": "https://www.nature.com/articles/1210225",
                    },
                    {
                        "paper": "Shao et al., 2023 (PMID 36631457)",
                        "genotypes": "FRMD3 overexpression and knockdown in breast-cancer model systems",
                        "phenotype": "Breast-cancer growth, migration, invasion, focal adhesion, and metastasis",
                        "finding": "The study reported FRMD3-mediated vimentin ubiquitination/degradation and reduced focal-adhesion signaling as a mechanism for lower breast-cancer growth and metastasis in models.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9834407/",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from FRMD3 diabetic kidney disease association, kidney-expression, BMP-regulatory, tumor-suppressor, cytoskeletal, and methylation-context literature.",
        "population_coverage_note": (
            "The bundled FRMD3 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because FRMD3 is not a ClinGen-curated monogenic disease gene, interpretation should prioritize current GWAS/ClinVar/gnomAD status, diabetes type, kidney phenotype definition, ancestry, renal expression, tissue context, and independent clinical evidence."
        ),
        "population_sources": [
            _evidence("NCBI Gene 257019: FRMD3 expression, GO context, and bibliography", "https://www.ncbi.nlm.nih.gov/gene/257019"),
            _evidence("Human Protein Atlas: FRMD3 tissue-enhanced and subcellular expression context", "https://www.proteinatlas.org/ENSG00000172159-FRMD3"),
            _evidence("ClinGen Gene Facts: no published FRMD3 clinical curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A24125"),
            _evidence("PubMed 19252134: T1D diabetic nephropathy GWAS", "https://pubmed.ncbi.nlm.nih.gov/19252134/"),
            _evidence("PMCID PMC3712052: FRMD3 rs1888747 BMP regulatory model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3712052/"),
            _evidence("PMCID PMC4706705: T2D rs1888747 and FRMD3 kidney expression study", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4706705/"),
            _evidence("PMCID PMC9834407: FRMD3 breast-cancer functional model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9834407/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "FRMD3 expression",
                "location_group": "Global pattern",
                "summary": "NCBI reports broad expression including heart and kidney, while Human Protein Atlas frames FRMD3 as tissue-enhanced in heart muscle, skeletal muscle, and tongue with Golgi-apparatus and plasma-membrane localization.",
            },
            {
                "variant": "rs1888747 diabetic nephropathy locus",
                "location_group": "Diabetes kidney cohorts",
                "summary": "The rs1888747/FRMD3-region marker is a recurrent DKD association candidate from T1D GWAS and T2D follow-up cohorts, but effect direction and strength are cohort dependent and should not be treated as deterministic.",
            },
            {
                "variant": "rs1888747 C/C genotype",
                "location_group": "Genotype-specific cohort context",
                "summary": "One Brazilian T2D cohort reported C/C as protective for DKD and found no direct genotype effect on FRMD3 mRNA or protein, while a Thai biopsy cohort associated C allele/CC with nondiabetic renal disease context among overt proteinuria cases.",
            },
            {
                "variant": "FRMD3 BMP regulatory model",
                "location_group": "Functional genomics",
                "summary": "A promoter-modeling study proposed that rs1888747 may alter transcription-factor binding and link FRMD3 to BMP-pathway regulation in diabetic nephropathy, providing a hypothesis-generating rather than clinically validated mechanism.",
            },
            {
                "variant": "FRMD3 tumor-suppressor and focal-adhesion biology",
                "location_group": "Cancer model biology",
                "summary": "NSCLC and breast-cancer studies support tumor-suppressor and focal-adhesion/vimentin model biology, but these data are expression and model-system context rather than germline cancer-risk rules.",
            },
            {
                "variant": "FRMD3 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body regulatory context, but no population methylation threshold is bundled for DKD, retinopathy, diabetes, BMP signaling, or cancer-migration interpretation.",
            },
        ],
    },
    {
        "gene_name": "TMEM218",
        "cytoband": "11q24.2",
        "chromosome": "11",
        "start": 124964285,
        "end": 124981522,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 219854 reports TMEM218 on GRCh37.p13 at NC_000011.9:124964285..124981522, "
            "complement, and on GRCh38.p14 at NC_000011.10:125094389..125111626, complement; "
            "ClinGen Gene Facts HGNC:27344 reports GRCh37 chr11:124964285-124981522 and GRCh38 chr11:125094389-125111626"
        ),
        "manifest_filter_region": "11:124964285-124982522",
        "gene_summary": (
            "TMEM218 encodes transmembrane protein 218, a small multi-pass membrane protein linked to ciliary transition-zone biology. "
            "NCBI, UniProt, Human Protein Atlas, and ciliary-disease studies frame TMEM218 as a cilium/membrane protein that participates in transition-zone function, MKS-module biology, and ciliary membrane diffusion-barrier control."
        ),
        "clinical_context": (
            "The local TMEM218 knowledge base is Joubert syndrome 39, Meckel syndrome, and broader syndromic-ciliopathy research oriented. "
            "Reported human disease is autosomal recessive: TMEM218 findings should be interpreted with phase, zygosity, variant classification, phenotype, and transcript context, and a single heterozygous finding should generally be treated as carrier or research context."
        ),
        "variant_effect_overview": [
            "Biallelic rare deleterious TMEM218 variants have been reported in families with Joubert-Meckel spectrum phenotypes including molar tooth sign, retinal dystrophy, occipital encephalocele, polycystic kidneys, and polydactyly.",
            "Epting et al. reported biallelic missense and nonsense TMEM218 variants in Bardet-Biedl-like, Joubert, and Meckel-Gruber phenotypes and showed that p.Arg37Ser reduced interaction with the MKS-module protein TMEM67/Meckelin.",
            "Functional work supports TMEM218 as a ciliary transition-zone and diffusion-barrier component: TMEM218-knockout cells mislocalize ciliary membrane proteins, and MKS-associated variants can fail to restore transition-zone integrity.",
            "Genotype-phenotype severity appears variant specific: truncating or severe transition-zone-disruptive alleles are enriched toward lethal Meckel syndrome, while some missense combinations are reported in Joubert-spectrum presentations.",
        ],
        "condition_research_overview": [
            "Joubert syndrome 39, Meckel syndrome, Bardet-Biedl-like ciliopathy features, and Joubert-Meckel-nephronophthisis spectrum research.",
            "Ciliary transition-zone assembly, MKS and NPHP module interaction, TMEM67/Meckelin interaction, ARL13B/GPR161 ciliary membrane localization, and ciliary diffusion-barrier function.",
            "Retinal dystrophy, molar tooth sign, occipital encephalocele, cystic or polycystic kidneys, polydactyly, nephronophthisis-like kidney and retina model phenotypes, and severe fetal ciliopathy context.",
        ],
        "methylation_interpretation": (
            "TMEM218 has a promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use TMEM218 methylation as local regulatory context around a reverse-strand ciliary transition-zone gene, not as a validated Joubert syndrome, Meckel syndrome, retinal dystrophy, renal ciliopathy, or diffusion-barrier biomarker."
        ),
        "methylation_effects": [
            "Promoter-proximal and gene-body methylation may help frame local TMEM218 regulatory context in samples where sequence, expression, or ciliary phenotype evidence is also available.",
            "No source-backed TMEM218 methylation threshold is bundled; beta values should be interpreted alongside tissue, developmental timing, cell composition, variant phase, zygosity, consequence, and current ClinVar/gnomAD evidence.",
        ],
        "methylation_condition_research": [
            "Use TMEM218 methylation as supportive local regulatory context in ciliopathy, transition-zone, retina, kidney, fetal-development, or ciliary membrane trafficking research.",
            "Do not infer Joubert syndrome, Meckel syndrome, nephronophthisis, retinal dystrophy, polydactyly, encephalocele, or ciliary diffusion-barrier failure from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 219854: TMEM218 gene summary, GRCh37/GRCh38 coordinates, expression, RefSeq, associated Joubert syndrome 39 context", "https://www.ncbi.nlm.nih.gov/gene/219854"),
            _evidence("Ensembl ENSG00000150433: TMEM218 current gene model and reverse-strand coordinates", "https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000150433"),
            _evidence("UniProt A2RU14: TMEM218 / transmembrane protein 218 entry", "https://www.uniprot.org/uniprotkb/A2RU14/entry"),
            _evidence("Human Protein Atlas ENSG00000150433-TMEM218: expression, predicted location, and protein-structure context", "https://www.proteinatlas.org/ENSG00000150433-TMEM218"),
            _evidence("ClinGen Gene Facts HGNC:27344: TMEM218 coordinates, MANE transcript, constraint, and GenCC classification", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27344/external-resources"),
            _evidence("ClinVar RCV002223152: NM_001258244.2(TMEM218):c.111G>T (p.Arg37Ser) likely pathogenic for Joubert syndrome 39", "https://www.ncbi.nlm.nih.gov/clinvar/RCV002223152/"),
            _evidence("PubMed 33791682: TMEM218 dysfunction causes ciliopathies, including Joubert and Meckel syndromes", "https://pubmed.ncbi.nlm.nih.gov/33791682/"),
            _evidence("PubMed 35137054: TMEM218 transition-zone protein, NPHP-module interaction, and syndromic ciliopathies", "https://pubmed.ncbi.nlm.nih.gov/35137054/"),
            _evidence("Human Molecular Genetics 2024: TMEM218 ciliopathy variants disrupt ciliary transition-zone diffusion-barrier function", "https://academic.oup.com/hmg/article-abstract/33/16/1442/7673982"),
            _evidence("PubMed 26982032: MKS5/CEP290 transition-zone assembly pathway identifies TMEM-218 as an MKS-module-associated transition-zone protein", "https://pubmed.ncbi.nlm.nih.gov/26982032/"),
        ],
        "variants": [
            {
                "variant": "TMEM218 c.111G>T / p.Arg37Ser",
                "display_name": "TMEM218 c.111G>T / p.Arg37Ser",
                "common_name": "TMEM218 R37S likely pathogenic JBTS39 marker",
                "position": 124971199,
                "lookup_keys": [
                    "TMEM218 c.111G>T",
                    "TMEM218 p.Arg37Ser",
                    "TMEM218 Arg37Ser",
                    "TMEM218 R37S",
                    "NM_001080546.3:c.111G>T",
                    "NM_001258244.2:c.111G>T",
                    "NP_001074015.1:p.Arg37Ser",
                    "rs1248550491",
                    "11:124971199",
                    "11:124971199:C>A",
                    "NC_000011.9:g.124971199C>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Likely pathogenic autosomal-recessive Joubert syndrome 39 marker / ciliary transition-zone research context",
                "clinical_interpretation": (
                    "TMEM218 c.111G>T / p.Arg37Ser is included because ClinVar curates it as likely pathogenic for Joubert syndrome 39 and Epting et al. reported the homozygous missense variant in a patient with syndromic ciliopathy features. "
                    "Functional evidence showed reduced interaction of p.Arg37Ser TMEM218 with TMEM67/MKS3/Meckelin, supporting an MKS-module transition-zone mechanism."
                ),
                "clinical_significance": "Likely pathogenic Joubert syndrome 39 marker; autosomal-recessive interpretation depends on biallelic dosage, phase, phenotype, and current clinical review.",
                "functional_effects": [
                    "Missense variant at a conserved TMEM218 arginine residue reported in a homozygous affected individual with Bardet-Biedl-like/Joubert-spectrum features.",
                    "Co-immunoprecipitation evidence showed reduced interaction between p.Arg37Ser TMEM218 and TMEM67/MKS3/Meckelin.",
                    "ClinVar lists the GRCh37 coordinate as chr11:124971199 and the HGVS as NM_001258244.2:c.111G>T / p.Arg37Ser with likely pathogenic classification for JBTS39.",
                ],
                "associated_conditions": [
                    "Joubert syndrome 39",
                    "Bardet-Biedl-like and Joubert-Meckel syndromic ciliopathy features",
                    "Ciliary transition-zone and MKS-module interaction research",
                ],
                "research_context": [
                    "Prioritize homozygous or compound-heterozygous TMEM218 context, parental segregation, transcript version, phenotype fit, and current variant classification.",
                    "A single heterozygous observation should generally be reported as carrier or research context unless paired with another pathogenic TMEM218 allele.",
                    "Interpret alongside neurological imaging, retinal, renal, polydactyly, encephalocele, and fetal or perinatal phenotype data when available.",
                ],
                "usual_variant_note": "Likely pathogenic JBTS39 missense marker; disease interpretation requires biallelic autosomal-recessive context.",
                "methylation_interpretation": (
                    "TMEM218 methylation can provide local regulatory context, but it does not establish p.Arg37Ser dosage, phase, pathogenicity, TMEM67 interaction, or ciliary transition-zone function."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar RCV002223152: p.Arg37Ser likely pathogenic for Joubert syndrome 39", "https://www.ncbi.nlm.nih.gov/clinvar/RCV002223152/"),
                    _evidence("PubMed 35137054: p.Arg37Ser TMEM218 in syndromic ciliopathy and reduced MKS3/TMEM67 interaction", "https://pubmed.ncbi.nlm.nih.gov/35137054/"),
                    _evidence("NCBI Gene 219854: TMEM218 coordinates and JBTS39 context", "https://www.ncbi.nlm.nih.gov/gene/219854"),
                ],
                "literature_findings": [
                    {
                        "paper": "Epting et al., 2022 (PMID 35137054)",
                        "genotypes": "Homozygous TMEM218 c.111G>T / p.Arg37Ser in an affected individual; heterozygous carrier parents",
                        "phenotype": "Bardet-Biedl-like, Joubert, and syndromic ciliopathy features including molar tooth sign and polydactyly context",
                        "finding": "The study reported p.Arg37Ser as a TMEM218 disease variant and showed reduced interaction with the MKS-module protein TMEM67/MKS3/Meckelin.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/35137054/",
                    },
                    {
                        "paper": "ClinVar RCV002223152",
                        "genotypes": "NM_001258244.2(TMEM218):c.111G>T / p.Arg37Ser",
                        "phenotype": "Joubert syndrome 39",
                        "finding": "ClinVar classifies the variant as likely pathogenic for JBTS39 with single-submitter criteria-provided review status.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/RCV002223152/",
                    },
                ],
            },
            {
                "variant": "TMEM218 biallelic Joubert-Meckel ciliopathy model",
                "display_name": "TMEM218 biallelic loss or reduced-function ciliopathy model",
                "common_name": "TMEM218 transition-zone Joubert-Meckel model",
                "position": None,
                "lookup_keys": [
                    "TMEM218 loss-of-function",
                    "TMEM218 loss of function",
                    "TMEM218 LOF",
                    "TMEM218 deletion",
                    "TMEM218 biallelic",
                    "TMEM218 Joubert syndrome",
                    "TMEM218 Meckel syndrome",
                    "TMEM218 Joubert-Meckel",
                    "TMEM218 transition zone",
                    "TMEM218 diffusion barrier",
                    "TMEM218 c.175C>T",
                    "TMEM218 p.Arg59Ter",
                    "TMEM218 p.Arg59*",
                    "TMEM218 R59*",
                    "TMEM218 p.Gly44Val",
                    "TMEM218 G44V",
                    "TMEM218 p.Arg94Ter",
                    "TMEM218 p.Arg94*",
                    "TMEM218 R94*",
                    "TMEM218 p.Arg115Cys",
                    "TMEM218 R115C",
                    "TMEM218 p.Arg115His",
                    "TMEM218 R115H",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Autosomal-recessive disease mechanism / ciliary transition-zone diffusion-barrier model",
                "clinical_interpretation": (
                    "TMEM218 biallelic loss, reduced-function, and transition-zone-disruptive variants are curated as a Joubert-Meckel ciliopathy mechanism because human genetic reports and functional studies link TMEM218 dysfunction to primary cilium transition-zone biology. "
                    "Reported phenotypes span Joubert syndrome with retinal dystrophy, Meckel syndrome, severe fetal or perinatal presentations, cystic kidney disease, encephalocele, polydactyly, and Bardet-Biedl-like overlap."
                ),
                "clinical_significance": "Strong autosomal-recessive Joubert-Meckel ciliopathy model; variant-level classification, phase, and phenotype fit are required before clinical escalation.",
                "functional_effects": [
                    "Biallelic rare deleterious variants were reported across six families with Joubert-Meckel spectrum phenotypes in the 2021 gene-discovery study.",
                    "The p.Arg59Ter truncating variant and related severe alleles were associated with fetal or perinatal Meckel-Gruber presentations in later TMEM218 reports.",
                    "TMEM218 participates in ciliary transition-zone biology and interacts genetically or physically with MKS/NPHP-module components including TMEM67/MKS3 and Nphp4.",
                    "TMEM218-knockout cell work showed impaired ciliary membrane protein localization, with MKS-associated variants such as R94* and R115C failing to restore barrier function in the cited model.",
                ],
                "associated_conditions": [
                    "Joubert syndrome 39",
                    "Meckel syndrome / Meckel-Gruber syndrome",
                    "Joubert-Meckel-nephronophthisis spectrum ciliopathy",
                    "Retinal dystrophy, cystic kidney disease, polydactyly, and occipital encephalocele research context",
                ],
                "research_context": [
                    "Use this broad model for biallelic pathogenic, likely pathogenic, or strongly supported TMEM218 loss/reduced-function evidence rather than isolated benign or VUS calls.",
                    "Pair variant calls with segregation, copy-number or transcript evidence, phenotype fit, neuroimaging, renal and retinal assessment, and updated clinical variant resources.",
                    "Variant severity should be assessed cautiously because R115C/R115H and other residue-level observations suggest cellular rescue and clinical severity can differ by amino-acid substitution.",
                ],
                "usual_variant_note": "Autosomal-recessive TMEM218 ciliopathy mechanism; not directly assayable as a single SNP unless a specific pathogenic allele is observed.",
                "methylation_interpretation": (
                    "TMEM218 methylation can support local regulatory review, but it does not prove biallelic loss, transition-zone diffusion-barrier failure, or Joubert-Meckel disease causality."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PubMed 33791682: biallelic TMEM218 variants in Joubert-Meckel spectrum ciliopathies", "https://pubmed.ncbi.nlm.nih.gov/33791682/"),
                    _evidence("PubMed 35137054: biallelic missense/nonsense TMEM218 variants and NPHP-module interaction", "https://pubmed.ncbi.nlm.nih.gov/35137054/"),
                    _evidence("Human Molecular Genetics 2024: TMEM218 variant severity and transition-zone diffusion-barrier defects", "https://academic.oup.com/hmg/article-abstract/33/16/1442/7673982"),
                    _evidence("PubMed 26982032: TMEM-218 as a transition-zone MKS-module-associated protein", "https://pubmed.ncbi.nlm.nih.gov/26982032/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Van De Weghe et al., 2021 (PMID 33791682)",
                        "genotypes": "Biallelic rare deleterious TMEM218 variants across Joubert-Meckel spectrum families",
                        "phenotype": "Joubert syndrome with retinal dystrophy and Meckel syndrome, including molar tooth sign, encephalocele, retinal dystrophy, polycystic kidneys, and polydactyly",
                        "finding": "The study made a strong human-genetic case that TMEM218 dysfunction causes ciliopathy phenotypes in the Joubert-Meckel spectrum.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/33791682/",
                    },
                    {
                        "paper": "Epting et al., 2022 (PMID 35137054)",
                        "genotypes": "Biallelic missense and nonsense TMEM218 variants including c.111G>T / p.Arg37Ser and c.175C>T / p.Arg59Ter",
                        "phenotype": "Bardet-Biedl-like, Joubert, and Meckel-Gruber syndromic ciliopathy presentations",
                        "finding": "The study characterized TMEM218 as a ciliary transition-zone protein interacting with the NPHP module and TMEM67/MKS3 biology.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/35137054/",
                    },
                    {
                        "paper": "Fujii et al., 2024 (DOI 10.1093/hmg/ddae083)",
                        "genotypes": "TMEM218(G44V), TMEM218(R94*), TMEM218(R115C), and TMEM218(R115H) cellular rescue models",
                        "phenotype": "Ciliary transition-zone integrity and membrane-protein diffusion-barrier function",
                        "finding": "The study linked cellular diffusion-barrier rescue severity to the clinical severity of TMEM218-associated MKS and JBTS variants.",
                        "url": "https://academic.oup.com/hmg/article-abstract/33/16/1442/7673982",
                    },
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from TMEM218 gene, ClinVar, Joubert-Meckel, transition-zone, and ciliary diffusion-barrier literature.",
        "population_coverage_note": (
            "The bundled TMEM218 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because TMEM218-associated disease is autosomal recessive and variant-specific, interpretation should prioritize current ClinVar/gnomAD frequency, consequence, transcript, phase, biallelic dosage, segregation, and phenotype fit."
        ),
        "population_sources": [
            _evidence("NCBI Gene 219854: TMEM218 expression, coordinates, and Joubert syndrome context", "https://www.ncbi.nlm.nih.gov/gene/219854"),
            _evidence("ClinGen Gene Facts HGNC:27344: coordinates, MANE transcript, constraint, and GenCC context", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27344/external-resources"),
            _evidence("Human Protein Atlas: TMEM218 protein class, expression, predicted location, and UniProt mapping", "https://www.proteinatlas.org/ENSG00000150433-TMEM218"),
            _evidence("ClinVar RCV002223152: TMEM218 p.Arg37Ser likely pathogenic JBTS39 marker", "https://www.ncbi.nlm.nih.gov/clinvar/RCV002223152/"),
            _evidence("PubMed 33791682: TMEM218 biallelic variants in Joubert-Meckel spectrum families", "https://pubmed.ncbi.nlm.nih.gov/33791682/"),
            _evidence("PubMed 35137054: TMEM218 reduced dosage and syndromic ciliopathy spectrum", "https://pubmed.ncbi.nlm.nih.gov/35137054/"),
            _evidence("Human Molecular Genetics 2024: TMEM218 variant effects on ciliary transition-zone diffusion barrier", "https://academic.oup.com/hmg/article-abstract/33/16/1442/7673982"),
        ],
        "gene_population_patterns": [
            {
                "variant": "TMEM218 expression and protein context",
                "location_group": "Global pattern",
                "summary": "NCBI reports ubiquitous expression and a cilium/membrane location prediction, while Human Protein Atlas maps TMEM218 to predicted intracellular/membrane location and UniProt A2RU14 protein evidence.",
            },
            {
                "variant": "TMEM218 p.Arg37Ser",
                "location_group": "ClinVar and literature marker",
                "summary": "ClinVar classifies NM_001258244.2:c.111G>T / p.Arg37Ser as likely pathogenic for Joubert syndrome 39, and Epting et al. reported reduced interaction with TMEM67/MKS3/Meckelin.",
            },
            {
                "variant": "Biallelic TMEM218 disease mechanism",
                "location_group": "Rare-disease families",
                "summary": "Human genetic reports support biallelic TMEM218 dysfunction as a Joubert-Meckel spectrum ciliopathy cause, with phenotypes including retinal dystrophy, molar tooth sign, encephalocele, cystic kidneys, and polydactyly.",
            },
            {
                "variant": "TMEM218 transition-zone diffusion barrier",
                "location_group": "Functional biology",
                "summary": "Functional studies place TMEM218 in ciliary transition-zone biology, MKS/NPHP-module interaction, TMEM67 interaction, and ciliary membrane protein diffusion-barrier control.",
            },
            {
                "variant": "Variant-specific severity",
                "location_group": "Functional and clinical gradient",
                "summary": "Cellular rescue and patient reports suggest that truncating or severe TMEM218 variants can align with Meckel syndrome while some missense contexts align with Joubert-spectrum disease, so variant class and phase matter.",
            },
            {
                "variant": "TMEM218 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body regulatory context, but no population methylation threshold is bundled for TMEM218-associated ciliopathy, renal, retinal, fetal, or diffusion-barrier interpretation.",
            },
        ],
    },
    {
        "gene_name": "FAM170A",
        "cytoband": "5q23.1",
        "chromosome": "5",
        "start": 118965253,
        "end": 118971517,
        "strand": "+",
        "coordinate_source": (
            "NCBI Gene 340069 reports FAM170A on GRCh37.p13 at NC_000005.9:118965253..118971517 "
            "and on GRCh38.p14 at NC_000005.10:119629558..119635822; Ensembl ENSG00000164334 reports "
            "the current GRCh38 gene on chr5:119629558-119635822 on the forward strand"
        ),
        "manifest_filter_region": "5:118964253-118971517",
        "gene_summary": (
            "FAM170A, also known as ZNFD, encodes protein FAM170A, a nuclear zinc-finger domain protein with DNA-, metal-, and zinc-binding annotations. "
            "Human ZNFD/FAM170A was characterized as a C2H2 zinc-finger transcriptional activator with nuclear localization and AP1(PMA)-responsive transcriptional activity; UniProt and ClinGen gene facts also frame it as a nuclear transcription factor that binds heat-shock promoter elements."
        ),
        "clinical_context": (
            "The local FAM170A knowledge base is male-fertility, spermiogenesis, and transcription-factor research oriented. "
            "Current ClinGen gene facts list no published FAM170A gene-disease validity, dosage, actionability, or variant-pathogenicity assertions, so FAM170A findings should remain exploratory unless external clinical review adds stronger evidence."
        ),
        "variant_effect_overview": [
            "The strongest functional disease-model evidence is mouse Fam170a deficiency, which causes male infertility and impaired sperm chromatin remodeling during histone-to-protamine exchange.",
            "A 2025 Nucleic Acids Research study reported very low FAM170A transcription in sperm from infertile men with teratozoospermia, supporting human relevance but not establishing a validated monogenic diagnostic rule.",
            "Human ZNFD/FAM170A overexpression activates AP1(PMA)-responsive transcription in COS7 cells, while mouse Znfd evidence supports heat-shock-element transcriptional activity and testis-specific expression.",
        ],
        "condition_research_overview": [
            "Male infertility, teratozoospermia, sperm-head morphology, and spermiogenesis chromatin remodeling.",
            "Histone-to-protamine exchange, transition-protein timing, protamine incorporation, and USP7-linked histone deubiquitination in germ-cell models.",
            "Nuclear zinc-finger transcription-factor biology, AP1(PMA) response, heat-shock-element regulation, and testis-enriched expression.",
        ],
        "methylation_interpretation": (
            "FAM170A has a compact promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use FAM170A methylation as local regulatory context around a testis-enriched nuclear transcription-factor gene, not as a validated male-infertility or teratozoospermia biomarker."
        ),
        "methylation_effects": [
            "Promoter and early-gene-body methylation may help frame local transcriptional accessibility for FAM170A.",
            "No source-backed FAM170A methylation threshold is bundled; beta values should be interpreted alongside tissue, cell composition, expression, sperm phenotype, and sequence/CNV evidence.",
        ],
        "methylation_condition_research": [
            "Use FAM170A methylation as supportive regulatory context in male-fertility, spermiogenesis, testis-expression, or transcription-factor research.",
            "Do not infer sperm chromatin remodeling defects from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 340069: FAM170A gene summary, coordinates, expression, RefSeq, and GO context", "https://www.ncbi.nlm.nih.gov/gene/340069"),
            _evidence("Ensembl ENSG00000164334: FAM170A current gene model and forward-strand coordinates", "https://www.ensembl.org/id/ENSG00000164334"),
            _evidence("UniProt A1A519: FAM170A / F170A_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A1A519/entry"),
            _evidence("ClinGen Gene Facts HGNC:27963: coordinates, MANE transcript, and no published curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27963"),
            _evidence("Human Protein Atlas ENSG00000164334: FAM170A testis expression context", "https://www.proteinatlas.org/ENSG00000164334-FAM170A"),
            _evidence("PubMed 20162441: human ZNFD/FAM170A zinc-finger AP1(PMA) transcriptional activity", "https://pubmed.ncbi.nlm.nih.gov/20162441/"),
            _evidence("PubMed 22231842: mouse Znfd heat-shock-element transcriptional activity and testis expression", "https://pubmed.ncbi.nlm.nih.gov/22231842/"),
            _evidence("PMCID PMC11760970: Fam170a deficiency and male infertility in mouse spermiogenesis", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11760970/"),
        ],
        "variants": [
            {
                "variant": "FAM170A loss-of-function/deletion model",
                "display_name": "FAM170A loss-of-function or deletion model",
                "common_name": "Fam170a deficiency male-infertility model",
                "position": None,
                "lookup_keys": [
                    "FAM170A loss-of-function",
                    "FAM170A loss of function",
                    "FAM170A deletion",
                    "FAM170A LOF",
                    "Fam170a knockout",
                    "Fam170a deficiency",
                    "FAM170A deficiency",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Preclinical loss-of-function model / male-fertility research context",
                "clinical_interpretation": (
                    "Fam170a loss-of-function is included as a model-system marker because complete mouse Fam170a deletion caused male infertility, abnormal sperm head morphology, impaired histone-to-protamine exchange, and altered USP7-linked histone deubiquitination during spermiogenesis. "
                    "Human translation remains research-grade: the same study reported very low FAM170A transcription in sperm from infertile men with teratozoospermia, but ClinGen has not published a FAM170A gene-disease validity or dosage curation."
                ),
                "clinical_significance": "Mouse model and expression-supported male-fertility research marker; not a curated human pathogenic variant.",
                "functional_effects": [
                    "Loss of Fam170a in mice impaired sperm chromatin remodeling and protamine incorporation during spermiogenesis.",
                    "Fam170a promoted USP7 nuclear translocation and deubiquitination of testis-specific H2A/H2B variants in the cited mouse study.",
                    "Low FAM170A transcription was observed in sperm from infertile men with teratozoospermia in the same study.",
                ],
                "associated_conditions": [
                    "Male infertility research",
                    "Teratozoospermia and abnormal sperm-head morphology",
                    "Spermiogenesis chromatin remodeling",
                    "Histone-to-protamine exchange",
                ],
                "research_context": [
                    "Prioritize germline assay type, CNV breakpoints or LoF consequence, tissue context, semen phenotype, and orthogonal confirmation.",
                    "Do not use a single generic FAM170A VCF signal as a diagnostic male-infertility call.",
                ],
                "usual_variant_note": "Model-system FAM170A/Fam170a deficiency marker rather than a specific curated human SNV.",
                "methylation_interpretation": (
                    "FAM170A methylation can provide local regulatory context, but it does not establish loss-of-function, sperm expression, or male-infertility causality."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PMCID PMC11760970: Fam170a deficiency causes male infertility in mouse spermiogenesis", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11760970/"),
                    _evidence("ClinGen Gene Facts HGNC:27963: no published FAM170A clinical curations", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27963"),
                ],
                "literature_findings": [
                    {
                        "paper": "Cheng et al., 2025 (PMID 39868537)",
                        "genotypes": "Fam170a complete knockout/deletion model in mice; low human FAM170A sperm transcription in teratozoospermia",
                        "phenotype": "Male infertility, abnormal sperm-head morphology, and impaired histone-to-protamine exchange",
                        "finding": "The study reported complete infertility in Fam170a-deleted male mice and implicated FAM170A/Fam170a in USP7-linked histone deubiquitination and sperm chromatin remodeling.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11760970/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from FAM170A/ZNFD testis expression, transcription-factor, and male-fertility model literature.",
        "population_coverage_note": (
            "The bundled FAM170A population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because ClinGen has not published FAM170A clinical curations, population interpretation should prioritize current ClinVar/gnomAD status, predicted consequence, tissue context, semen phenotype, and independent confirmation."
        ),
        "population_sources": [
            _evidence("NCBI Gene 340069: FAM170A expression and GO context", "https://www.ncbi.nlm.nih.gov/gene/340069"),
            _evidence("ClinGen Gene Facts HGNC:27963: clinical curation status", "https://search.clinicalgenome.org/kb/genes/HGNC%3A27963"),
            _evidence("Human Protein Atlas: FAM170A testis protein/RNA context", "https://www.proteinatlas.org/ENSG00000164334-FAM170A"),
            _evidence("PubMed 20162441: human ZNFD/FAM170A transcriptional activation study", "https://pubmed.ncbi.nlm.nih.gov/20162441/"),
            _evidence("PMCID PMC11760970: Fam170a male-infertility model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11760970/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "FAM170A testis-enriched expression",
                "location_group": "Global pattern",
                "summary": "NCBI and Human Protein Atlas resources frame FAM170A as testis-enriched or testis-relevant, matching the sperm and spermiogenesis literature focus.",
            },
            {
                "variant": "FAM170A deficiency model",
                "location_group": "Functional biology",
                "summary": "Mouse Fam170a deficiency supports a male-fertility and sperm chromatin-remodeling research thesis, but it should not be treated as a validated human diagnostic gene-disease rule.",
            },
            {
                "variant": "ZNFD transcription-factor activity",
                "location_group": "Functional biology",
                "summary": "Human and mouse ZNFD/FAM170A studies support nuclear zinc-finger transcription-factor behavior, including AP1(PMA) and heat-shock-element reporter activity.",
            },
            {
                "variant": "FAM170A TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body methylation context, but no population methylation threshold is bundled for fertility or transcriptional activity.",
            },
        ],
    },
    {
        "gene_name": "SYCE3",
        "cytoband": "22q13.33",
        "chromosome": "22",
        "start": 50989541,
        "end": 51001348,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 644186 reports SYCE3 on GRCh37.p13 at NC_000022.10:50989541..51001348, "
            "complement, and on GRCh38.p14 at NC_000022.11:50551112..50562919, complement"
        ),
        "manifest_filter_region": "22:50989541-51002348",
        "gene_summary": (
            "SYCE3 encodes synaptonemal complex central element protein 3, a small coiled-coil component of the synaptonemal complex central element. "
            "UniProt and structural studies frame SYCE3 as a major meiotic prophase I component that helps assemble central-element lattice structures, recruit other central-element complexes, and initiate synapsis between homologous chromosomes."
        ),
        "clinical_context": (
            "The local SYCE3 knowledge base is meiotic synapsis, recombination, and fertility research oriented. "
            "Mouse Syce3 knockout causes infertility in both sexes through synapsis initiation failure and meiotic arrest, but recent NOA reviews note that confirmed human SYCE3 pathogenic mutations are not yet established; app findings should therefore remain exploratory."
        ),
        "variant_effect_overview": [
            "The strongest functional evidence is the Syce3 knockout mouse model, where loss of SYCE3 blocks central-element assembly, prevents synapsis initiation, abolishes MLH1 crossover foci, and causes infertility in both sexes.",
            "Protein-structure work supports SYCE3 self-assembly and coiled-coil behavior, providing a mechanistic basis for central-element lattice formation.",
            "Human interpretation is model-informed rather than variant-confirmed: current reviews discuss SYCE3 as biologically plausible for non-obstructive azoospermia and premature ovarian insufficiency contexts, while noting the absence of confirmed human SYCE3 mutations.",
        ],
        "condition_research_overview": [
            "Synaptonemal complex central-element assembly, homologous chromosome synapsis, and meiotic recombination.",
            "Non-obstructive azoospermia, meiotic arrest, premature ovarian insufficiency, and recurrent pregnancy-loss research involving synaptonemal-complex genes.",
            "Testis-enriched expression and spermatocyte meiotic division/transition cell-type context.",
        ],
        "methylation_interpretation": (
            "SYCE3 has a bundled promoter-plus-gene EPIC slice from the local hg19 manifest. "
            "Use SYCE3 methylation as local regulatory context around a compact reverse-strand meiotic central-element gene, not as a validated infertility, NOA, POI, or recurrent-pregnancy-loss biomarker."
        ),
        "methylation_effects": [
            "The SYCE3 methylation view can summarize promoter-proximal and early-gene-body CpGs around the reverse-strand transcription start.",
            "No source-backed SYCE3 methylation threshold is bundled; beta values should be interpreted alongside tissue, cell composition, meiotic-cell expression, sequence/CNV evidence, and reproductive phenotype.",
        ],
        "methylation_condition_research": [
            "Use SYCE3 methylation as supportive regulatory context in meiotic synapsis, synaptonemal-complex, spermatogenesis, or fertility research.",
            "Do not infer meiotic arrest or reproductive risk from peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 644186: SYCE3 gene summary, coordinates, RefSeq, expression, and GO context", "https://www.ncbi.nlm.nih.gov/gene/644186"),
            _evidence("Ensembl ENSG00000217442: SYCE3 current gene model", "https://www.ensembl.org/id/ENSG00000217442"),
            _evidence("UniProt A1L190: SYCE3 / SYCE3_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A1L190/entry"),
            _evidence("Human Protein Atlas ENSG00000217442: SYCE3 testis-enriched expression context", "https://www.proteinatlas.org/ENSG00000217442-SYCE3"),
            _evidence("ClinGen Gene Facts HGNC:35245: SYCE3 gene facts and constraint metrics", "https://search.clinicalgenome.org/kb/genes/HGNC%3A35245"),
            _evidence("PMCID PMC3102746: Syce3 knockout mouse central-element loading, recombination, and fertility evidence", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3102746/"),
            _evidence("PMCID PMC6556580: molecular model for SYCE3 self-assembly", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6556580/"),
            _evidence("PMCID PMC12676892: NOA review noting SYCE3 mouse phenotype and no confirmed human mutations", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12676892/"),
        ],
        "variants": [
            {
                "variant": "SYCE3 loss-of-function/deletion model",
                "display_name": "SYCE3 loss-of-function or deletion model",
                "common_name": "Syce3 knockout meiotic-arrest fertility model",
                "position": None,
                "lookup_keys": [
                    "SYCE3 loss-of-function",
                    "SYCE3 loss of function",
                    "SYCE3 deletion",
                    "SYCE3 LOF",
                    "Syce3 knockout",
                    "Syce3 deficiency",
                    "SYCE3 deficiency",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Preclinical loss-of-function model / meiotic synapsis and fertility research context",
                "clinical_interpretation": (
                    "Syce3 loss-of-function is included as a model-system marker because knockout mice are infertile in both sexes, fail to initiate homologous chromosome synapsis, and lack MLH1 crossover foci despite apparently normal initiation of recombination. "
                    "Human translation remains research-grade because current reviews state that confirmed human SYCE3 mutations have not yet been established."
                ),
                "clinical_significance": "Mouse model-supported meiotic-arrest and fertility research marker; not a curated human pathogenic variant.",
                "functional_effects": [
                    "Loss of Syce3 blocks loading of synaptonemal-complex central-element proteins in mouse meiocytes.",
                    "Knockout meiocytes show synapsis initiation failure, severely impaired recombination progression, and absence of MLH1 crossover foci.",
                    "SYCE3 self-assembly and coiled-coil structure support a structural central-element mechanism.",
                ],
                "associated_conditions": [
                    "Male infertility and non-obstructive azoospermia research",
                    "Premature ovarian insufficiency and meiotic-arrest research",
                    "Synaptonemal-complex assembly and homologous chromosome synapsis",
                    "Crossover formation and meiotic recombination",
                ],
                "research_context": [
                    "Prioritize CNV breakpoint, predicted LoF consequence, zygosity, reproductive phenotype, meiotic-arrest histology, and orthogonal confirmation.",
                    "Do not use a generic SYCE3 VCF or methylation signal as a diagnostic infertility or POI call.",
                ],
                "usual_variant_note": "Model-system SYCE3/Syce3 deficiency marker rather than a specific curated human SNV.",
                "methylation_interpretation": (
                    "SYCE3 methylation can provide local regulatory context, but it does not establish loss-of-function, meiotic synapsis failure, or reproductive phenotype causality."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("PMCID PMC3102746: Syce3 knockout infertility, synapsis initiation failure, and absent MLH1 foci", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3102746/"),
                    _evidence("PMCID PMC12676892: SYCE3 NOA review context and no confirmed human mutations", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12676892/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Schramm et al., 2011 (PMID 21637789)",
                        "genotypes": "Syce3 knockout/deletion mouse model",
                        "phenotype": "Infertility in both sexes, synapsis initiation failure, meiotic arrest, and absent crossover markers",
                        "finding": "The study reported that SYCE3 is a mammalian synaptonemal-complex central-element component required for central-element protein loading, recombination progression, and fertility.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3102746/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from SYCE3 synaptonemal-complex, testis-expression, and meiotic-infertility model literature.",
        "population_coverage_note": (
            "The bundled SYCE3 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because confirmed human SYCE3 pathogenic mutations have not yet been established in the cited reviews, interpretation should prioritize current ClinVar/gnomAD status, predicted consequence, reproductive phenotype, and independent confirmation."
        ),
        "population_sources": [
            _evidence("NCBI Gene 644186: SYCE3 expression and meiotic GO context", "https://www.ncbi.nlm.nih.gov/gene/644186"),
            _evidence("Human Protein Atlas: SYCE3 testis and spermatocyte expression context", "https://www.proteinatlas.org/ENSG00000217442-SYCE3"),
            _evidence("PMCID PMC3102746: Syce3 knockout mouse fertility model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3102746/"),
            _evidence("PMCID PMC6556580: SYCE3 structural self-assembly model", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6556580/"),
            _evidence("PMCID PMC12676892: NOA review covering SYCE2/SYCE3 model phenotypes", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12676892/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "SYCE3 testis-enriched expression",
                "location_group": "Global pattern",
                "summary": "NCBI and Human Protein Atlas resources frame SYCE3 as testis-enriched, especially in spermatocyte and spermatid meiotic cell states.",
            },
            {
                "variant": "SYCE3 deficiency model",
                "location_group": "Functional biology",
                "summary": "Mouse Syce3 deficiency supports a meiotic-arrest and fertility research thesis, but it should not be treated as a validated human diagnostic gene-disease rule.",
            },
            {
                "variant": "SYCE3 central-element structure",
                "location_group": "Functional biology",
                "summary": "Structural work supports SYCE3 coiled-coil self-assembly as a mechanistic layer for synaptonemal-complex central-element lattice formation.",
            },
            {
                "variant": "SYCE3 TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "Bundled EPIC probes provide local promoter and gene-body methylation context, but no population methylation threshold is bundled for meiotic arrest, NOA, POI, or recurrent pregnancy loss.",
            },
        ],
    },
    {
        "gene_name": "POTEB3",
        "cytoband": "15q11.2",
        "chromosome": "15",
        "start": 21405401,
        "end": 21440499,
        "strand": "-",
        "assembly": "GRCh38 / hg38",
        "coordinate_source": (
            "NCBI Gene 102724631 and Ensembl ENSG00000278522 report POTEB3 on GRCh38 at "
            "NC_000015.10/chr15:21405401..21440499, complement; Ensembl assembly mapping returned no direct GRCh37 projection for this primary interval"
        ),
        "skip_manifest_subset": True,
        "curated_methylation_probe_ids": [],
        "promoter_review_region": {
            "label": "GRCh38 POTEB3 reverse-strand promoter review window",
            "start": 21440500,
            "end": 21441499,
            "definition": (
                "A practical 1 kb upstream window relative to the GRCh38 reverse-strand POTEB3 transcription start. "
                "This app stores it as context only because the bundled hg19 EPIC manifest has no POTEB3-specific probe slice."
            ),
        },
        "promoter_hotspot_region": {
            "label": "No bundled POTEB3 EPIC methylation probe hotspot",
            "start": 21440500,
            "end": 21441499,
            "definition": (
                "No POTEB3-specific EPIC probes are bundled from the local hg19 manifest. "
                "Methylation interpretation should use custom GRCh38-aware probe annotation or orthogonal expression data."
            ),
        },
        "gene_summary": (
            "POTEB3 encodes POTE ankyrin domain family member B3, a POTE-family intracellular protein with ankyrin-repeat domains and a supported MANE/RefSeq coding transcript. "
            "POTE genes are recent, highly paralogous cancer-testis/reproductive-tissue genes; current POTEB3 expression resources emphasize testis-enriched or restricted expression and careful paralog-aware interpretation."
        ),
        "clinical_context": (
            "The local POTEB3 knowledge base is paralog-aware and copy-number-context oriented. "
            "POTEB3 lies in the complex 15q11.2 POTE segmental-duplication region, has no direct GRCh37 primary-locus projection from Ensembl mapping, and should not be interpreted like a high-confidence single-copy clinical gene."
        ),
        "variant_effect_overview": [
            "Most POTEB3 findings should be treated as structural-variation, segmental-duplication, expression, or POTE-family research context rather than as single-nucleotide diagnostic calls.",
            "Short-read variant interpretation can be confounded by close POTE paralogs, mapping ambiguity, and the absence of a direct GRCh37 primary-locus mapping for the current POTEB3 annotation.",
            "ClinVar copy-number records spanning the 15q11.1-q11.2 region include POTEB3 among many genes and are not POTEB3-specific pathogenic alleles.",
        ],
        "condition_research_overview": [
            "POTE-family cancer-testis antigen, spermatogenesis, and reproductive-tissue expression research.",
            "15q11.1-q11.2 segmental duplication and copy-number interpretation.",
            "Paralog-aware mapping and annotation differences between GRCh37 and GRCh38.",
        ],
        "methylation_interpretation": (
            "No POTEB3-specific methylation whitelist is bundled because the local EPIC manifest is hg19-oriented and the current POTEB3 primary interval is GRCh38-only in the sources checked. "
            "Any methylation analysis should be treated as unavailable unless the user supplies GRCh38-aware probe mappings or validated POTEB3-specific assays."
        ),
        "methylation_effects": [
            "The bundled POTEB3 whitelist is intentionally empty.",
            "Do not infer POTEB3 promoter methylation from neighboring POTEB/POTEB2 or other POTE-family probes without paralog-aware remapping.",
        ],
        "methylation_condition_research": [
            "Use custom GRCh38 probe annotation, long-read mapping, RNA expression, or copy-number assays if POTEB3 regulation is the research question.",
            "Keep POTEB3 methylation separate from broader POTE-family methylation until probe uniqueness has been validated.",
        ],
        "evidence": [
            _evidence("NCBI Gene 102724631: POTEB3 gene summary and GRCh38 coordinates", "https://www.ncbi.nlm.nih.gov/gene/102724631"),
            _evidence("Ensembl ENSG00000278522: POTEB3 GRCh38 gene model", "https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=ENSG00000278522"),
            _evidence("UniProt A0JP26: POTEB3 / POTB3_HUMAN protein entry", "https://www.uniprot.org/uniprotkb/A0JP26/entry"),
            _evidence("Human Protein Atlas: POTEB3 tissue-enriched testis expression context", "https://www.proteinatlas.org/ENSG00000278522-POTEB3/tissue"),
            _evidence("PMCID PMC139254: foundational POTE family expression and paralog study", "https://pmc.ncbi.nlm.nih.gov/articles/PMC139254/"),
            _evidence("ClinVar RCV000136295: benign 15q11.1-q11.2 CNV including POTEB3", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000136295/"),
        ],
        "variants": [
            {
                "variant": "15q11.1-q11.2 CNV including POTEB3",
                "display_name": "15q11.1-q11.2 copy-number region including POTEB3",
                "common_name": "Benign ClinVar regional CNV context",
                "position": None,
                "lookup_keys": [
                    "RCV000136295",
                    "nsv533590",
                    "nssv707330",
                    "15q11.1-q11.2 CNV",
                    "POTEB3 copy-number region",
                ],
                "region_class": "structural_region",
                "interpretation_scope": "Regional copy-number context / not POTEB3-specific",
                "clinical_interpretation": (
                    "ClinVar RCV000136295 describes a copy-number loss spanning a broad 15q11.1-q11.2 interval that includes POTEB3 and many neighboring genes, with a benign germline classification from one submission. "
                    "Use this as regional CNV context only; it is not evidence that POTEB3 point variants or POTEB3 dosage alone are clinically benign or pathogenic."
                ),
                "clinical_significance": "Benign ClinVar regional CNV; not a POTEB3-specific pathogenic or protective variant.",
                "functional_effects": [
                    "Large regional copy-number event rather than a single POTEB3 coding or regulatory variant.",
                    "Interpretation is dominated by 15q11.2 segmental-duplication structure and multi-gene CNV context.",
                ],
                "associated_conditions": [
                    "15q11.1-q11.2 copy-number variation",
                    "Segmental-duplication and paralog-aware structural-variant interpretation",
                ],
                "research_context": [
                    "Confirm assembly, breakpoints, and copy-number method before making any POTEB3-specific claim.",
                    "Short-read SNV rows should not be promoted to POTEB3-specific findings without uniqueness and mapping checks.",
                ],
                "usual_variant_note": "Regional CNV context that includes POTEB3 among many 15q11.2 genes.",
                "methylation_interpretation": (
                    "Regional copy-number context does not provide a POTEB3 methylation biomarker, especially without GRCh38-aware probe uniqueness checks."
                ),
                "is_assayable_in_snp_vcf": False,
                "evidence": [
                    _evidence("ClinVar RCV000136295: benign copy-number loss spanning POTEB3", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000136295/"),
                    _evidence("NCBI Gene 102724631: POTEB3 gene identity", "https://www.ncbi.nlm.nih.gov/gene/102724631"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar RCV000136295 / ISCA submission",
                        "genotypes": "15q11.1-q11.2 regional copy-number loss",
                        "phenotype": "See-cases regional CNV submission",
                        "finding": "The CNV spans POTEB3 and many other genes and is classified as benign in the cited ClinVar record; it should be used as broad regional context rather than a POTEB3-specific variant interpretation.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/RCV000136295/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from POTEB3/POTE-family expression, paralogy, and 15q11.2 copy-number literature.",
        "population_coverage_note": (
            "The bundled POTEB3 population database is literature oriented and does not include a complete allele-frequency panel. "
            "Because POTEB3 is highly paralogous and lacks a direct GRCh37 primary-locus mapping in the checked Ensembl assembly-map result, population interpretation should prioritize assembly, copy-number, and mapping-method caveats."
        ),
        "population_sources": [
            _evidence("NCBI Gene 102724631: POTEB3 gene identity and expression note", "https://www.ncbi.nlm.nih.gov/gene/102724631"),
            _evidence("Human Protein Atlas: POTEB3 testis-enriched RNA expression", "https://www.proteinatlas.org/ENSG00000278522-POTEB3/tissue"),
            _evidence("PMCID PMC139254: POTE family expression and paralogy", "https://pmc.ncbi.nlm.nih.gov/articles/PMC139254/"),
            _evidence("ClinVar RCV000136295: regional CNV including POTEB3", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000136295/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "POTE-family paralogy",
                "location_group": "Global pattern",
                "summary": "POTEB3 interpretation should preserve POTE-family paralogy and mapping ambiguity, especially for short-read or hg19-oriented datasets.",
            },
            {
                "variant": "POTEB3 expression",
                "location_group": "Functional biology",
                "summary": "Current expression resources frame POTEB3 as testis-enriched or restricted, matching the broader POTE-family cancer-testis and reproductive-tissue literature.",
            },
            {
                "variant": "15q11.2 regional CNV",
                "location_group": "Structural variation",
                "summary": "ClinVar regional 15q11.1-q11.2 CNVs can include POTEB3, but they should be interpreted as multi-gene structural events rather than POTEB3-specific SNV evidence.",
            },
        ],
    },
    {
        "gene_name": "BLTP3B",
        "cytoband": "12q23.1",
        "chromosome": "12",
        "start": 100430850,
        "end": 100536652,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 23074 reports BLTP3B, formerly UHRF1BP1L/SHIP164, on GRCh37.p13 at "
            "NC_000012.11:100430850..100536652, complement"
        ),
        "manifest_filter_region": "12:100430850-100537652",
        "gene_summary": (
            "BLTP3B, historically known as UHRF1BP1L or SHIP164, encodes bridge-like lipid transfer protein family member 3B. "
            "The protein is a large chorein-motif lipid-transfer factor linked to early endosome-to-Golgi traffic, GARP/syntaxin-6-associated sorting, "
            "and intermembrane lipid transfer biology."
        ),
        "clinical_context": (
            "The local BLTP3B knowledge base is membrane-trafficking and research-association oriented. "
            "It focuses on lipid-transfer/endosome-Golgi biology and on MYP3 high-grade myopia locus evidence for the former UHRF1BP1L symbol, "
            "not on a validated high-penetrance diagnostic BLTP3B variant panel."
        ),
        "variant_effect_overview": [
            "Most interpretable inherited BLTP3B signals in this bundle are research association markers rather than diagnostic pathogenic variants.",
            "Functional interpretation centers on bridge-like lipid transfer, early endosome-to-Golgi trafficking, and syntaxin-6/GARP-associated sorting biology.",
            "MYP3 locus markers near or within BLTP3B/UHRF1BP1L should remain cohort-level high-grade myopia context unless independent clinical evidence is added.",
        ],
        "condition_research_overview": [
            "Early endosome-to-Golgi traffic, GARP/syntaxin-6-associated sorting, and bulk lipid-transfer biology.",
            "High-grade myopia and MYP3 chromosome 12q21-q23 association-mapping studies.",
            "Membrane-contact-site and VPS13/ATG2-like bridge lipid transport research.",
        ],
        "methylation_interpretation": (
            "BLTP3B is a reverse-strand gene and the bundled EPIC slice intentionally includes the 1 kb downstream promoter/TSS window. "
            "Many manifest rows still use the legacy UHRF1BP1L annotation; the curated whitelist therefore anchors interpretation by probe ID around the BLTP3B TSS. "
            "Treat beta values as promoter-proximal regulatory context, not as a validated BLTP3B disease methylation assay."
        ),
        "methylation_effects": [
            "The bundled BLTP3B whitelist prioritizes TSS200/TSS1500 and 5'UTR/first-exon EPIC probes around the reverse-strand TSS at 100536652.",
            "No source-backed BLTP3B methylation threshold is bundled; high or low beta values should be interpreted alongside tissue, cell composition, expression, and variant evidence.",
        ],
        "methylation_condition_research": [
            "Use BLTP3B methylation as a local regulatory layer when investigating membrane-trafficking, ocular-development, or MYP3-locus hypotheses.",
            "Because vendor annotations often use UHRF1BP1L, probe-ID whitelist matching is preferred over gene-name-only matching for this locus.",
        ],
        "evidence": [
            _evidence("NCBI Gene 23074: BLTP3B gene summary and GRCh37 coordinates", "https://www.ncbi.nlm.nih.gov/gene/23074"),
            _evidence("UniProt A0JNW5: BLTP3B / SHIP164 protein entry", "https://www.uniprot.org/uniprotkb/A0JNW5/entry"),
            _evidence("PMCID PMC9067936: SHIP164 lipid-transfer and endosome-Golgi traffic study", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9067936/"),
            _evidence("PMCID PMC3621505: high-grade myopia MYP3 association mapping", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3621505/"),
        ],
        "variants": [
            {
                "variant": "rs7134216",
                "display_name": "rs7134216 (BLTP3B/UHRF1BP1L MYP3-region marker)",
                "common_name": "UHRF1BP1L intronic MYP3 high-grade myopia association marker",
                "position": 100430850,
                "lookup_keys": [
                    "rs7134216",
                    "12:100430850",
                    "12:100430850:C>T",
                    "12:100430850:T>C",
                    "BLTP3B rs7134216",
                    "UHRF1BP1L rs7134216",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Research association / MYP3 high-grade myopia locus marker",
                "clinical_interpretation": (
                    "rs7134216 was reported as an intronic UHRF1BP1L marker in the MYP3 high-grade myopia association-mapping study and replicated for the quantitative sphere trait in an independent high-grade myopia cohort. "
                    "The local database treats a match as ocular-development and MYP3-locus research context, not as a diagnostic or deterministic myopia-risk call."
                ),
                "clinical_significance": "Research association marker; not a pathogenic clinical BLTP3B allele.",
                "functional_effects": [
                    "Intronic marker in the legacy UHRF1BP1L/BLTP3B locus used in high-grade myopia association mapping.",
                    "No bundled functional mechanism directly assigns this SNP to altered BLTP3B protein function.",
                ],
                "associated_conditions": [
                    "High-grade myopia / MYP3 locus research",
                    "Quantitative spherical refractive error association studies",
                ],
                "research_context": [
                    "Interpret through cohort, ancestry, linkage, and local LD context rather than as a monogenic BLTP3B result.",
                    "Use current dbSNP/gnomAD/clinical databases before escalating this marker beyond exploratory research context.",
                ],
                "usual_variant_note": "MYP3 high-grade myopia association marker in the legacy UHRF1BP1L locus.",
                "methylation_interpretation": (
                    "Nearby BLTP3B/UHRF1BP1L TSS methylation can provide local regulatory context but does not establish the effect direction of rs7134216."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("PMCID PMC3621505: rs7134216 in UHRF1BP1L and MYP3 association mapping", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3621505/"),
                    _evidence("NCBI Gene 23074: BLTP3B coordinates and aliases", "https://www.ncbi.nlm.nih.gov/gene/23074"),
                ],
                "literature_findings": [
                    {
                        "paper": "Hawthorne et al., 2013 (PMCID PMC3621505)",
                        "genotypes": "rs7134216 in the UHRF1BP1L/BLTP3B locus",
                        "phenotype": "High-grade myopia and quantitative spherical refractive error",
                        "finding": "The study reported rs7134216 as an intronic UHRF1BP1L marker in the MYP3 locus and observed replication for the quantitative sphere trait in an independent high-grade myopia cohort.",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3621505/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from BLTP3B/UHRF1BP1L membrane-trafficking and MYP3-locus literature.",
        "population_coverage_note": (
            "The bundled BLTP3B population database is literature oriented. "
            "It does not embed a complete allele-frequency panel, and rs7134216 interpretation should be checked against current population and phenotype-specific resources before individual-level use."
        ),
        "population_sources": [
            _evidence("NCBI Gene 23074: BLTP3B gene summary", "https://www.ncbi.nlm.nih.gov/gene/23074"),
            _evidence("PMCID PMC9067936: SHIP164 lipid-transfer and endosome-Golgi traffic", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9067936/"),
            _evidence("PMCID PMC3621505: MYP3 high-grade myopia association mapping", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3621505/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "BLTP3B lipid-transfer biology",
                "location_group": "Functional biology",
                "summary": "BLTP3B population interpretation should start from a conservative functional lane: the protein is best supported as a bridge-like lipid-transfer factor in early endosome-to-Golgi traffic.",
            },
            {
                "variant": "rs7134216 / MYP3 locus",
                "location_group": "Disease cohorts",
                "summary": "rs7134216 belongs in a cohort-level high-grade myopia/MYP3 research frame, especially for quantitative refractive-error association rather than monogenic diagnosis.",
            },
            {
                "variant": "BLTP3B TSS methylation",
                "location_group": "Local regulatory context",
                "summary": "The bundled promoter/TSS EPIC probes provide local regulatory context for BLTP3B/UHRF1BP1L, but no disease-specific methylation threshold is bundled.",
            },
        ],
    },
    {
        "gene_name": "CIROP",
        "cytoband": "14q11.2",
        "chromosome": "14",
        "start": 23568271,
        "end": 23574198,
        "strand": "-",
        "coordinate_source": (
            "NCBI Gene 100128908 reports CIROP on GRCh37.p13 at NC_000014.8:23568271..23574198, complement."
        ),
        "curated_methylation_probe_ids": ["cg19577365", "cg11790074"],
        "promoter_hotspot_region": {
            "label": "CIROP TSS-proximal EPIC probe window",
            "start": 23573850,
            "end": 23574175,
            "definition": (
                "Two bundled EPIC manifest probes fall within the reverse-strand CIROP transcribed interval near the transcription start. "
                "They are treated as local regulatory context rather than as a validated diagnostic methylation assay."
            ),
        },
        "gene_summary": (
            "CIROP, previously known as LMLN2, encodes ciliated left-right organizer metallopeptidase. "
            "The gene is implicated in ciliated left-right organizer biology, vertebrate left-right axis specification, and autosomal visceral heterotaxy 12."
        ),
        "clinical_context": (
            "The local CIROP knowledge base is developmental-genetics oriented. "
            "It focuses on rare heterotaxy 12 variants and left-right asymmetry biology rather than common adult trait prediction."
        ),
        "variant_effect_overview": [
            "Loss-of-function and selected missense CIROP variants have been reported in recessive situs anomalies and visceral heterotaxy 12.",
            "CIROP interpretation should be handled as rare developmental-disease context, with zygosity, phase, inheritance, and phenotype matching central to any clinical follow-up.",
            "A single heterozygous CIROP marker in an exploratory VCF should not be promoted to a diagnosis without external clinical confirmation.",
        ],
        "condition_research_overview": [
            "Autosomal visceral heterotaxy 12 and situs anomaly cohorts.",
            "Ciliated left-right organizer signaling upstream of asymmetric developmental patterning.",
            "Rare variant interpretation in congenital heart and laterality-disorder sequencing.",
        ],
        "methylation_interpretation": (
            "CIROP has two bundled EPIC probes close to the reverse-strand transcription start, but vendor gene annotations may not name CIROP on those rows. "
            "Use CIROP methylation as local regulatory context only, and keep rare coding-variant interpretation separate from CpG beta summaries."
        ),
        "methylation_effects": [
            "The curated CIROP whitelist includes cg19577365 and cg11790074 because they sit near the GRCh37 CIROP TSS.",
            "No source-backed CIROP methylation biomarker is bundled; beta values should not be interpreted as heterotaxy risk by themselves.",
        ],
        "methylation_condition_research": [
            "If CIROP regulation is the research question, use dedicated developmental tissue, expression, or functional data rather than relying on peripheral EPIC methylation alone.",
        ],
        "evidence": [
            _evidence("NCBI Gene 100128908: CIROP gene summary and GRCh37 coordinates", "https://www.ncbi.nlm.nih.gov/gene/100128908"),
            _evidence("PubMed 34903892: CIROP and vertebrate left-right asymmetry", "https://pubmed.ncbi.nlm.nih.gov/34903892/"),
            _evidence("ClinVar Miner: pathogenic CIROP variants for heterotaxy 12", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
            _evidence("ClinVar: CIROP c.571C>T / p.Arg191Ter", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344495/"),
        ],
        "variants": [
            {
                "variant": "CIROP c.92C>T",
                "display_name": "CIROP c.92C>T / p.Ser31Phe",
                "common_name": "S31F heterotaxy 12 missense marker",
                "position": 23574038,
                "lookup_keys": [
                    "CIROP c.92C>T",
                    "NM_001354640.2:c.92C>T",
                    "NM_001354640.2(CIROP):c.92C>T",
                    "p.Ser31Phe",
                    "S31F",
                    "rs553352307",
                    "14:23574038",
                    "14:23574038:G>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "ClinVar heterotaxy 12 marker / recessive developmental-disease context",
                "clinical_interpretation": (
                    "CIROP c.92C>T / p.Ser31Phe is listed among pathogenic CIROP variants for autosomal visceral heterotaxy 12. "
                    "Because CIROP disease interpretation is recessive and phenotype dependent, use this marker as a rare laterality-disorder flag that needs phase, inheritance, and clinical correlation."
                ),
                "clinical_significance": "ClinVar pathogenic for heterotaxy, visceral, 12, autosomal; conflicting broader-condition assertions may exist.",
                "functional_effects": [
                    "Missense change near the N terminus of CIROP.",
                    "Reported in CIROP heterotaxy 12 variant catalogs.",
                ],
                "associated_conditions": [
                    "Heterotaxy, visceral, 12, autosomal",
                    "Situs anomaly and left-right patterning research",
                ],
                "research_context": [
                    "Interpret as rare developmental-disease evidence, not as a common trait marker.",
                    "Confirm ancestry-frequency, phase, and phenotype match before clinical use.",
                ],
                "usual_variant_note": "Pathogenic CIROP heterotaxy 12 missense marker with interpretation caveats.",
                "methylation_interpretation": (
                    "Methylation values near the CIROP TSS do not establish pathogenicity for this coding variant."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CIROP c.92C>T / p.Ser31Phe", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1335920/"),
                    _evidence("ClinVar Miner: pathogenic CIROP heterotaxy 12 variants", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar / CIROP heterotaxy 12 submissions",
                        "genotypes": "NM_001354640.2:c.92C>T",
                        "phenotype": "Autosomal visceral heterotaxy 12",
                        "finding": "The variant is curated as a CIROP heterotaxy 12 marker, but interpretation should retain recessive-inheritance and phenotype-match caveats.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/1335920/",
                    }
                ],
            },
            {
                "variant": "CIROP c.571C>T",
                "display_name": "CIROP c.571C>T / p.Arg191Ter",
                "common_name": "R191* truncating heterotaxy 12 marker",
                "position": 23572916,
                "lookup_keys": [
                    "CIROP c.571C>T",
                    "NM_001354640.2:c.571C>T",
                    "NM_001354640.2(CIROP):c.571C>T",
                    "p.Arg191Ter",
                    "R191*",
                    "rs764530848",
                    "14:23572916",
                    "14:23572916:G>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic ClinVar truncating variant / recessive heterotaxy 12 context",
                "clinical_interpretation": (
                    "CIROP c.571C>T / p.Arg191Ter creates a premature stop in the CIROP coding sequence and is classified in ClinVar as pathogenic for autosomal visceral heterotaxy 12. "
                    "In an exploratory run, this should trigger rare-variant review rather than an app-only clinical conclusion."
                ),
                "clinical_significance": "Pathogenic ClinVar germline variant for heterotaxy, visceral, 12, autosomal.",
                "functional_effects": [
                    "Nonsense variant expected to truncate CIROP.",
                    "Reported in the CIROP left-right asymmetry and heterotaxy evidence stream.",
                ],
                "associated_conditions": [
                    "Heterotaxy, visceral, 12, autosomal",
                    "Recessive situs anomaly research",
                ],
                "research_context": [
                    "Check whether another CIROP pathogenic allele is present in trans before inferring recessive disease context.",
                    "Clinical interpretation requires phenotype, inheritance, and confirmatory testing.",
                ],
                "usual_variant_note": "Truncating CIROP heterotaxy 12 marker.",
                "methylation_interpretation": (
                    "Methylation beta values can provide local regulatory context but should not dilute the variant-level pathogenicity review."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CIROP c.571C>T / p.Arg191Ter", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344495/"),
                    _evidence("PubMed 34903892: CIROP and human recessive situs anomalies", "https://pubmed.ncbi.nlm.nih.gov/34903892/"),
                ],
                "literature_findings": [
                    {
                        "paper": "Szenker-Ravi et al., Nature Genetics 2022",
                        "genotypes": "Loss-of-function CIROP variants",
                        "phenotype": "Recessive situs anomalies and heterotaxy",
                        "finding": "The study identified human patients with loss-of-function CIROP mutations and recessive situs anomalies, supporting CIROP as essential for human left-right patterning.",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/34903892/",
                    }
                ],
            },
            {
                "variant": "CIROP c.1037G>A",
                "display_name": "CIROP c.1037G>A / p.Trp346Ter",
                "common_name": "W346* truncating heterotaxy 12 marker",
                "position": 23571650,
                "lookup_keys": [
                    "CIROP c.1037G>A",
                    "NM_001354640.2:c.1037G>A",
                    "NM_001354640.2(CIROP):c.1037G>A",
                    "p.Trp346Ter",
                    "W346*",
                    "rs1014082266",
                    "14:23571650",
                    "14:23571650:C>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic ClinVar truncating variant / recessive heterotaxy 12 context",
                "clinical_interpretation": (
                    "CIROP c.1037G>A / p.Trp346Ter is a truncating CIROP variant listed as pathogenic for autosomal visceral heterotaxy 12. "
                    "Treat a matched sample row as high-priority rare developmental-disease context that needs clinical-grade confirmation."
                ),
                "clinical_significance": "Pathogenic ClinVar variant for heterotaxy, visceral, 12, autosomal.",
                "functional_effects": [
                    "Nonsense variant expected to truncate CIROP.",
                    "Supports a loss-of-function CIROP thesis when paired with inheritance and phenotype evidence.",
                ],
                "associated_conditions": [
                    "Heterotaxy, visceral, 12, autosomal",
                    "Left-right axis specification disorders",
                ],
                "research_context": [
                    "Prioritize phase and second-allele review because CIROP heterotaxy is recessive.",
                ],
                "usual_variant_note": "Truncating CIROP heterotaxy 12 marker.",
                "methylation_interpretation": "Nearby CIROP methylation probes do not replace variant-level confirmation.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CIROP c.1037G>A / p.Trp346Ter", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344490/"),
                    _evidence("ClinVar Miner: pathogenic CIROP heterotaxy 12 variants", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar / CIROP heterotaxy 12 submissions",
                        "genotypes": "NM_001354640.2:c.1037G>A",
                        "phenotype": "Autosomal visceral heterotaxy 12",
                        "finding": "The variant is cataloged as a pathogenic CIROP truncating marker for heterotaxy 12.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344490/",
                    }
                ],
            },
            {
                "variant": "CIROP c.1151C>T",
                "display_name": "CIROP c.1151C>T / p.Ser384Leu",
                "common_name": "S384L heterotaxy 12 missense marker",
                "position": 23571459,
                "lookup_keys": [
                    "CIROP c.1151C>T",
                    "NM_001354640.2:c.1151C>T",
                    "NM_001354640.2(CIROP):c.1151C>T",
                    "p.Ser384Leu",
                    "S384L",
                    "rs183023758",
                    "14:23571459",
                    "14:23571459:G>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic ClinVar missense variant / recessive heterotaxy 12 context",
                "clinical_interpretation": (
                    "CIROP c.1151C>T / p.Ser384Leu is listed as pathogenic for autosomal visceral heterotaxy 12. "
                    "A matched exploratory VCF call should be interpreted through rare laterality-disorder genetics, not as a broad adult phenotype predictor."
                ),
                "clinical_significance": "Pathogenic ClinVar variant for heterotaxy, visceral, 12, autosomal.",
                "functional_effects": [
                    "Missense variant in the CIROP coding sequence.",
                    "Reported in the pathogenic CIROP heterotaxy 12 variant set.",
                ],
                "associated_conditions": [
                    "Heterotaxy, visceral, 12, autosomal",
                    "Situs anomaly sequencing studies",
                ],
                "research_context": [
                    "Confirm zygosity and phase before elevating the finding beyond a local research flag.",
                ],
                "usual_variant_note": "Pathogenic CIROP heterotaxy 12 missense marker.",
                "methylation_interpretation": "Nearby CIROP methylation is regulatory context, not a pathogenicity modifier in this bundle.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CIROP c.1151C>T / p.Ser384Leu", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344488/"),
                    _evidence("ClinVar Miner: pathogenic CIROP heterotaxy 12 variants", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar / CIROP heterotaxy 12 submissions",
                        "genotypes": "NM_001354640.2:c.1151C>T",
                        "phenotype": "Autosomal visceral heterotaxy 12",
                        "finding": "The variant is cataloged as a pathogenic CIROP missense marker for heterotaxy 12.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344488/",
                    }
                ],
            },
            {
                "variant": "CIROP c.1166G>T",
                "display_name": "CIROP c.1166G>T / p.Arg389Ile",
                "common_name": "R389I heterotaxy 12 missense marker",
                "position": 23571444,
                "lookup_keys": [
                    "CIROP c.1166G>T",
                    "NM_001354640.2:c.1166G>T",
                    "NM_001354640.2(CIROP):c.1166G>T",
                    "p.Arg389Ile",
                    "R389I",
                    "rs2140282332",
                    "14:23571444",
                    "14:23571444:C>A",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic ClinVar missense variant / recessive heterotaxy 12 context",
                "clinical_interpretation": (
                    "CIROP c.1166G>T / p.Arg389Ile is listed as pathogenic for autosomal visceral heterotaxy 12. "
                    "The strongest interpretation is a rare developmental-disease marker that needs clinical confirmation and inheritance review."
                ),
                "clinical_significance": "Pathogenic ClinVar variant for heterotaxy, visceral, 12, autosomal.",
                "functional_effects": [
                    "Missense variant in the CIROP coding sequence.",
                    "Reported in the pathogenic CIROP heterotaxy 12 variant set.",
                ],
                "associated_conditions": [
                    "Heterotaxy, visceral, 12, autosomal",
                    "Left-right patterning disorder research",
                ],
                "research_context": [
                    "Interpret with rare-disease quality controls, phenotype review, and parental testing when relevant.",
                ],
                "usual_variant_note": "Pathogenic CIROP heterotaxy 12 missense marker.",
                "methylation_interpretation": "Nearby CIROP methylation does not establish clinical significance for this missense variant.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CIROP c.1166G>T / p.Arg389Ile", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344493/"),
                    _evidence("ClinVar Miner: pathogenic CIROP heterotaxy 12 variants", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar / CIROP heterotaxy 12 submissions",
                        "genotypes": "NM_001354640.2:c.1166G>T",
                        "phenotype": "Autosomal visceral heterotaxy 12",
                        "finding": "The variant is cataloged as a pathogenic CIROP missense marker for heterotaxy 12.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/1344493/",
                    }
                ],
            },
            {
                "variant": "CIROP c.1364TCT[1]",
                "display_name": "CIROP c.1364TCT[1] / p.Phe456del",
                "common_name": "F456del heterotaxy 12 in-frame deletion",
                "position": 23571061,
                "lookup_keys": [
                    "CIROP c.1364TCT[1]",
                    "NM_001354640.2:c.1364TCT[1]",
                    "NM_001354640.2(CIROP):c.1364TCT[1]",
                    "p.Phe456del",
                    "F456del",
                    "rs1392604380",
                    "14:23571061",
                    "14:23571061:CAGA>C",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "Pathogenic ClinVar in-frame deletion / recessive heterotaxy 12 context",
                "clinical_interpretation": (
                    "CIROP c.1364TCT[1] / p.Phe456del is an in-frame deletion listed as pathogenic for autosomal visceral heterotaxy 12. "
                    "Because deletion representation can vary across VCF normalizers, rsID, transcript HGVS, and coordinate evidence should be reconciled during review."
                ),
                "clinical_significance": "Pathogenic ClinVar variant for heterotaxy, visceral, 12, autosomal.",
                "functional_effects": [
                    "In-frame deletion affecting the CIROP coding sequence.",
                    "Reported in the pathogenic CIROP heterotaxy 12 variant set.",
                ],
                "associated_conditions": [
                    "Heterotaxy, visceral, 12, autosomal",
                    "Recessive situs anomaly research",
                ],
                "research_context": [
                    "Normalize indel representation before relying on coordinate-only matching.",
                    "Confirm phase and second-allele status in suspected recessive disease contexts.",
                ],
                "usual_variant_note": "Pathogenic CIROP heterotaxy 12 in-frame deletion marker.",
                "methylation_interpretation": "Nearby CIROP methylation values are separate regulatory context.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: CIROP c.1364TCT[1] / p.Phe456del", "https://www.ncbi.nlm.nih.gov/clinvar/variation/1335922/"),
                    _evidence("ClinVar Miner: pathogenic CIROP heterotaxy 12 variants", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
                ],
                "literature_findings": [
                    {
                        "paper": "ClinVar / CIROP heterotaxy 12 submissions",
                        "genotypes": "NM_001354640.2:c.1364TCT[1]",
                        "phenotype": "Autosomal visceral heterotaxy 12",
                        "finding": "The variant is cataloged as a pathogenic CIROP in-frame deletion marker for heterotaxy 12.",
                        "url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/1335922/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from CIROP heterotaxy, left-right asymmetry, and rare-variant interpretation literature.",
        "population_coverage_note": (
            "The bundled CIROP population database is gene- and literature-oriented. "
            "It is not a complete allele-frequency panel, and rare heterotaxy-marker interpretation should be checked against current ClinVar, gnomAD, inheritance, and phenotype data."
        ),
        "population_sources": [
            _evidence("NCBI Gene 100128908: CIROP gene summary and related phenotype", "https://www.ncbi.nlm.nih.gov/gene/100128908"),
            _evidence("PubMed 34903892: CIROP and human left-right asymmetry", "https://pubmed.ncbi.nlm.nih.gov/34903892/"),
            _evidence("ClinVar Miner: pathogenic CIROP variants for heterotaxy 12", "https://clinvarminer.genetics.utah.edu/variants-by-condition/Heterotaxy%2C%20visceral%2C%2012%2C%20autosomal/gene/CIROP/pathogenic"),
        ],
        "gene_population_patterns": [
            {
                "variant": "CIROP rare pathogenic variants",
                "location_group": "Heterotaxy cohorts",
                "summary": "The most interpretable CIROP signals are rare variants reported in heterotaxy or situs anomaly contexts, especially when inheritance supports a recessive model.",
            },
            {
                "variant": "CIROP left-right asymmetry module",
                "location_group": "Functional biology",
                "summary": "CIROP is best interpreted through developmental left-right organizer biology rather than through common-variant population trait screens.",
            },
            {
                "variant": "CIROP EPIC methylation probes",
                "location_group": "Local regulatory context",
                "summary": "The two bundled CIROP-proximal EPIC probes can summarize local methylation in a sample, but no population methylation threshold is bundled for heterotaxy interpretation.",
            },
        ],
    },
    {
        "gene_name": "MT-RNR1",
        "cytoband": "mitochondrial genome",
        "chromosome": "MT",
        "start": 648,
        "end": 1601,
        "strand": "+",
        "coordinate_source": (
            "NCBI Gene 4549 reports MT-RNR1 on the mitochondrial reference sequence NC_012920.1 at positions 648..1601 for both GRCh37 and GRCh38."
        ),
        "skip_manifest_subset": True,
        "curated_methylation_probe_ids": [],
        "promoter_review_region": {
            "label": "Mitochondrial control-region and upstream transcription-control review window",
            "start": 1,
            "end": 647,
            "definition": (
                "Operational mitochondrial review span before MT-RNR1 on NC_012920.1. "
                "It includes the start of the circular mitochondrial reference so low-coordinate MT VCF calls, such as positions 64, 73, 143, and 146, are surfaced alongside the MT-RNR1 interval."
            ),
        },
        "promoter_hotspot_region": {
            "label": "No EPIC methylation probe hotspot",
            "start": 1,
            "end": 647,
            "definition": (
                "MT-RNR1 is a mitochondrial rRNA gene and the bundled EPIC manifest workflow does not provide a validated CpG methylation hotspot for this locus. "
                "This placeholder span preserves variant review across the low-coordinate mitochondrial control region."
            ),
        },
        "gene_summary": (
            "MT-RNR1 encodes the mitochondrially encoded 12S ribosomal RNA, the small-subunit rRNA of the human mitochondrial ribosome. "
            "Its sequence is clinically important because selected MT-RNR1 variants make the mitochondrial ribosomal decoding site more bacterial-like, increasing susceptibility to aminoglycoside cochleotoxicity."
        ),
        "clinical_context": (
            "The local MT-RNR1 knowledge base is pharmacogenetic and mitochondrial-hearing-loss oriented. "
            "It focuses on MT-RNR1 variants used by CPIC to assign aminoglycoside-induced hearing-loss risk, especially m.1555A>G, m.1494C>T, and m.1095T>C."
        ),
        "variant_effect_overview": [
            "High-risk MT-RNR1 variants can predispose carriers to severe, bilateral, often irreversible sensorineural hearing loss after aminoglycoside exposure.",
            "MT-RNR1 is maternally inherited and mitochondrial variant interpretation should consider heteroplasmy, tissue sampling, maternal family history, and test design.",
            "The strongest local interpretations are pharmacogenetic risk categories rather than generic mitochondrial disease calls.",
        ],
        "condition_research_overview": [
            "Aminoglycoside-induced hearing loss and ototoxicity pharmacogenetics.",
            "Mitochondrial nonsyndromic sensorineural hearing loss.",
            "Population screening and maternal-lineage studies of MT-RNR1 risk alleles.",
        ],
        "methylation_interpretation": (
            "MT-RNR1 methylation should not be inferred from the standard nuclear EPIC CpG manifest. "
            "The bundled methylation whitelist is intentionally empty; use dedicated mitochondrial assays or expression/ribosome-context data if MT-RNR1 regulation is the research question."
        ),
        "methylation_effects": [
            "No validated app-bundled EPIC CpG methylation probes are assigned to MT-RNR1.",
            "Variant interpretation should usually dominate MT-RNR1 synthesis unless custom mitochondrial methylation or expression data are supplied.",
        ],
        "methylation_condition_research": [
            "Dedicated mitochondrial DNA methylation or RNA-expression assays, when available, should be interpreted separately from nuclear EPIC methylation summaries.",
        ],
        "evidence": [
            _evidence("NCBI Gene 4549: MT-RNR1 gene summary and coordinates", "https://www.ncbi.nlm.nih.gov/gene/4549"),
            _evidence("NCBI Bookshelf: CPIC recommendations for aminoglycosides and MT-RNR1", "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/"),
            _evidence("CPIC guideline page for aminoglycosides and MT-RNR1", "https://cpicpgx.org/guidelines/cpic-guideline-for-aminoglycosides-and-mt-rnr1/"),
            _evidence("PharmGKB VIP summary for MT-RNR1", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5083147/"),
        ],
        "variants": [
            {
                "variant": "m.1555A>G",
                "display_name": "MT-RNR1 m.1555A>G",
                "common_name": "A1555G aminoglycoside ototoxicity risk allele",
                "position": 1555,
                "lookup_keys": [
                    "m.1555A>G",
                    "MT-RNR1:m.1555A>G",
                    "NC_012920.1:m.1555A>G",
                    "rs267606617",
                    "MT:1555",
                    "MT:1555:A>G",
                    "M:1555:A>G",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "CPIC increased-risk pharmacogenetic allele",
                "clinical_interpretation": (
                    "MT-RNR1 m.1555A>G is the best-established aminoglycoside-induced hearing-loss risk allele in this gene. "
                    "CPIC assigns it to the increased-risk phenotype, and ClinVar/Medical Genetics Summaries link it to aminoglycoside-induced deafness and mitochondrial nonsyndromic sensorineural hearing loss."
                ),
                "clinical_significance": "CPIC increased-risk allele; ClinVar/OMIM pathogenic hearing-loss pharmacogenetic marker.",
                "functional_effects": [
                    "Alters mitochondrial 12S rRNA in a way that increases similarity to the bacterial aminoglycoside-binding target.",
                    "Can confer high susceptibility to cochleotoxicity after systemic aminoglycoside exposure.",
                ],
                "associated_conditions": [
                    "Aminoglycoside-induced deafness",
                    "Mitochondrial nonsyndromic sensorineural hearing loss",
                    "Gentamicin, streptomycin, kanamycin, amikacin, tobramycin, and related aminoglycoside toxicity contexts",
                ],
                "research_context": [
                    "Interpret alongside heteroplasmy level and maternal inheritance.",
                    "Medication decisions require clinical review and current prescribing guidance.",
                ],
                "usual_variant_note": "Best-studied MT-RNR1 aminoglycoside ototoxicity risk allele.",
                "methylation_interpretation": (
                    "No EPIC methylation whitelist is assigned; the pharmacogenetic variant evidence is the primary interpretive layer."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: MT-RNR1 m.1555A>G", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000505667.11/"),
                    _evidence("NCBI Bookshelf: Gentamicin therapy and MT-RNR1 genotype", "https://www.ncbi.nlm.nih.gov/books/NBK285956/"),
                    _evidence("PharmGKB VIP summary for MT-RNR1", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5083147/"),
                ],
                "literature_findings": [
                    {
                        "paper": "CPIC guideline / NCBI Medical Genetics Summaries",
                        "genotypes": "NC_012920.1:m.1555A>G",
                        "phenotype": "Aminoglycoside-induced hearing-loss risk",
                        "finding": "CPIC assigns m.1555A>G to the MT-RNR1 increased-risk phenotype and recommends avoiding aminoglycosides unless infection severity and lack of alternatives justify the risk.",
                        "url": "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/",
                    }
                ],
            },
            {
                "variant": "m.1494C>T",
                "display_name": "MT-RNR1 m.1494C>T",
                "common_name": "C1494T aminoglycoside ototoxicity risk allele",
                "position": 1494,
                "lookup_keys": [
                    "m.1494C>T",
                    "MT-RNR1:m.1494C>T",
                    "NC_012920.1:m.1494C>T",
                    "rs267606619",
                    "MT:1494",
                    "MT:1494:C>T",
                    "M:1494:C>T",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "CPIC increased-risk pharmacogenetic allele",
                "clinical_interpretation": (
                    "MT-RNR1 m.1494C>T is a high-evidence CPIC increased-risk allele for aminoglycoside-induced hearing loss. "
                    "ClinVar links it to aminoglycoside-induced deafness and mitochondrial nonsyndromic sensorineural hearing loss."
                ),
                "clinical_significance": "CPIC increased-risk allele; ClinVar pathogenic or likely pathogenic hearing-loss marker depending on condition/submission.",
                "functional_effects": [
                    "Changes mitochondrial 12S rRNA near the aminoglycoside-sensitive decoding region.",
                    "Reported in aminoglycoside-induced and nonsyndromic hearing-loss cohorts.",
                ],
                "associated_conditions": [
                    "Aminoglycoside-induced deafness",
                    "Mitochondrial nonsyndromic sensorineural hearing loss",
                ],
                "research_context": [
                    "Evidence base is smaller than m.1555A>G but strong enough for CPIC increased-risk classification.",
                    "Population distribution is uneven, with many early reports from Chinese families and hearing-loss cohorts.",
                ],
                "usual_variant_note": "High-evidence MT-RNR1 aminoglycoside ototoxicity risk allele.",
                "methylation_interpretation": "No EPIC methylation whitelist is assigned for MT-RNR1.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: MT-RNR1 m.1494C>T", "https://www.ncbi.nlm.nih.gov/clinvar/298330196/"),
                    _evidence("NCBI Bookshelf: CPIC recommendations for MT-RNR1", "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/"),
                ],
                "literature_findings": [
                    {
                        "paper": "CPIC guideline / NCBI Medical Genetics Summaries",
                        "genotypes": "NC_012920.1:m.1494C>T",
                        "phenotype": "Aminoglycoside-induced hearing-loss risk",
                        "finding": "CPIC assigns m.1494C>T to the increased-risk phenotype for aminoglycoside-induced hearing loss.",
                        "url": "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/",
                    }
                ],
            },
            {
                "variant": "m.1095T>C",
                "display_name": "MT-RNR1 m.1095T>C",
                "common_name": "T1095C aminoglycoside ototoxicity risk allele",
                "position": 1095,
                "lookup_keys": [
                    "m.1095T>C",
                    "MT-RNR1:m.1095T>C",
                    "NC_012920.1:m.1095T>C",
                    "rs267606618",
                    "MT:1095",
                    "MT:1095:T>C",
                    "M:1095:T>C",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "CPIC increased-risk pharmacogenetic allele",
                "clinical_interpretation": (
                    "MT-RNR1 m.1095T>C is included by CPIC as an increased-risk genotype for aminoglycoside-induced hearing loss, with a more moderate evidence base than m.1555A>G and m.1494C>T."
                ),
                "clinical_significance": "CPIC increased-risk allele; ClinVar/OMIM pathogenic aminoglycoside-induced deafness marker.",
                "functional_effects": [
                    "Mitochondrial 12S rRNA variant associated with aminoglycoside ototoxicity susceptibility.",
                    "Evidence should be interpreted with more caution than the two high-evidence MT-RNR1 risk alleles.",
                ],
                "associated_conditions": [
                    "Aminoglycoside-induced deafness",
                    "Mitochondrial nonsyndromic sensorineural hearing loss research",
                ],
                "research_context": [
                    "Use as a CPIC increased-risk pharmacogenetic flag while preserving evidence-strength caveats.",
                ],
                "usual_variant_note": "Moderate-evidence CPIC increased-risk MT-RNR1 allele.",
                "methylation_interpretation": "No EPIC methylation whitelist is assigned for MT-RNR1.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("ClinVar: MT-RNR1 m.1095T>C", "https://www.ncbi.nlm.nih.gov/clinvar/RCV000010259/"),
                    _evidence("NCBI Bookshelf: CPIC recommendations for MT-RNR1", "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/"),
                ],
                "literature_findings": [
                    {
                        "paper": "CPIC guideline / NCBI Medical Genetics Summaries",
                        "genotypes": "NC_012920.1:m.1095T>C",
                        "phenotype": "Aminoglycoside-induced hearing-loss risk",
                        "finding": "CPIC includes m.1095T>C among MT-RNR1 increased-risk genotypes for aminoglycoside-induced hearing loss.",
                        "url": "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/",
                    }
                ],
            },
            {
                "variant": "m.827A>G",
                "display_name": "MT-RNR1 m.827A>G",
                "common_name": "A827G CPIC normal-risk example allele",
                "position": 827,
                "lookup_keys": [
                    "m.827A>G",
                    "MT-RNR1:m.827A>G",
                    "NC_012920.1:m.827A>G",
                    "MT:827",
                    "MT:827:A>G",
                    "M:827:A>G",
                ],
                "region_class": "gene_body",
                "interpretation_scope": "CPIC normal-risk example allele / hearing-loss literature caveat",
                "clinical_interpretation": (
                    "MT-RNR1 m.827A>G has appeared in nonsyndromic hearing-loss literature, but CPIC uses it as an example of a normal-risk MT-RNR1 genotype for aminoglycoside-induced hearing loss. "
                    "The local database includes it to prevent overcalling every MT-RNR1 change as an aminoglycoside contraindication."
                ),
                "clinical_significance": "CPIC normal-risk example for aminoglycoside-induced hearing loss.",
                "functional_effects": [
                    "Reported in hearing-loss families and cohorts, but CPIC does not classify it as increased AIHL risk.",
                ],
                "associated_conditions": [
                    "Mitochondrial nonsyndromic sensorineural hearing-loss research",
                    "Aminoglycoside risk classification contrast allele",
                ],
                "research_context": [
                    "Do not treat m.827A>G as a CPIC increased-risk aminoglycoside allele without newer expert guidance.",
                ],
                "usual_variant_note": "Normal-risk contrast allele in CPIC aminoglycoside recommendations.",
                "methylation_interpretation": "No EPIC methylation whitelist is assigned for MT-RNR1.",
                "is_assayable_in_snp_vcf": True,
                "evidence": [
                    _evidence("NCBI Bookshelf: CPIC recommendations for MT-RNR1", "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/"),
                    _evidence("NCBI Gene 4549: related MT-RNR1 hearing-loss literature", "https://www.ncbi.nlm.nih.gov/gene/4549"),
                ],
                "literature_findings": [
                    {
                        "paper": "CPIC guideline / NCBI Medical Genetics Summaries",
                        "genotypes": "NC_012920.1:m.827A>G",
                        "phenotype": "Aminoglycoside-induced hearing-loss risk classification",
                        "finding": "CPIC uses m.827A>G as a normal-risk example for aminoglycoside-induced hearing loss while still recommending standard aminoglycoside precautions.",
                        "url": "https://www.ncbi.nlm.nih.gov/books/NBK285956/table/gentamicin.T.the_cpic_recommendations_fo/",
                    }
                ],
            },
        ],
        "population_intro": "Broader population patterns curated from MT-RNR1 pharmacogenetic, hearing-loss, and maternal-lineage literature.",
        "population_coverage_note": (
            "The bundled MT-RNR1 population database is literature and guideline oriented. "
            "Mitochondrial variants are not represented like diploid nuclear variants in many exome/genome population panels, and interpretation should consider heteroplasmy, haplogroup background, and maternal inheritance."
        ),
        "population_sources": [
            _evidence("NCBI Gene 4549: MT-RNR1 gene summary and literature links", "https://www.ncbi.nlm.nih.gov/gene/4549"),
            _evidence("NCBI Bookshelf: Gentamicin therapy and MT-RNR1 genotype", "https://www.ncbi.nlm.nih.gov/books/NBK285956/"),
            _evidence("PharmGKB VIP summary for MT-RNR1", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5083147/"),
        ],
        "gene_population_patterns": [
            {
                "variant": "m.1555A>G",
                "location_group": "Global pattern",
                "summary": "m.1555A>G is the most widely reported MT-RNR1 aminoglycoside ototoxicity allele, with prevalence and penetrance shaped by population background, aminoglycoside exposure, heteroplasmy, and modifying variants.",
            },
            {
                "variant": "m.1494C>T",
                "location_group": "Disease cohorts",
                "summary": "m.1494C>T is rarer than m.1555A>G but has strong guideline support as an increased-risk allele; many reported families and cohorts are East Asian, especially Chinese, though it is not exclusive to that ancestry.",
            },
            {
                "variant": "m.1095T>C",
                "location_group": "Disease cohorts",
                "summary": "m.1095T>C is curated as a CPIC increased-risk allele with more moderate evidence, so population interpretation should preserve uncertainty around penetrance and study size.",
            },
            {
                "variant": "m.827A>G",
                "location_group": "Global pattern",
                "summary": "m.827A>G is included as a normal-risk contrast allele in CPIC aminoglycoside recommendations; this helps separate hearing-loss literature signals from expert pharmacogenetic risk categories.",
            },
        ],
    },
]


def _build_allergic_inflammation_evidence(config: dict[str, Any]) -> list[dict[str, str]]:
    gene_name = str(config["gene_name"])
    canonical_symbol = str(config.get("canonical_symbol") or gene_name)
    ensembl_id = str(config["ensembl_id"])
    evidence = [
        _evidence(
            f"NCBI Gene {config['ncbi_id']}: {canonical_symbol} gene summary",
            f"https://www.ncbi.nlm.nih.gov/gene/{config['ncbi_id']}",
        ),
        _evidence(
            f"Ensembl GRCh37 {ensembl_id}: {canonical_symbol} coordinates and transcripts",
            f"https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g={ensembl_id}",
        ),
    ]
    if config.get("uniprot_id"):
        evidence.append(
            _evidence(
                f"UniProt {config['uniprot_id']}: {canonical_symbol} protein entry",
                f"https://www.uniprot.org/uniprotkb/{config['uniprot_id']}/entry",
            )
        )
    evidence.append(
        _evidence(
            f"GWAS Catalog gene page: {canonical_symbol}",
            f"https://www.ebi.ac.uk/gwas/genes/{canonical_symbol}",
        )
    )
    evidence.extend(config.get("extra_evidence", []))
    return evidence


def _build_allergic_inflammation_variant(
    config: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    gene_name = str(config["gene_name"])
    canonical_symbol = str(config.get("canonical_symbol") or gene_name)
    variant_name = str(variant["variant"])
    display_name = str(variant.get("display_name") or variant_name)
    common_name = str(
        variant.get("common_name")
        or f"{gene_name} allergic-inflammation research marker"
    )
    position = variant.get("position")

    lookup_keys = list(
        dict.fromkeys(
            [
                variant_name,
                display_name,
                common_name,
                f"{gene_name}:{variant_name}",
                f"{canonical_symbol}:{variant_name}",
                *variant.get("lookup_keys", []),
            ]
        )
    )
    if position is not None:
        lookup_keys.append(f"{config['chromosome']}:{position}")

    associated_conditions = list(config["conditions"])
    marker_context = str(
        variant.get("marker_context")
        or "allergic disease, asthma, eczema, airway inflammation, or immune-regulatory cohort literature"
    )

    return {
        "variant": variant_name,
        "display_name": display_name,
        "common_name": common_name,
        "position": position,
        "lookup_keys": lookup_keys,
        "region_class": variant.get("region_class", "regulatory_locus"),
        "interpretation_scope": variant.get(
            "interpretation_scope",
            "Research association / immune-airway regulatory context",
        ),
        "clinical_interpretation": variant.get(
            "clinical_interpretation",
            (
                f"{display_name} is bundled as a {gene_name} research marker for {marker_context}. "
                "Treat it as low-penetrance, cohort-aware association context rather than as a diagnostic or treatment-selection allele."
            ),
        ),
        "clinical_significance": variant.get(
            "clinical_significance",
            "Research-level allergic and immune-inflammatory association marker; not a deterministic clinical allele.",
        ),
        "functional_effects": variant.get(
            "functional_effects",
            [
                f"May tag regulatory, expression, coding, haplotype, or linked-locus variation around {gene_name}.",
                "Reported effects should be interpreted with ancestry, phenotype definition, tissue, and genotype dosage in mind.",
            ],
        ),
        "associated_conditions": associated_conditions,
        "research_context": variant.get(
            "research_context",
            [
                "Use this marker as a research context flag and not as a stand-alone diagnosis.",
                "Direction and effect size can vary across asthma, eczema, allergy, immune, epithelial, and ancestry-specific cohorts.",
                "Confirm exact allele, build, zygosity, QC, and LD/haplotype context before drawing marker-level conclusions.",
            ],
        ),
        "usual_variant_note": variant.get(
            "usual_variant_note",
            f"{gene_name} marker used for allergic-inflammation research context.",
        ),
        "methylation_interpretation": (
            f"{gene_name} methylation can provide local regulatory context, but it does not establish the variant, haplotype, expression state, or phenotype direction for {display_name}."
        ),
        "is_assayable_in_snp_vcf": bool(variant.get("is_assayable_in_snp_vcf", True)),
        "evidence": variant.get("evidence", _build_allergic_inflammation_evidence(config)),
        "literature_findings": variant.get(
            "literature_findings",
            [
                {
                    "paper": "Asthma, allergy, and immune-trait GWAS and functional follow-up literature",
                    "genotypes": f"{display_name} or linked {gene_name} locus genotype states",
                    "phenotype": ", ".join(associated_conditions[:3]),
                    "finding": (
                        f"The local curation treats {display_name} as a research-grade {gene_name} signal for allergic, airway, epithelial-barrier, or immune-inflammatory biology."
                    ),
                    "url": f"https://www.ebi.ac.uk/gwas/genes/{canonical_symbol}",
                }
            ],
        ),
    }


def _build_allergic_inflammation_gene_definition(config: dict[str, Any]) -> dict[str, Any]:
    gene_name = str(config["gene_name"])
    canonical_symbol = str(config.get("canonical_symbol") or gene_name)
    display_label = gene_name if gene_name == canonical_symbol else f"{gene_name} / {canonical_symbol}"
    evidence = _build_allergic_inflammation_evidence(config)
    variants = [
        _build_allergic_inflammation_variant(config, variant)
        for variant in config.get("variants", [])
    ]

    strand = str(config["strand"])
    start = int(config["start"])
    end = int(config["end"])
    if strand == "-":
        manifest_start = start
        manifest_end = end + 1000
    else:
        manifest_start = max(1, start - 1000)
        manifest_end = end

    return {
        "gene_name": gene_name,
        "cytoband": config["cytoband"],
        "chromosome": str(config["chromosome"]),
        "start": start,
        "end": end,
        "strand": strand,
        "coordinate_source": (
            f"Ensembl GRCh37 lookup for {config['ensembl_id']} on the primary assembly, aligned to the hg19 coordinate system used by this app"
        ),
        "manifest_filter_region": config.get(
            "manifest_filter_region",
            f"{config['chromosome']}:{manifest_start}-{manifest_end}",
        ),
        "gene_summary": config["gene_summary"],
        "clinical_context": config.get(
            "clinical_context",
            (
                f"The bundled {display_label} knowledge base is focused on asthma, allergy, atopic dermatitis, airway inflammation, epithelial-barrier biology, and immune-regulatory research. "
                "Seeded markers and locus models are intended as research context and should not be promoted to diagnostic, prognostic, or treatment-selection claims without external clinical curation."
            ),
        ),
        "variant_effect_overview": config["variant_effect_overview"],
        "condition_research_overview": config["conditions"],
        "methylation_interpretation": config.get(
            "methylation_interpretation",
            (
                f"{gene_name} methylation should be read as local regulatory context around an allergic-inflammation gene. "
                "Promoter-proximal or gene-body beta values may help frame accessibility or expression hypotheses, but they are not validated stand-alone biomarkers for asthma, eczema, allergy, immune activation, or treatment response."
            ),
        ),
        "methylation_effects": config.get(
            "methylation_effects",
            [
                f"Promoter-proximal methylation may suggest a more restrained or permissive local regulatory state for {gene_name} expression potential.",
                "Interpret methylation with tissue, cell composition, environmental exposure, disease state, medication history, and genotype context.",
            ],
        ),
        "methylation_condition_research": config.get(
            "methylation_condition_research",
            [
                "Asthma, allergy, atopic dermatitis, eosinophilic inflammation, and airway epithelial studies.",
                "Immune-cell, epithelial-barrier, cytokine-signaling, and inflammatory-disease epigenetic research.",
            ],
        ),
        "evidence": evidence,
        "variants": variants,
        "population_intro": (
            f"Broader population patterns curated from {display_label} asthma, atopy, airway, epithelial-barrier, and immune-inflammatory literature."
        ),
        "population_coverage_note": (
            f"The bundled {display_label} population database is literature oriented and does not include a complete allele-frequency panel. "
            "Interpretation should remain ancestry-, phenotype-, tissue-, assay-, and LD-aware."
        ),
        "population_sources": evidence,
        "gene_population_patterns": config.get(
            "gene_population_patterns",
            [
                {
                    "variant": variant.get("display_name", variant["variant"]),
                    "location_group": variant.get("location_group", "Disease cohorts"),
                    "summary": (
                        f"{variant.get('display_name', variant['variant'])} is curated as a low-penetrance research signal in {gene_name} or a linked locus; cohort design, ancestry, LD, and phenotype definition matter."
                    ),
                }
                for variant in config.get("variants", [])
            ]
            + [
                {
                    "variant": f"{gene_name} regulatory context",
                    "location_group": "Global pattern",
                    "summary": (
                        f"{gene_name} interpretation is most useful when sequence, methylation, expression, tissue, and allergic-immune phenotype evidence are integrated rather than treated as a single deterministic SNP call."
                    ),
                }
            ],
        ),
    }


ALLERGIC_INFLAMMATION_GENE_CONFIGS: list[dict[str, Any]] = [
    {
        "gene_name": "IL33",
        "ensembl_id": "ENSG00000137033",
        "ncbi_id": "90865",
        "uniprot_id": "O95760",
        "cytoband": "9p24.1",
        "chromosome": "9",
        "start": 6215805,
        "end": 6257983,
        "strand": "+",
        "gene_summary": "IL33 encodes interleukin-33, an IL-1-family alarmin released from stressed epithelial, endothelial, and barrier tissues and sensed through the IL1RL1/ST2 receptor axis.",
        "variant_effect_overview": [
            "IL33 variants are usually interpreted as regulatory or linked-locus modifiers of epithelial alarmin expression and type 2 inflammatory priming.",
            "Asthma and allergy GWAS signals near IL33 should be read as low-penetrance susceptibility context, not as monogenic disease alleles.",
        ],
        "conditions": ["Asthma susceptibility", "Atopic dermatitis and allergic disease", "Airway epithelial alarmin and type 2 inflammation research"],
        "variants": [
            {
                "variant": "rs1342326",
                "position": 6190076,
                "common_name": "IL33 asthma GWAS regulatory-locus marker",
                "lookup_keys": ["9:6190076:A>C"],
                "clinical_interpretation": (
                    "rs1342326 is curated as an upstream/flanking A/C marker near IL33 for asthma and allergic-inflammation research. "
                    "The A/C genomic alias is retained alongside the T/G reverse-complement notation used in parts of the literature, so strand notation should be checked before assigning direction."
                ),
                "research_context": [
                    "The marker sits about 25.7 kb upstream of the IL33 transcribed interval in the bundled GRCh37 coordinate context.",
                    "Use the A/C genomic REF/ALT alias and the T/G reverse-complement literature notation carefully when comparing sources.",
                    "Interpret rs1342326 as low-penetrance asthma and allergic-inflammation association context rather than as a diagnostic allele.",
                ],
                "evidence": [
                    _evidence("Moffatt asthma GWAS IL33 rs1342326 signal", "https://pubmed.ncbi.nlm.nih.gov/20860503/"),
                    _evidence("Ensembl GRCh37 IL33 rs1342326 A/C coordinate context", "https://grch37.ensembl.org/Homo_sapiens/Variation/Explore?v=rs1342326"),
                    _evidence("GWAS Catalog rs1342326 T/G asthma notation", "https://www.ebi.ac.uk/gwas/variants/rs1342326"),
                ],
            },
            {"variant": "rs2381416", "position": 6193455, "common_name": "IL33 asthma/allergy linked regulatory marker"},
        ],
    },
    {
        "gene_name": "IL1RL1",
        "ensembl_id": "ENSG00000115602",
        "ncbi_id": "9173",
        "uniprot_id": "Q01638",
        "cytoband": "2q12.1",
        "chromosome": "2",
        "start": 102927962,
        "end": 102968497,
        "strand": "+",
        "gene_summary": "IL1RL1 encodes the ST2 receptor chain for IL-33 and produces membrane and soluble isoforms that shape type 2 inflammation, mast-cell biology, eosinophilia, airway remodeling, and allergic immune tone.",
        "variant_effect_overview": [
            "IL1RL1 markers often sit in the broader 2q12 IL1 receptor gene cluster and can tag ST2 expression, soluble ST2 biology, eosinophil traits, asthma, or atopy context.",
            "Variant interpretation should preserve IL1RL1, IL18R1, and IL18RAP cluster LD instead of assuming a single causal gene from one SNP.",
        ],
        "conditions": ["Asthma and eosinophilic airway inflammation", "Atopy and allergic disease", "IL-33/ST2 signaling and soluble ST2 biomarker research"],
        "variants": [
            {"variant": "rs3771180", "position": 102953617, "common_name": "IL1RL1 asthma and type 2 inflammation locus marker"},
            {"variant": "rs13408661", "position": 102955082, "common_name": "IL1RL1/IL18R1 cluster asthma marker"},
        ],
    },
    {
        "gene_name": "ORMDL3",
        "ensembl_id": "ENSG00000172057",
        "ncbi_id": "94103",
        "uniprot_id": "Q8N138",
        "cytoband": "17q12-q21.1",
        "chromosome": "17",
        "start": 38077294,
        "end": 38083854,
        "strand": "-",
        "gene_summary": "ORMDL3 encodes an endoplasmic-reticulum membrane protein that regulates sphingolipid homeostasis and is embedded in the chromosome 17q12-q21 asthma locus with GSDMB and neighboring regulatory elements.",
        "variant_effect_overview": [
            "ORMDL3 locus markers often reflect 17q12-q21 regulatory haplotypes that influence ORMDL3, GSDMB, and neighboring gene expression.",
            "Reported effects are strongest in childhood asthma and allergic-airway GWAS but remain LD-, ancestry-, and tissue-context dependent.",
        ],
        "conditions": ["Childhood asthma susceptibility", "Allergic airway inflammation", "17q12-q21 regulatory haplotype and sphingolipid biology"],
        "variants": [
            {"variant": "rs7216389", "position": 38069949, "common_name": "17q12-q21 ORMDL3/GSDMB asthma haplotype marker"},
            {"variant": "rs4794820", "position": 38089344, "common_name": "ORMDL3 linked asthma regulatory marker"},
        ],
    },
    {
        "gene_name": "GSDMB",
        "ensembl_id": "ENSG00000073605",
        "ncbi_id": "55876",
        "uniprot_id": "Q8TAX9",
        "cytoband": "17q12-q21.1",
        "chromosome": "17",
        "start": 38060848,
        "end": 38076107,
        "strand": "-",
        "gene_summary": "GSDMB encodes gasdermin B, a gasdermin-family protein at the 17q12-q21 asthma locus with airway, epithelial, immune, and pyroptosis-related research relevance.",
        "variant_effect_overview": [
            "GSDMB variants can tag coding, splicing, and expression effects within the ORMDL3/GSDMB asthma haplotype block.",
            "Interpretation should separate GSDMB protein-function hypotheses from linked 17q12-q21 regulatory effects on neighboring genes.",
        ],
        "conditions": ["Asthma susceptibility", "Airway epithelial and immune-cell inflammation", "17q12-q21 GSDMB/ORMDL3 locus research"],
        "variants": [
            {"variant": "rs11078927", "position": 38064405, "common_name": "GSDMB splice/coding asthma-locus marker"},
            {"variant": "rs7216389", "position": 38069949, "common_name": "17q12-q21 GSDMB/ORMDL3 asthma haplotype marker"},
        ],
    },
    {
        "gene_name": "HLA-DQA1",
        "ensembl_id": "ENSG00000196735",
        "ncbi_id": "3117",
        "uniprot_id": "P01909",
        "cytoband": "6p21.32",
        "chromosome": "6",
        "start": 32595956,
        "end": 32614839,
        "strand": "+",
        "gene_summary": "HLA-DQA1 encodes the alpha chain of HLA-DQ, an MHC class II antigen-presentation molecule whose haplotypes influence adaptive immune recognition and many autoimmune, allergic, and inflammatory association studies.",
        "variant_effect_overview": [
            "HLA-DQA1 interpretation is usually haplotype and allele based rather than single-SNP deterministic.",
            "Nearby HLA-DQ GWAS SNPs can tag complex MHC LD and should not be treated as HLA-DQA1-specific functional alleles without HLA typing or fine mapping.",
        ],
        "conditions": ["HLA-DQ antigen presentation", "Autoimmune and allergic disease association research", "Asthma and immune-trait MHC locus context"],
        "variants": [
            {"variant": "rs9273349", "position": 32625869, "common_name": "HLA-DQ asthma and immune-trait MHC marker"},
            {"variant": "HLA-DQ haplotype model", "common_name": "HLA-DQA1 haplotype and antigen-presentation model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "HLA-DQB1",
        "ensembl_id": "ENSG00000179344",
        "ncbi_id": "3119",
        "uniprot_id": "P01920",
        "cytoband": "6p21.32",
        "chromosome": "6",
        "start": 32627244,
        "end": 32636160,
        "strand": "-",
        "gene_summary": "HLA-DQB1 encodes the beta chain of HLA-DQ, an MHC class II antigen-presentation molecule with strong haplotype-level relevance to immune, autoimmune, allergy, and asthma association studies.",
        "variant_effect_overview": [
            "HLA-DQB1 should be interpreted through HLA allele, haplotype, and MHC LD context rather than through isolated SNP rows.",
            "HLA-DQ region SNPs can be useful research tags but require fine mapping before assigning causality to HLA-DQB1.",
        ],
        "conditions": ["HLA-DQ antigen presentation", "Autoimmune and allergic disease association research", "Asthma and immune-trait MHC locus context"],
        "variants": [
            {"variant": "rs9273349", "position": 32625869, "common_name": "HLA-DQ asthma and immune-trait MHC marker"},
            {"variant": "HLA-DQ haplotype model", "common_name": "HLA-DQB1 haplotype and antigen-presentation model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "TSLP",
        "ensembl_id": "ENSG00000145777",
        "ncbi_id": "85480",
        "uniprot_id": "Q969D9",
        "cytoband": "5q22.1",
        "chromosome": "5",
        "start": 110405760,
        "end": 110413722,
        "strand": "+",
        "gene_summary": "TSLP encodes thymic stromal lymphopoietin, an epithelial cytokine that primes dendritic cells and type 2 inflammation and is a therapeutic target in asthma.",
        "variant_effect_overview": [
            "TSLP markers are usually interpreted as regulatory modifiers of epithelial cytokine tone and allergic-airway susceptibility.",
            "Because TSLP is medication-relevant, the local bundle keeps variants as research context rather than drug-response rules.",
        ],
        "conditions": ["Asthma susceptibility", "Atopic dermatitis and allergic disease", "Epithelial alarmin and type 2 inflammation research"],
        "variants": [
            {"variant": "rs1837253", "position": 110401872, "common_name": "TSLP asthma and allergic-disease regulatory marker"},
        ],
    },
    {
        "gene_name": "IL4R",
        "ensembl_id": "ENSG00000077238",
        "ncbi_id": "3566",
        "uniprot_id": "P24394",
        "cytoband": "16p12.1",
        "chromosome": "16",
        "start": 27324989,
        "end": 27376099,
        "strand": "+",
        "gene_summary": "IL4R encodes the interleukin-4 receptor alpha chain shared by IL-4 and IL-13 receptor complexes, linking JAK-STAT6 signaling to type 2 inflammation, asthma, eczema, and allergy biology.",
        "variant_effect_overview": [
            "IL4R missense and regulatory polymorphisms are studied as modifiers of atopy, asthma, IgE biology, and type 2 cytokine signaling.",
            "Variant effects are context dependent and should not be used alone for biologic therapy selection.",
        ],
        "conditions": ["Asthma and atopy", "Atopic dermatitis and IgE biology", "IL-4/IL-13 receptor and STAT6 signaling research"],
        "variants": [
            {"variant": "rs1801275", "display_name": "rs1801275 (IL4R Ile50Val)", "common_name": "IL4R missense atopy/asthma marker"},
            {"variant": "rs1805010", "display_name": "rs1805010 (IL4R Gln576Arg)", "common_name": "IL4R missense type 2 signaling marker"},
        ],
    },
    {
        "gene_name": "STAT6",
        "ensembl_id": "ENSG00000166888",
        "ncbi_id": "6778",
        "uniprot_id": "P42226",
        "cytoband": "12q13.3",
        "chromosome": "12",
        "start": 57489191,
        "end": 57525922,
        "strand": "-",
        "gene_summary": "STAT6 encodes the transcription factor activated downstream of IL-4 and IL-13 signaling, regulating type 2 immune differentiation, IgE-related programs, epithelial remodeling, and allergic inflammation.",
        "variant_effect_overview": [
            "STAT6 variants are studied as modifiers of IL-4/IL-13 transcriptional signaling, asthma, atopy, and allergic sensitization.",
            "Interpretation should focus on pathway tuning rather than high-penetrance disease prediction.",
        ],
        "conditions": ["Asthma and allergic sensitization", "Type 2 cytokine signaling", "IgE, eosinophilia, and epithelial remodeling research"],
        "variants": [
            {"variant": "rs324011", "common_name": "STAT6 regulatory asthma/atopy marker"},
            {"variant": "rs3024974", "common_name": "STAT6 allergic-disease association marker"},
        ],
    },
    {
        "gene_name": "IL13",
        "ensembl_id": "ENSG00000169194",
        "ncbi_id": "3596",
        "uniprot_id": "P35225",
        "cytoband": "5q31.1",
        "chromosome": "5",
        "start": 131991955,
        "end": 131996802,
        "strand": "+",
        "gene_summary": "IL13 encodes interleukin-13, a type 2 cytokine that drives mucus production, airway remodeling, eosinophilic inflammation, IgE biology, and epithelial barrier changes.",
        "variant_effect_overview": [
            "IL13 variants can modify type 2 cytokine signaling, atopy, asthma, IgE traits, and eczema susceptibility.",
            "Coding and promoter markers should be interpreted as low-penetrance pathway modifiers.",
        ],
        "conditions": ["Asthma and airway remodeling", "Atopy, IgE, and allergic sensitization", "Eczema and epithelial-barrier inflammation"],
        "variants": [
            {"variant": "rs20541", "display_name": "rs20541 (IL13 Arg130Gln)", "position": 131995964, "common_name": "IL13 missense atopy/asthma marker"},
            {"variant": "rs1800925", "common_name": "IL13 promoter asthma/IgE marker"},
        ],
    },
    {
        "gene_name": "IL4",
        "ensembl_id": "ENSG00000113520",
        "ncbi_id": "3565",
        "uniprot_id": "P05112",
        "cytoband": "5q31.1",
        "chromosome": "5",
        "start": 132009678,
        "end": 132018368,
        "strand": "+",
        "gene_summary": "IL4 encodes interleukin-4, a central type 2 cytokine that promotes Th2 differentiation, B-cell class switching, IgE biology, and allergic inflammation.",
        "variant_effect_overview": [
            "IL4 promoter and regulatory markers are studied as modifiers of IL-4 expression, IgE, asthma, and atopy.",
            "The IL4/IL13/RAD50 5q31 region should be interpreted with LD and cytokine-cluster context.",
        ],
        "conditions": ["Asthma and allergic sensitization", "IgE and Th2 differentiation", "5q31 cytokine-cluster research"],
        "variants": [
            {"variant": "rs2243250", "common_name": "IL4 promoter atopy/asthma marker"},
            {"variant": "IL4/IL13 cytokine-cluster model", "common_name": "5q31 IL4-IL13 regulatory haplotype model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "FLG",
        "ensembl_id": "ENSG00000143631",
        "ncbi_id": "2312",
        "uniprot_id": "P20930",
        "cytoband": "1q21.3",
        "chromosome": "1",
        "start": 152274651,
        "end": 152297679,
        "strand": "-",
        "gene_summary": "FLG encodes filaggrin, a key epidermal barrier protein whose loss-of-function variants are strongly linked to ichthyosis vulgaris, atopic dermatitis, allergic sensitization, and asthma risk in barrier-impaired contexts.",
        "variant_effect_overview": [
            "FLG loss-of-function alleles are higher-impact barrier-disruption markers than most common allergic-disease GWAS SNPs.",
            "Asthma interpretation is usually strongest when atopic dermatitis, eczema, or barrier dysfunction is also present.",
        ],
        "conditions": ["Atopic dermatitis and ichthyosis vulgaris", "Epidermal barrier dysfunction", "Allergic sensitization and atopic asthma context"],
        "variants": [
            {"variant": "rs61816761", "display_name": "rs61816761 (FLG R501X)", "common_name": "FLG loss-of-function barrier marker", "clinical_significance": "Clinically important FLG loss-of-function research and barrier-risk marker; classification still depends on zygosity and external clinical review."},
            {"variant": "FLG 2282del4", "common_name": "FLG recurrent loss-of-function deletion marker", "clinical_significance": "Clinically important FLG loss-of-function research and barrier-risk marker; classification still depends on zygosity and external clinical review."},
        ],
    },
    {
        "gene_name": "TLR10",
        "ensembl_id": "ENSG00000174123",
        "ncbi_id": "81793",
        "uniprot_id": "Q9BXR5",
        "cytoband": "4p14",
        "chromosome": "4",
        "start": 38773860,
        "end": 38784611,
        "strand": "-",
        "gene_summary": "TLR10 encodes toll-like receptor 10, an innate immune receptor in the TLR1/6/10 cluster with immunomodulatory roles in microbial sensing, inflammation, and allergy/asthma association research.",
        "variant_effect_overview": [
            "TLR10 markers can tag innate immune signaling and the 4p14 TLR cluster rather than a single isolated receptor effect.",
            "Interpretation should account for infection exposure, ancestry, immune-cell state, and linked TLR1/TLR6 variation.",
        ],
        "conditions": ["Asthma and allergic inflammation", "Innate immune signaling and microbial exposure research", "TLR1/TLR6/TLR10 cluster context"],
        "variants": [
            {"variant": "rs4129009", "common_name": "TLR10 innate-immune/allergy association marker"},
            {"variant": "TLR1/TLR6/TLR10 cluster model", "common_name": "4p14 toll-like receptor haplotype model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "TNFRSF8",
        "ensembl_id": "ENSG00000120949",
        "ncbi_id": "943",
        "uniprot_id": "P28908",
        "cytoband": "1p36.22",
        "chromosome": "1",
        "start": 12123434,
        "end": 12204264,
        "strand": "+",
        "gene_summary": "TNFRSF8 encodes CD30, a tumor necrosis factor receptor superfamily member expressed on activated lymphocytes and central to CD30-positive lymphoproliferative biology and immune activation research.",
        "variant_effect_overview": [
            "TNFRSF8 interpretation is usually expression-, cell-state-, and disease-context driven rather than centered on common diagnostic SNPs.",
            "Sequence or methylation signals should be integrated with CD30 expression, lymphocyte activation, and inflammatory or lymphoma phenotype context.",
        ],
        "conditions": ["Activated T-cell and B-cell immune biology", "CD30-positive lymphoma and lymphoproliferative disease research", "Allergic and inflammatory immune activation context"],
        "variants": [
            {"variant": "TNFRSF8/CD30 expression model", "common_name": "CD30 immune-activation expression model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "CD30",
        "canonical_symbol": "TNFRSF8",
        "ensembl_id": "ENSG00000120949",
        "ncbi_id": "943",
        "uniprot_id": "P28908",
        "cytoband": "1p36.22",
        "chromosome": "1",
        "start": 12123434,
        "end": 12204264,
        "strand": "+",
        "gene_summary": "CD30 is the common immunology name for TNFRSF8, a tumor necrosis factor receptor superfamily member expressed on activated lymphocytes and used as a marker in CD30-positive lymphoproliferative disease research.",
        "variant_effect_overview": [
            "CD30/TNFRSF8 interpretation is primarily expression-, activation-state-, and disease-context based rather than a common-SNP diagnostic model.",
            "The CD30 alias bundle lets users request the familiar CD30 name while preserving TNFRSF8 coordinates and evidence.",
        ],
        "conditions": ["Activated lymphocyte and CD30 biology", "CD30-positive lymphoma and inflammatory immune activation research", "Allergic and immune-cell activation context"],
        "variants": [
            {"variant": "CD30/TNFRSF8 expression model", "common_name": "CD30 immune-activation alias model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "MUC5AC",
        "ensembl_id": "ENSG00000215182",
        "ncbi_id": "4586",
        "uniprot_id": "P98088",
        "cytoband": "11p15.5",
        "chromosome": "11",
        "start": 1151580,
        "end": 1222364,
        "strand": "+",
        "gene_summary": "MUC5AC encodes a gel-forming secreted mucin that is central to airway mucus, goblet-cell metaplasia, mucociliary biology, asthma, chronic airway disease, and epithelial inflammation research.",
        "variant_effect_overview": [
            "MUC5AC findings are often regulatory, expression, or mucin-locus context rather than classic high-penetrance coding-variant calls.",
            "Methylation and expression context can be especially important because airway mucus phenotypes are tissue- and exposure-dependent.",
        ],
        "conditions": ["Asthma mucus hypersecretion", "Airway epithelial goblet-cell metaplasia", "Chronic airway inflammation and mucin biology"],
        "variants": [
            {"variant": "MUC5AC regulatory-expression model", "common_name": "MUC5AC airway mucus expression model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "SMAD3",
        "ensembl_id": "ENSG00000166949",
        "ncbi_id": "4088",
        "uniprot_id": "P84022",
        "cytoband": "15q22.33",
        "chromosome": "15",
        "start": 67356101,
        "end": 67487533,
        "strand": "+",
        "gene_summary": "SMAD3 encodes a TGF-beta pathway transcriptional mediator involved in epithelial remodeling, fibrosis, immune regulation, vascular biology, and asthma or inflammatory-disease association studies.",
        "variant_effect_overview": [
            "SMAD3 variants can tag TGF-beta signaling and airway remodeling context in asthma and inflammatory phenotypes.",
            "Rare high-impact SMAD3 variants have separate clinical genetics implications; this bundle keeps common markers as research-level association context.",
        ],
        "conditions": ["Asthma and airway remodeling", "TGF-beta signaling and fibrosis biology", "Inflammatory and connective-tissue disease context"],
        "variants": [
            {"variant": "rs744910", "position": 67446785, "common_name": "SMAD3 asthma and airway-remodeling association marker"},
            {"variant": "rs17293632", "common_name": "SMAD3 immune/inflammatory disease regulatory marker"},
        ],
    },
    {
        "gene_name": "IL18R1",
        "ensembl_id": "ENSG00000115604",
        "ncbi_id": "8809",
        "uniprot_id": "Q13478",
        "cytoband": "2q12.1",
        "chromosome": "2",
        "start": 102927989,
        "end": 103015218,
        "strand": "+",
        "gene_summary": "IL18R1 encodes the IL-18 receptor alpha chain, a cytokine receptor in the 2q12 IL1 receptor cluster that supports IL-18-driven Th1, NK-cell, inflammatory, and asthma/eosinophil research biology.",
        "variant_effect_overview": [
            "IL18R1 markers often sit in LD with IL1RL1 and IL18RAP cluster signals and should be interpreted at the 2q12 receptor-cluster level when needed.",
            "Variant signals may relate to asthma, eosinophils, immune activation, or inflammatory disease, but causal gene assignment is often unresolved.",
        ],
        "conditions": ["Asthma and eosinophil trait research", "IL-18 receptor signaling and Th1/NK-cell inflammation", "2q12 IL1 receptor-cluster association context"],
        "variants": [
            {"variant": "rs3771166", "position": 102986222, "common_name": "IL18R1 asthma/eosinophil receptor-cluster marker"},
            {"variant": "rs13408661", "position": 102955082, "common_name": "IL1RL1/IL18R1 receptor-cluster asthma marker"},
        ],
    },
    {
        "gene_name": "IL18RAP",
        "ensembl_id": "ENSG00000115607",
        "ncbi_id": "8807",
        "uniprot_id": "O95256",
        "cytoband": "2q12.1",
        "chromosome": "2",
        "start": 103035149,
        "end": 103069025,
        "strand": "+",
        "gene_summary": "IL18RAP encodes the IL-18 receptor accessory protein, the co-receptor that helps IL18R1 transmit IL-18 signaling in inflammatory, Th1, NK-cell, autoimmune, and airway immune research.",
        "variant_effect_overview": [
            "IL18RAP markers can tag IL-18 receptor complex function and the broader 2q12 IL1 receptor gene cluster.",
            "Interpretation should account for IL18R1 and IL1RL1 LD, inflammatory phenotype, and tissue-specific expression.",
        ],
        "conditions": ["IL-18 receptor signaling and inflammatory disease", "Asthma and immune activation research", "2q12 IL1 receptor-cluster association context"],
        "variants": [
            {"variant": "rs917997", "common_name": "IL18RAP inflammatory-disease receptor-cluster marker"},
            {"variant": "IL18R1/IL18RAP receptor-complex model", "common_name": "IL-18 receptor complex model", "is_assayable_in_snp_vcf": False},
        ],
    },
]


BIORENDER_VISUAL_LIBRARY: dict[str, dict[str, Any]] = {
    "dna_repair": {
        "template_title": "DNA Repair Mechanisms",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-5ee11f56287bce00b20b3f92?source=mcp",
        "icon_search_terms": ["DNA repair", "genome editing"],
        "recommended_icons": ["BRCA1 (schematic)", "BRCA2 (schematic)", "Cytosine base editor"],
    },
    "receptor_signaling": {
        "template_title": "Intracellular Layout - Receptor Signaling from Plasma Membrane to Nucleus (2)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-61748edc02c76300a75f7771?source=mcp",
        "icon_search_terms": ["growth factor receptor", "EGFR"],
        "recommended_icons": ["EGFR in endosome (ubiquitinated)", "EGF (pathway)", "VEGFR2 (dimer, schematic)"],
    },
    "lipid_metabolism": {
        "template_title": "Intracellular Metabolic Pathway Comparison (Layout)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-61f9dec1241da3009dcc4c2c?source=mcp",
        "icon_search_terms": ["lipid metabolism", "APOE", "LDL receptor"],
        "recommended_icons": ["Apolipoprotein A brush (kringle)", "Vessel (atherosclerosis, cross-section)", "Atherosclerosis (with hypoxia, cross-section)"],
    },
    "mitochondria": {
        "template_title": "Mitochondrial Damage Pathways (Layout)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-63dd8cb60cba5b3e6469eb30?source=mcp",
        "icon_search_terms": ["mitochondria", "aldehyde dehydrogenase"],
        "recommended_icons": ["Mitochondria (elongated)", "Mitochondria (2D, with ROS symbol)", "Metabolic pathway (in mitochondria)"],
    },
    "ion_channel": {
        "template_title": "Types of Cellular Receptors",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-60ba45fff7aa2e00a8b7081c?source=mcp",
        "icon_search_terms": ["ion channel", "calcium channel", "CFTR"],
        "recommended_icons": ["Ion channel (ligand-gated, open, schematic)", "ATP sensitive potassium channel (open, in membrane)", "Cardiac action potential diagram"],
    },
    "hematology": {
        "template_title": "Pathway and Genetic Markers (Layout)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-62e6fb114bd49370e5f33be2?source=mcp",
        "icon_search_terms": ["hemoglobin", "iron metabolism"],
        "recommended_icons": ["Vessel cross-section with red  blood cells (anemic)", "Blood vessel (with red  blood cells, anemic)", "Osmoregulation (red blood cell)"],
    },
    "synapse": {
        "template_title": "CREB Signaling Pathway",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-608c41c359b80e00a38478b4?source=mcp",
        "icon_search_terms": ["synapse", "neurotrophin"],
        "recommended_icons": ["Synaptic vesicle with neurotransmitters (medium)", "Pre-synaptic membrane (with cycling vesicles)", "Post-synaptic membrane (3D, dendrite)"],
    },
    "metabolic_pathway": {
        "template_title": "Intracellular Metabolic Pathway Comparison (Layout)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-61f9dec1241da3009dcc4c2c?source=mcp",
        "icon_search_terms": ["folate metabolism", "protein kinase", "metabolic pathway"],
        "recommended_icons": ["Metabolic pathway (in mitochondria)", "Simulated metabolic activity", "PKA bound to cAMP"],
    },
    "cardiac_contractile": {
        "template_title": "Pathway and Genetic Markers (Layout)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-62e6fb114bd49370e5f33be2?source=mcp",
        "icon_search_terms": ["myosin sarcomere", "cardiomyocyte"],
        "recommended_icons": ["Cardiomyocyte (branched, hypertrophy)", "Cardiomyocyte 2", "Cardiomyocyte (branched)"],
    },
    "allergic_inflammation": {
        "template_title": "Mechanism of Allergic Inflammation Induction",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-63403fd8cda8d1d02e0a3126?source=mcp",
        "icon_search_terms": ["mast cell allergy IgE cytokine", "allergic inflammation", "eosinophil cytokine"],
        "recommended_icons": ["Antibody IgE (round style, with domains)", "IL-5 receptor-ligand pair (cartoon)", "Cytokine (2D)"],
    },
    "pigmentation_hair": {
        "template_title": "Hair Follicle Anatomy",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-641dc82b59511ede76dc7b32?source=mcp",
        "icon_search_terms": ["melanocyte", "melanin", "hair follicle", "skin epidermis hair follicle"],
        "recommended_icons": ["Skin (epidermis)", "Epithelial layers (skin)", "Male balding (stage 0, lateral)"],
    },
    "aging_telomere": {
        "template_title": "Hallmarks of Aging (Circle Layout)",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-6629073064f3f05486a9a0ab?source=mcp",
        "icon_search_terms": ["telomere", "cellular senescence", "aging"],
        "recommended_icons": ["Chromosome (with telomeric DNA sequence)", "Human telomerase (RNA)", "p53 (pathway)"],
    },
    "neurodegeneration": {
        "template_title": "Microglial Activation and Neurodegeneration Linked to Aging and Inflammation",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-6995911c0580ae042d7236bc?source=mcp",
        "icon_search_terms": ["neuron synapse neurodegeneration", "synapse", "microglia"],
        "recommended_icons": ["Microglial cell (with Abeta plaque)", "Axonal-dendritic synaptic cleft (3D)", "Myelin sheath (long, degenerating)"],
    },
    "gut_barrier": {
        "template_title": "Intestinal Barrier Structure with Microbiota",
        "template_url": "https://app.biorender.com/biorender-templates/details/t-69eded0a2c5ddd9536c15eda?source=mcp",
        "icon_search_terms": ["intestinal barrier gut epithelium", "digestion", "microbiome"],
        "recommended_icons": ["Intestinal epithelium (Peyer's patch)", "Colon epithelium", "Colon epithelium (mucous)"],
    },
}


def _build_biorender_visuals(gene_name: str, visual_key: str, figure_focus: str) -> dict[str, Any]:
    visual = BIORENDER_VISUAL_LIBRARY[visual_key]
    return {
        "provider": "BioRender",
        "label": f"{gene_name} biochemical figure starter",
        "focus": figure_focus,
        "template_title": visual["template_title"],
        "template_url": visual["template_url"],
        "icon_search_terms": visual["icon_search_terms"],
        "recommended_icons": visual["recommended_icons"],
        "usage_note": (
            "BioRender template and icon-library search results are bundled as figure starters. "
            "Open the template link in BioRender, then add or replace the listed icons to make a gene-specific biochemical pathway image."
        ),
    }


def _build_biochemistry_evidence(config: dict[str, Any]) -> list[dict[str, str]]:
    gene_name = str(config["gene_name"])
    evidence = [
        _evidence(f"NCBI Gene {config['ncbi_id']}: {gene_name}", f"https://www.ncbi.nlm.nih.gov/gene/{config['ncbi_id']}"),
        _evidence(f"UniProt {config['uniprot_id']}: {gene_name}_HUMAN", f"https://www.uniprot.org/uniprotkb/{config['uniprot_id']}/entry"),
        _evidence(
            f"Ensembl GRCh37 {config['ensembl_id']}: {gene_name} coordinate source",
            f"https://grch37.ensembl.org/Homo_sapiens/Gene/Summary?g={config['ensembl_id']}",
        ),
    ]
    evidence.extend(config.get("evidence", []))
    return evidence


def _build_biochemistry_variant(
    config: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    gene_name = str(config["gene_name"])
    axis = str(config["biochemical_axis"])
    conditions = list(config["conditions"])
    variant_name = str(variant["variant"])
    display_name = str(variant.get("display_name") or variant_name)
    common_name = str(variant.get("common_name") or f"{gene_name} biochemical marker")
    assayable = bool(variant.get("is_assayable_in_snp_vcf", True))
    marker_context = str(variant.get("marker_context") or ", ".join(conditions[:2]))
    default_evidence = variant.get("evidence", []) or config.get("evidence", [])
    record: dict[str, Any] = {
        "variant": variant_name,
        "display_name": display_name,
        "common_name": common_name,
        "lookup_keys": variant.get("lookup_keys") or [
            variant_name,
            display_name,
            f"{gene_name}:{variant_name}",
            f"{gene_name.lower()}:{variant_name.lower()}",
        ],
        "region_class": str(variant.get("region_class") or "gene_body"),
        "interpretation_scope": str(
            variant.get("interpretation_scope")
            or "Biochemistry research marker / clinical-context marker"
        ),
        "clinical_interpretation": str(
            variant.get("clinical_interpretation")
            or (
                f"{display_name} is curated as a {gene_name} marker for {marker_context}. "
                f"Read it through genotype dosage, phenotype context, and the {axis} pathway rather than as a stand-alone deterministic result."
            )
        ),
        "clinical_significance": str(
            variant.get("clinical_significance")
            or "Curated research and clinical-context marker; external clinical classification is required for diagnostic use."
        ),
        "functional_effects": variant.get("functional_effects")
        or [
            f"Can modify or tag {axis}.",
            "Best interpreted with tissue, ancestry, phenotype, and assay context.",
        ],
        "associated_conditions": variant.get("associated_conditions") or conditions,
        "research_context": variant.get("research_context")
        or [
            f"Use this marker as a {gene_name} pathway-context signal, not as a stand-alone diagnosis.",
            "Sequence, methylation, expression, and protein data may point in different directions and should be reconciled.",
        ],
        "usual_variant_note": str(variant.get("usual_variant_note") or common_name),
        "methylation_interpretation": str(
            variant.get("methylation_interpretation")
            or (
                f"Pair {display_name} with {gene_name} methylation only as local regulatory context; "
                "methylation does not prove the variant's biochemical effect."
            )
        ),
        "is_assayable_in_snp_vcf": assayable,
        "evidence": default_evidence,
        "literature_findings": variant.get("literature_findings")
        or [
            {
                "paper": str(variant.get("evidence_label") or f"{gene_name} curated evidence set"),
                "genotypes": f"{display_name} or linked {gene_name} locus genotype states",
                "phenotype": marker_context,
                "finding": str(
                    variant.get("finding")
                    or (
                        f"The local curation treats {display_name} as a gene-specific marker for {marker_context} and {axis}."
                    )
                ),
                "url": str(variant.get("evidence_url") or (default_evidence[0]["url"] if default_evidence else "")),
            }
        ],
    }
    if "position" in variant:
        record["position"] = int(variant["position"])
    if not assayable:
        record["clinical_parameter_summary"] = (
            "This record describes a recurrent model, haplotype, fusion, or loss-of-function class rather than a simple SNP row."
        )
    return record


def _build_biochemistry_gene_definition(config: dict[str, Any]) -> dict[str, Any]:
    gene_name = str(config["gene_name"])
    axis = str(config["biochemical_axis"])
    conditions = list(config["conditions"])
    evidence = _build_biochemistry_evidence(config)
    variants = [_build_biochemistry_variant(config, variant) for variant in config["variants"]]
    definition = {
        "gene_name": gene_name,
        "ensembl_id": config["ensembl_id"],
        "ncbi_id": config["ncbi_id"],
        "uniprot_id": config["uniprot_id"],
        "cytoband": config["cytoband"],
        "chromosome": config["chromosome"],
        "start": int(config["start"]),
        "end": int(config["end"]),
        "strand": config["strand"],
        "coordinate_source": "Ensembl GRCh37 lookup, aligned to the hg19 coordinate system used by this app",
        "gene_summary": config["gene_summary"],
        "clinical_context": config["clinical_context"],
        "variant_effect_overview": config["variant_effect_overview"],
        "condition_research_overview": conditions,
        "methylation_interpretation": (
            f"{gene_name} methylation should be treated as local regulatory context around a {axis} gene. "
            "Promoter-proximal methylation can support expression-potential review, but it should not replace genotype, RNA, protein, or phenotype data."
        ),
        "methylation_effects": [
            f"Promoter or first-exon methylation may suggest a more restrained or permissive local expression state for {gene_name}.",
            f"Because {gene_name} biology is tissue and pathway dependent, methylation values are context rather than a stand-alone biomarker.",
            f"Combined sequence plus methylation findings are most useful when interpreted against {axis}.",
        ],
        "methylation_condition_research": [
            f"Epigenetic regulation studies that use {gene_name} promoter or gene-body methylation as expression-context evidence.",
            f"Variant-plus-methylation review for {axis}.",
            *conditions[:2],
        ],
        "evidence": evidence,
        "variants": variants,
        "population_intro": f"Broader population patterns curated from {gene_name} {axis} and condition-association literature.",
        "population_coverage_note": (
            f"The bundled {gene_name} population database stores gene-level and marker-level interpretation notes, "
            "but does not embed a full ancestry-frequency matrix for these markers yet."
        ),
        "population_sources": evidence,
        "gene_population_patterns": [
            {
                "variant": f"{gene_name} biochemical pathway context",
                "location_group": "Global pattern",
                "summary": (
                    f"{gene_name} findings should be read against {axis}; population frequency, ascertainment, and phenotype definition can change the apparent effect size."
                ),
            },
            *[
                {
                    "variant": record["display_name"],
                    "location_group": "Marker and cohort pattern",
                    "summary": (
                        f"{record['display_name']} is curated as a {gene_name} marker for {', '.join(record['associated_conditions'][:2])}; "
                        "interpret ancestry and cohort patterns before assigning biological direction."
                    ),
                }
                for record in variants[:3]
            ],
        ],
        "biorender_visuals": _build_biorender_visuals(
            gene_name,
            str(config["visual_key"]),
            str(config.get("figure_focus") or axis),
        ),
        "concrete_variant_prediction": config["concrete_variant_prediction"],
    }
    for optional_key in ("manifest_filter_region", "allow_empty_manifest_subset", "skip_manifest_subset"):
        if optional_key in config:
            definition[optional_key] = config[optional_key]
    return definition


BIOCHEMISTRY_GENE_CONFIGS: list[dict[str, Any]] = [
    {
        "gene_name": "TP53",
        "ensembl_id": "ENSG00000141510",
        "ncbi_id": "7157",
        "uniprot_id": "P04637",
        "cytoband": "17p13.1",
        "chromosome": "17",
        "start": 7565097,
        "end": 7590856,
        "strand": "-",
        "biochemical_axis": "DNA damage checkpoint, p53 transcriptional control, apoptosis, cell-cycle arrest, and senescence",
        "gene_summary": "TP53 encodes the p53 tumor suppressor, a stress-responsive transcription factor that integrates DNA damage, oncogene activation, hypoxia, ribosomal stress, and metabolic stress to control cell-cycle arrest, apoptosis, DNA repair, and senescence programs.",
        "clinical_context": "The local TP53 bundle is tumor-suppressor and DNA-damage-response oriented. Germline pathogenic variants require clinical review for Li-Fraumeni syndrome context, while common and somatic markers should stay separated from germline risk interpretation.",
        "variant_effect_overview": [
            "TP53 interpretation depends heavily on variant class, zygosity, tumor versus germline origin, and whether the event affects the DNA-binding domain or regulatory regions.",
            "Common polymorphisms such as Arg72Pro are research modifiers; truncating, missense hotspot, or deletion events can have much higher clinical impact when externally classified.",
        ],
        "conditions": ["Cancer susceptibility and tumor-suppressor biology", "DNA damage response and apoptosis", "Li-Fraumeni syndrome and somatic oncology context"],
        "visual_key": "dna_repair",
        "evidence": [_evidence("ClinVar TP53 variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=TP53%5Bgene%5D")],
        "variants": [
            {"variant": "rs1042522", "display_name": "rs1042522 (TP53 Arg72Pro)", "common_name": "TP53 p.Arg72Pro apoptosis/cancer association marker"},
            {"variant": "TP53 loss-of-function model", "common_name": "TP53 high-impact tumor-suppressor loss model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a TP53 stress-response thesis: the individual may carry a marker relevant to DNA-damage checkpoint tone, apoptosis, cell-cycle arrest, senescence, cancer cohort biology, or high-impact tumor-suppressor review depending on the exact allele class. Do not merge benign/common TP53 markers with pathogenic germline or somatic hotspot interpretation without external classification.",
    },
    {
        "gene_name": "BRCA1",
        "ensembl_id": "ENSG00000012048",
        "ncbi_id": "672",
        "uniprot_id": "P38398",
        "cytoband": "17q21.31",
        "chromosome": "17",
        "start": 41196312,
        "end": 41277500,
        "strand": "-",
        "biochemical_axis": "homologous recombination repair, DNA double-strand break response, replication-fork protection, and genome stability",
        "gene_summary": "BRCA1 encodes a tumor-suppressor scaffold for homologous recombination, DNA damage signaling, checkpoint control, replication-fork protection, and chromatin-linked repair.",
        "clinical_context": "The local BRCA1 bundle is DNA-repair and hereditary cancer oriented. Pathogenic BRCA1 alleles are clinically important, but app-level matches remain research/triage context until confirmed by clinical-grade variant interpretation.",
        "variant_effect_overview": [
            "BRCA1 truncating, splice, and damaging missense variants can compromise homologous recombination and hereditary breast/ovarian cancer biology.",
            "Founder variants and VUS records require zygosity, transcript, classification, ancestry, and family-history context before interpretation.",
        ],
        "conditions": ["Hereditary breast and ovarian cancer context", "Homologous recombination repair deficiency", "DNA double-strand break response"],
        "visual_key": "dna_repair",
        "evidence": [_evidence("ClinVar BRCA1 variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=BRCA1%5Bgene%5D")],
        "variants": [
            {"variant": "BRCA1 c.68_69delAG", "display_name": "BRCA1 c.68_69delAG / 185delAG", "common_name": "BRCA1 frameshift founder marker", "is_assayable_in_snp_vcf": False},
            {"variant": "BRCA1 c.5266dupC", "display_name": "BRCA1 c.5266dupC / 5382insC", "common_name": "BRCA1 frameshift founder marker", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a BRCA1 homologous-recombination thesis: the individual may carry a signal relevant to DNA double-strand break repair, replication-fork protection, hereditary breast/ovarian cancer review, or HRD pathway biology. Only externally classified pathogenic or likely pathogenic BRCA1 variants should be escalated toward clinical hereditary-cancer risk interpretation.",
    },
    {
        "gene_name": "EGFR",
        "ensembl_id": "ENSG00000146648",
        "ncbi_id": "1956",
        "uniprot_id": "P00533",
        "cytoband": "7p11.2",
        "chromosome": "7",
        "start": 55086714,
        "end": 55324313,
        "strand": "+",
        "biochemical_axis": "receptor tyrosine kinase signaling through RAS-MAPK, PI3K-AKT, PLC-gamma, endocytosis, and growth-control pathways",
        "gene_summary": "EGFR encodes the epidermal growth factor receptor, a receptor tyrosine kinase that links ligand binding and receptor dimerization to phosphorylation cascades controlling proliferation, survival, differentiation, and receptor trafficking.",
        "clinical_context": "The local EGFR bundle is signaling and somatic-oncology oriented. Canonical activating and resistance mutations are therapy-relevant in tumor testing, while germline VCF context should be interpreted cautiously.",
        "variant_effect_overview": [
            "EGFR exon 19 deletions and L858R-like mutations are activating somatic markers in selected cancers, especially lung adenocarcinoma.",
            "T790M-like resistance markers require tumor context and treatment history; common germline SNPs are not equivalent to somatic driver calls.",
        ],
        "conditions": ["Receptor tyrosine kinase signaling", "Non-small cell lung cancer somatic driver context", "EGFR inhibitor sensitivity and resistance research"],
        "visual_key": "receptor_signaling",
        "evidence": [_evidence("ClinVar EGFR variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=EGFR%5Bgene%5D")],
        "variants": [
            {"variant": "EGFR p.Leu858Arg", "display_name": "EGFR L858R", "common_name": "EGFR activating kinase-domain marker"},
            {"variant": "EGFR exon 19 deletion model", "common_name": "EGFR activating exon 19 deletion class", "is_assayable_in_snp_vcf": False},
            {"variant": "EGFR p.Thr790Met", "display_name": "EGFR T790M", "common_name": "EGFR resistance-gatekeeper marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an EGFR receptor-tyrosine-kinase thesis: the sample may carry a signal relevant to growth-factor signaling, receptor phosphorylation, endosomal trafficking, or somatic oncology driver/resistance review. Therapy-facing EGFR interpretation requires tumor context, assay validation, and exact variant classification.",
    },
    {
        "gene_name": "APOE",
        "ensembl_id": "ENSG00000130203",
        "ncbi_id": "348",
        "uniprot_id": "P02649",
        "cytoband": "19q13.32",
        "chromosome": "19",
        "start": 45409011,
        "end": 45412650,
        "strand": "+",
        "biochemical_axis": "lipoprotein transport, cholesterol handling, remnant clearance, neuroinflammation, and amyloid/tau-associated brain lipid biology",
        "gene_summary": "APOE encodes apolipoprotein E, a lipid-transport protein central to cholesterol and triglyceride-rich lipoprotein handling and to brain lipid, amyloid, tau, and neuroinflammatory research.",
        "clinical_context": "The local APOE bundle is lipid and neurogenetics oriented. APOE epsilon genotypes are risk modifiers rather than deterministic Alzheimer disease calls, and lipid/cardiovascular interpretation depends on full haplotype and phenotype context.",
        "variant_effect_overview": [
            "The APOE epsilon2/epsilon3/epsilon4 isoforms are defined mainly by rs429358 and rs7412 combinations, so single-site interpretation is incomplete.",
            "APOE variants modify cohort-level risk for Alzheimer disease, lipid traits, cardiovascular biology, and type III hyperlipoproteinemia context.",
        ],
        "conditions": ["Alzheimer disease risk-modifier research", "Lipid transport and cardiovascular traits", "ApoE isoform and neuroinflammation biology"],
        "visual_key": "lipid_metabolism",
        "evidence": [_evidence("Alzforum APOE gene overview", "https://www.alzforum.org/mutations/apoe")],
        "variants": [
            {"variant": "rs429358", "common_name": "APOE epsilon4-defining Cys130Arg marker"},
            {"variant": "rs7412", "common_name": "APOE epsilon2-defining Arg176Cys marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an APOE lipid-transport and neurogenetics thesis: the individual may carry partial or complete epsilon haplotype context relevant to lipoprotein clearance, Alzheimer disease cohort risk, neuroinflammation, and cardiovascular lipid biology. Full APOE epsilon interpretation requires both rs429358 and rs7412 genotype dosage.",
    },
    {
        "gene_name": "ACE2",
        "ensembl_id": "ENSG00000130234",
        "ncbi_id": "59272",
        "uniprot_id": "Q9BYF1",
        "cytoband": "Xp22.2",
        "chromosome": "X",
        "start": 15579156,
        "end": 15620271,
        "strand": "-",
        "biochemical_axis": "renin-angiotensin peptide processing, carboxypeptidase activity, cardiopulmonary vascular biology, and viral spike receptor biology",
        "gene_summary": "ACE2 encodes angiotensin-converting enzyme 2, a membrane carboxypeptidase that converts angiotensin peptides and also serves as an entry receptor for SARS-CoV and SARS-CoV-2 spike proteins.",
        "clinical_context": "The local ACE2 bundle is cardiopulmonary, vascular, and receptor-biology oriented. ACE2 variants are mostly research-context modifiers unless externally classified and phenotypically anchored.",
        "variant_effect_overview": [
            "ACE2 regulatory and coding variants can be studied for expression, peptide balance, cardiovascular traits, or host-pathogen interaction context.",
            "X-linked location, sex, tissue expression, ancestry, and infection/exposure context all matter for interpretation.",
        ],
        "conditions": ["Renin-angiotensin system biology", "Cardiopulmonary and vascular research", "SARS-CoV/SARS-CoV-2 receptor interaction context"],
        "visual_key": "receptor_signaling",
        "evidence": [_evidence("UniProt ACE2 coronavirus receptor annotation", "https://www.uniprot.org/uniprotkb/Q9BYF1/entry")],
        "variants": [
            {"variant": "rs2285666", "common_name": "ACE2 regulatory/splice-region expression marker"},
            {"variant": "ACE2 spike-binding model", "common_name": "ACE2 receptor-binding functional model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an ACE2 receptor-peptidase thesis: the individual may carry research context relevant to angiotensin peptide processing, vascular or cardiopulmonary biology, ACE2 expression, or viral spike-receptor interaction studies. This is not a deterministic infection-risk or cardiovascular-risk call.",
    },
    {
        "gene_name": "CFTR",
        "ensembl_id": "ENSG00000001626",
        "ncbi_id": "1080",
        "uniprot_id": "P13569",
        "cytoband": "7q31.2",
        "chromosome": "7",
        "start": 117105838,
        "end": 117356025,
        "strand": "+",
        "biochemical_axis": "ATP-regulated chloride and bicarbonate channel function, epithelial fluid balance, mucus hydration, and ABC transporter folding/trafficking",
        "gene_summary": "CFTR encodes an ATP-binding cassette chloride and bicarbonate channel that regulates epithelial salt and water transport in lung, pancreas, intestine, sweat gland, and reproductive tissues.",
        "clinical_context": "The local CFTR bundle is cystic-fibrosis and epithelial-ion-transport oriented. CFTR interpretation requires exact variant class, phase, zygosity, phenotype, and external clinical classification.",
        "variant_effect_overview": [
            "CFTR pathogenic variants can affect channel synthesis, folding, gating, conductance, splicing, or protein stability.",
            "Carrier, compound-heterozygous, and modifier contexts should be kept separate from single-site exploratory findings.",
        ],
        "conditions": ["Cystic fibrosis and CFTR-related disorder context", "Epithelial chloride/bicarbonate transport", "Airway mucus hydration and pancreatic/intestinal biology"],
        "visual_key": "ion_channel",
        "evidence": [_evidence("CFTR2 variant database", "https://cftr2.org/")],
        "variants": [
            {"variant": "rs113993960", "display_name": "CFTR p.Phe508del / F508del", "common_name": "CFTR class II folding/trafficking marker"},
            {"variant": "CFTR gating-variant model", "common_name": "CFTR channel gating and conductance model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a CFTR epithelial-ion-channel thesis: the individual may carry a marker relevant to chloride/bicarbonate transport, channel folding or gating, mucus hydration, cystic-fibrosis biology, or CFTR-related disorder review. Clinical interpretation requires phase, zygosity, exact allele classification, and phenotype match.",
    },
    {
        "gene_name": "PCSK9",
        "ensembl_id": "ENSG00000169174",
        "ncbi_id": "255738",
        "uniprot_id": "Q8NBP7",
        "cytoband": "1p32.3",
        "chromosome": "1",
        "start": 55505221,
        "end": 55530525,
        "strand": "+",
        "biochemical_axis": "LDL receptor trafficking, hepatic cholesterol homeostasis, secreted protease biology, and LDL-C regulation",
        "gene_summary": "PCSK9 encodes a secreted proprotein convertase that binds LDL receptor and promotes receptor degradation, thereby tuning hepatic LDL uptake and circulating LDL cholesterol.",
        "clinical_context": "The local PCSK9 bundle is lipid and pharmacogenetic-mechanism oriented. Loss-of-function and gain-of-function alleles can have different LDL-C and cardiovascular-risk implications.",
        "variant_effect_overview": [
            "PCSK9 loss-of-function alleles tend to lower LDL-C through increased LDL receptor availability.",
            "Rare gain-of-function variants can support familial hypercholesterolemia context when externally classified.",
        ],
        "conditions": ["LDL cholesterol and cardiovascular risk biology", "Familial hypercholesterolemia review context", "PCSK9 inhibitor target biology"],
        "visual_key": "lipid_metabolism",
        "evidence": [_evidence("ClinVar PCSK9 variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=PCSK9%5Bgene%5D")],
        "variants": [
            {"variant": "rs11591147", "display_name": "rs11591147 (PCSK9 R46L)", "common_name": "PCSK9 loss-of-function LDL-C marker"},
            {"variant": "PCSK9 gain-of-function model", "common_name": "PCSK9 familial hypercholesterolemia gain-of-function model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a PCSK9 LDL-receptor-trafficking thesis: the individual may carry context relevant to LDL-C set point, hepatic LDL receptor recycling/degradation, cardiovascular lipid traits, familial hypercholesterolemia review, or PCSK9 inhibitor target biology.",
    },
    {
        "gene_name": "LDLR",
        "ensembl_id": "ENSG00000130164",
        "ncbi_id": "3949",
        "uniprot_id": "P01130",
        "cytoband": "19p13.2",
        "chromosome": "19",
        "start": 11200038,
        "end": 11244492,
        "strand": "+",
        "biochemical_axis": "LDL particle binding, receptor-mediated endocytosis, cholesterol uptake, and familial hypercholesterolemia biology",
        "gene_summary": "LDLR encodes the low-density lipoprotein receptor, which binds apoB/apoE-containing lipoproteins and mediates LDL uptake through clathrin-associated endocytosis.",
        "clinical_context": "The local LDLR bundle is familial-hypercholesterolemia and lipid-uptake oriented. High-impact LDLR findings require clinical-grade classification, inheritance, phenotype, and lipid data.",
        "variant_effect_overview": [
            "LDLR variants can affect ligand binding, trafficking, recycling, splicing, or receptor abundance.",
            "Common markers are risk modifiers, while pathogenic LDLR variants can have monogenic familial hypercholesterolemia implications.",
        ],
        "conditions": ["Familial hypercholesterolemia review context", "LDL-C and apoB lipoprotein uptake", "Atherosclerotic cardiovascular disease lipid biology"],
        "visual_key": "lipid_metabolism",
        "evidence": [_evidence("ClinVar LDLR variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=LDLR%5Bgene%5D")],
        "variants": [
            {"variant": "rs688", "common_name": "LDLR splicing/lipid association marker"},
            {"variant": "LDLR loss-of-function model", "common_name": "LDLR familial hypercholesterolemia loss-of-function model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an LDLR receptor-mediated-endocytosis thesis: the individual may carry context relevant to LDL particle clearance, LDL-C levels, apoB lipoprotein uptake, familial hypercholesterolemia review, or atherosclerotic lipid biology.",
    },
    {
        "gene_name": "HBB",
        "ensembl_id": "ENSG00000244734",
        "ncbi_id": "3043",
        "uniprot_id": "P68871",
        "cytoband": "11p15.4",
        "chromosome": "11",
        "start": 5246694,
        "end": 5250625,
        "strand": "-",
        "biochemical_axis": "beta-globin oxygen transport, hemoglobin tetramer assembly, erythrocyte physiology, and hemoglobinopathy biology",
        "gene_summary": "HBB encodes beta-globin, the beta subunit of adult hemoglobin that pairs with alpha-globin to transport oxygen in red blood cells.",
        "clinical_context": "The local HBB bundle is hemoglobinopathy oriented. HBB findings depend strongly on exact allele, zygosity, phase with other globin variants, hematologic phenotype, and clinical classification.",
        "variant_effect_overview": [
            "HBB missense, nonsense, splice, and regulatory variants can produce sickle hemoglobin, beta-thalassemia, or other hemoglobinopathy contexts.",
            "Carrier and disease interpretations require phase, genotype combination, and hematology data.",
        ],
        "conditions": ["Sickle cell disease and trait context", "Beta-thalassemia and hemoglobinopathy review", "Erythrocyte oxygen transport biology"],
        "visual_key": "hematology",
        "evidence": [_evidence("ClinVar HBB variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=HBB%5Bgene%5D")],
        "variants": [
            {"variant": "rs334", "display_name": "rs334 (HBB HbS / Glu6Val)", "common_name": "HBB sickle hemoglobin marker"},
            {"variant": "HBB beta-thalassemia model", "common_name": "HBB beta-thalassemia variant-class model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an HBB hemoglobinopathy thesis: the individual may carry context relevant to beta-globin structure, oxygen transport, sickle hemoglobin, beta-thalassemia, or carrier/disease review. Interpretation requires exact allele, zygosity, phase, and hematology context.",
    },
    {
        "gene_name": "HFE",
        "ensembl_id": "ENSG00000010704",
        "ncbi_id": "3077",
        "uniprot_id": "Q30201",
        "cytoband": "6p22.2",
        "chromosome": "6",
        "start": 26087509,
        "end": 26098571,
        "strand": "+",
        "biochemical_axis": "iron sensing, transferrin receptor interaction, hepcidin regulation, and systemic iron homeostasis",
        "gene_summary": "HFE encodes an MHC class I-like iron-regulatory protein that interacts with transferrin receptor biology and contributes to hepcidin-mediated systemic iron balance.",
        "clinical_context": "The local HFE bundle is hereditary hemochromatosis and iron-metabolism oriented. Penetrance is incomplete and interpretation depends on genotype combination, sex, ferritin/transferrin saturation, ancestry, and clinical context.",
        "variant_effect_overview": [
            "C282Y and H63D are common HFE markers with genotype-combination-dependent hemochromatosis relevance.",
            "Single heterozygous findings often have limited predictive value without iron studies or compound genotype context.",
        ],
        "conditions": ["Hereditary hemochromatosis review context", "Iron overload and hepcidin regulation", "Transferrin receptor and iron homeostasis biology"],
        "visual_key": "hematology",
        "evidence": [_evidence("ClinVar HFE variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=HFE%5Bgene%5D")],
        "variants": [
            {"variant": "rs1800562", "display_name": "rs1800562 (HFE C282Y)", "common_name": "HFE C282Y iron-overload marker"},
            {"variant": "rs1799945", "display_name": "rs1799945 (HFE H63D)", "common_name": "HFE H63D modifier marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an HFE iron-homeostasis thesis: the individual may carry context relevant to transferrin receptor interaction, hepcidin regulation, iron overload susceptibility, or hereditary hemochromatosis review. Penetrance and clinical meaning depend on genotype combination and iron studies.",
    },
    {
        "gene_name": "ALDH2",
        "ensembl_id": "ENSG00000111275",
        "ncbi_id": "217",
        "uniprot_id": "P05091",
        "cytoband": "12q24.12",
        "chromosome": "12",
        "start": 112204691,
        "end": 112247782,
        "strand": "+",
        "biochemical_axis": "mitochondrial aldehyde detoxification, acetaldehyde metabolism, oxidative-stress handling, and alcohol-response pharmacogenetics",
        "gene_summary": "ALDH2 encodes the mitochondrial aldehyde dehydrogenase that oxidizes acetaldehyde and other reactive aldehydes, linking ethanol metabolism, mitochondrial stress, and cardiovascular/toxicology research.",
        "clinical_context": "The local ALDH2 bundle is pharmacogenetic and mitochondrial-detoxification oriented. ALDH2*2 effects are genotype-dose and ancestry dependent and should not be translated into medical advice without clinical context.",
        "variant_effect_overview": [
            "ALDH2 rs671 reduces ALDH2 enzymatic activity and can affect acetaldehyde accumulation after alcohol exposure.",
            "Interpretation should consider alcohol exposure, ancestry, heterozygous versus homozygous dosage, and broader aldehyde/toxicology context.",
        ],
        "conditions": ["Alcohol flushing and acetaldehyde metabolism", "Mitochondrial aldehyde detoxification", "Cardiometabolic and cancer-risk exposure interaction research"],
        "visual_key": "mitochondria",
        "evidence": [_evidence("PharmGKB ALDH2 gene summary", "https://www.pharmgkb.org/gene/PA24678")],
        "variants": [
            {"variant": "rs671", "display_name": "rs671 (ALDH2 Glu504Lys / ALDH2*2)", "common_name": "ALDH2 reduced-activity alcohol-response marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an ALDH2 aldehyde-detoxification thesis: the individual may carry context relevant to acetaldehyde metabolism, alcohol flushing biology, mitochondrial oxidative stress, and exposure-linked risk studies. Direction depends strongly on rs671 dosage and exposure context.",
    },
    {
        "gene_name": "MTHFR",
        "ensembl_id": "ENSG00000177000",
        "ncbi_id": "4524",
        "uniprot_id": "P42898",
        "cytoband": "1p36.22",
        "chromosome": "1",
        "start": 11845780,
        "end": 11866977,
        "strand": "-",
        "biochemical_axis": "folate one-carbon metabolism, 5-methyltetrahydrofolate production, homocysteine remethylation, and methyl-donor balance",
        "gene_summary": "MTHFR encodes methylenetetrahydrofolate reductase, an enzyme that produces 5-methyltetrahydrofolate for homocysteine remethylation and one-carbon methyl-donor metabolism.",
        "clinical_context": "The local MTHFR bundle is one-carbon-metabolism oriented. Common MTHFR variants are context modifiers and should not be overinterpreted as diagnostic thrombophilia or deterministic disease findings.",
        "variant_effect_overview": [
            "C677T and A1298C can influence enzyme activity and homocysteine context, especially with folate status and genotype dosage.",
            "Clinical interpretation should follow current professional guidance and avoid unsupported deterministic claims.",
        ],
        "conditions": ["Folate and homocysteine metabolism", "Methyl-donor and one-carbon pathway context", "Nutritional and cardiometabolic association research"],
        "visual_key": "metabolic_pathway",
        "evidence": [_evidence("ClinVar MTHFR variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=MTHFR%5Bgene%5D")],
        "variants": [
            {"variant": "rs1801133", "display_name": "rs1801133 (MTHFR C677T / Ala222Val)", "common_name": "MTHFR thermolabile activity marker"},
            {"variant": "rs1801131", "display_name": "rs1801131 (MTHFR A1298C / Glu429Ala)", "common_name": "MTHFR one-carbon modifier marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an MTHFR one-carbon-metabolism thesis: the individual may carry context relevant to folate-dependent methyl-donor balance, homocysteine remethylation, nutritional status, or cardiometabolic association studies. Common MTHFR variants should not be treated as deterministic clinical diagnoses.",
    },
    {
        "gene_name": "LCT",
        "ensembl_id": "ENSG00000115850",
        "ncbi_id": "3938",
        "uniprot_id": "P09848",
        "cytoband": "2q21.3",
        "chromosome": "2",
        "start": 136545410,
        "end": 136594750,
        "strand": "-",
        "biochemical_axis": "intestinal lactose hydrolysis, brush-border disaccharidase activity, enhancer regulation, and lactase persistence biology",
        "gene_summary": "LCT encodes lactase-phlorizin hydrolase, an intestinal brush-border enzyme that digests lactose and whose adult persistence is shaped by upstream regulatory variation.",
        "clinical_context": "The local LCT bundle is nutrition-genetics and regulatory-enhancer oriented. Lactase persistence and intolerance interpretation depends on ancestry, genotype, age, diet, symptoms, and non-genetic gastrointestinal context.",
        "variant_effect_overview": [
            "rs4988235-like upstream enhancer variation is a classic lactase persistence marker in many European-ancestry contexts.",
            "Population specificity matters because different lactase-persistence alleles occur in different ancestry groups.",
        ],
        "conditions": ["Lactase persistence and lactose intolerance context", "Intestinal brush-border carbohydrate digestion", "Nutrition genetics and enhancer regulation"],
        "visual_key": "metabolic_pathway",
        "evidence": [_evidence("OMIM LCT gene entry", "https://omim.org/entry/603202")],
        "variants": [
            {"variant": "rs4988235", "display_name": "rs4988235 (-13910 C>T lactase persistence marker)", "common_name": "LCT/MCM6 enhancer lactase-persistence marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an LCT lactase-persistence thesis: the individual may carry regulatory context relevant to adult lactase expression, lactose digestion, diet-response biology, and ancestry-specific enhancer evolution. Symptom interpretation still depends on diet, age, microbiome, and gastrointestinal context.",
    },
    {
        "gene_name": "BDNF",
        "ensembl_id": "ENSG00000176697",
        "ncbi_id": "627",
        "uniprot_id": "P23560",
        "cytoband": "11p14.1",
        "chromosome": "11",
        "start": 27676440,
        "end": 27743605,
        "strand": "-",
        "biochemical_axis": "neurotrophin signaling, TrkB activation, synaptic plasticity, dendritic remodeling, and activity-dependent neuronal survival",
        "gene_summary": "BDNF encodes brain-derived neurotrophic factor, a secreted neurotrophin that supports neuronal survival, synaptic plasticity, dendritic architecture, learning, memory, mood, and metabolic-neural crosstalk research.",
        "clinical_context": "The local BDNF bundle is neuroplasticity and neuropsychiatric-research oriented. Common BDNF variants are trait modifiers and should not be treated as deterministic psychiatric or cognitive predictions.",
        "variant_effect_overview": [
            "Val66Met is studied for activity-dependent BDNF secretion, synaptic plasticity, memory, stress, mood, and exercise-response contexts.",
            "Direction and magnitude are cohort, phenotype, and ancestry dependent.",
        ],
        "conditions": ["Synaptic plasticity and learning/memory research", "Mood, stress, and neuropsychiatric association context", "Neurotrophin signaling and neuronal survival"],
        "visual_key": "synapse",
        "evidence": [_evidence("UniProt BDNF function annotation", "https://www.uniprot.org/uniprotkb/P23560/entry")],
        "variants": [
            {"variant": "rs6265", "display_name": "rs6265 (BDNF Val66Met)", "common_name": "BDNF activity-dependent secretion marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a BDNF neurotrophin-plasticity thesis: the individual may carry context relevant to activity-dependent BDNF secretion, synaptic plasticity, memory, mood/stress association studies, exercise-response biology, or TrkB pathway signaling. This is not a psychiatric or cognitive diagnosis.",
    },
    {
        "gene_name": "COMT",
        "ensembl_id": "ENSG00000093010",
        "ncbi_id": "1312",
        "uniprot_id": "P21964",
        "cytoband": "22q11.21",
        "chromosome": "22",
        "start": 19929130,
        "end": 19957498,
        "strand": "+",
        "biochemical_axis": "catecholamine methylation, dopamine/norepinephrine turnover, prefrontal neurotransmitter tone, and methyl-donor use",
        "gene_summary": "COMT encodes catechol-O-methyltransferase, an enzyme that methylates catecholamines and catechol estrogens, linking neurotransmitter turnover, pain, stress, cognition, and pharmacogenetic research.",
        "clinical_context": "The local COMT bundle is neurotransmitter-metabolism oriented. Val158Met is a common functional marker but remains a probabilistic research modifier, not a deterministic behavior or psychiatric predictor.",
        "variant_effect_overview": [
            "Val158Met alters COMT enzyme thermostability/activity and is widely studied in dopamine, pain, stress, cognition, and medication-response contexts.",
            "Interpretation should account for sex, tissue isoform, phenotype definition, ancestry, and environment.",
        ],
        "conditions": ["Catecholamine metabolism and prefrontal dopamine tone", "Pain, stress, and cognition association research", "Pharmacogenetic and neuropsychiatric context"],
        "visual_key": "synapse",
        "evidence": [_evidence("PharmGKB COMT gene summary", "https://www.pharmgkb.org/gene/PA117")],
        "variants": [
            {"variant": "rs4680", "display_name": "rs4680 (COMT Val158Met)", "common_name": "COMT catecholamine-metabolism activity marker"},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a COMT catecholamine-metabolism thesis: the individual may carry context relevant to dopamine/norepinephrine turnover, prefrontal signaling, pain/stress traits, methyl-donor use, or pharmacogenetic research. It should not be treated as a deterministic personality or psychiatric call.",
    },
    {
        "gene_name": "CACNA1C",
        "ensembl_id": "ENSG00000151067",
        "ncbi_id": "775",
        "uniprot_id": "Q13936",
        "cytoband": "12p13.33",
        "chromosome": "12",
        "start": 2079952,
        "end": 2802108,
        "strand": "+",
        "biochemical_axis": "L-type voltage-gated calcium channel function, excitation-transcription coupling, cardiac action potentials, and neuronal calcium signaling",
        "gene_summary": "CACNA1C encodes the alpha-1C subunit of the L-type voltage-gated calcium channel Cav1.2, central to cardiac excitation-contraction coupling, action potentials, and neuronal calcium signaling.",
        "clinical_context": "The local CACNA1C bundle is cardiac and neuropsychiatric channel-biology oriented. Rare pathogenic variants have separate clinical implications from common GWAS markers.",
        "variant_effect_overview": [
            "Common CACNA1C regulatory markers are studied in psychiatric, neurodevelopmental, and electrophysiology contexts.",
            "Rare high-impact CACNA1C variants require clinical classification and phenotype matching for Timothy syndrome or arrhythmia review.",
        ],
        "conditions": ["Cardiac electrophysiology and action-potential biology", "Neuronal calcium signaling and psychiatric GWAS context", "L-type calcium channel pharmacology"],
        "visual_key": "ion_channel",
        "evidence": [_evidence("ClinVar CACNA1C variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=CACNA1C%5Bgene%5D")],
        "variants": [
            {"variant": "rs1006737", "common_name": "CACNA1C psychiatric/calcium-channel regulatory marker"},
            {"variant": "CACNA1C high-impact channelopathy model", "common_name": "CACNA1C Timothy/arrhythmia channelopathy model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a CACNA1C calcium-channel thesis: the individual may carry context relevant to L-type calcium channel gating/expression, cardiac action potentials, neuronal calcium signaling, psychiatric GWAS biology, or rare channelopathy review depending on exact allele class.",
    },
    {
        "gene_name": "SCN5A",
        "ensembl_id": "ENSG00000183873",
        "ncbi_id": "6331",
        "uniprot_id": "Q14524",
        "cytoband": "3p22.2",
        "chromosome": "3",
        "start": 38589548,
        "end": 38691164,
        "strand": "-",
        "biochemical_axis": "cardiac voltage-gated sodium channel Nav1.5 gating, conduction velocity, depolarization, and arrhythmia biology",
        "gene_summary": "SCN5A encodes the Nav1.5 cardiac sodium channel alpha subunit, the main driver of fast inward sodium current in cardiomyocyte depolarization and cardiac conduction.",
        "clinical_context": "The local SCN5A bundle is cardiac-channelopathy oriented. SCN5A interpretation requires exact allele, ECG/phenotype, inheritance, ancestry, and expert variant classification.",
        "variant_effect_overview": [
            "SCN5A variants can produce gain- or loss-of-function effects depending on channel gating, trafficking, and cellular context.",
            "Common modifiers such as H558R are not equivalent to pathogenic Brugada, long-QT, conduction disease, or cardiomyopathy variants.",
        ],
        "conditions": ["Cardiac conduction and arrhythmia biology", "Brugada/long-QT/conduction disease review context", "Nav1.5 channel gating and trafficking"],
        "visual_key": "ion_channel",
        "evidence": [_evidence("ClinVar SCN5A variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=SCN5A%5Bgene%5D")],
        "variants": [
            {"variant": "rs1805124", "display_name": "rs1805124 (SCN5A H558R)", "common_name": "SCN5A common channel-modifier marker"},
            {"variant": "SCN5A pathogenic channelopathy model", "common_name": "SCN5A arrhythmia-channelopathy model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an SCN5A Nav1.5 thesis: the individual may carry context relevant to cardiac sodium current, conduction velocity, arrhythmia susceptibility, channel trafficking/gating, or clinical channelopathy review. Exact classification and phenotype are essential.",
    },
    {
        "gene_name": "MYH7",
        "ensembl_id": "ENSG00000092054",
        "ncbi_id": "4625",
        "uniprot_id": "P12883",
        "cytoband": "14q11.2",
        "chromosome": "14",
        "start": 23881947,
        "end": 23904927,
        "strand": "-",
        "biochemical_axis": "beta-myosin heavy-chain motor activity, sarcomere contraction, ATPase force generation, and cardiomyopathy biology",
        "gene_summary": "MYH7 encodes beta-myosin heavy chain, a cardiac and slow-skeletal muscle motor protein that converts ATP hydrolysis into sarcomere force generation.",
        "clinical_context": "The local MYH7 bundle is sarcomere and cardiomyopathy oriented. High-impact MYH7 variants require cardiac phenotype, family history, inheritance, and expert variant classification.",
        "variant_effect_overview": [
            "MYH7 missense variants can alter myosin motor function and are important in hypertrophic, dilated, or other cardiomyopathy contexts.",
            "Variant location, conservation, segregation, and phenotype matching are critical for interpretation.",
        ],
        "conditions": ["Hypertrophic and dilated cardiomyopathy review context", "Sarcomere force generation and ATPase motor biology", "Cardiac muscle structure-function research"],
        "visual_key": "cardiac_contractile",
        "evidence": [_evidence("ClinVar MYH7 variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=MYH7%5Bgene%5D")],
        "variants": [
            {"variant": "MYH7 p.Arg403Gln", "display_name": "MYH7 p.Arg403Gln / R403Q", "common_name": "MYH7 hypertrophic cardiomyopathy marker"},
            {"variant": "MYH7 sarcomere motor-function model", "common_name": "MYH7 sarcomere motor-function model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a MYH7 sarcomere-motor thesis: the individual may carry context relevant to beta-myosin force generation, cardiomyocyte contraction, hypertrophic or dilated cardiomyopathy review, or muscle ATPase biology. Clinical meaning requires variant classification and phenotype match.",
    },
    {
        "gene_name": "NTRK1",
        "ensembl_id": "ENSG00000198400",
        "ncbi_id": "4914",
        "uniprot_id": "P04629",
        "cytoband": "1q23.1",
        "chromosome": "1",
        "start": 156785432,
        "end": 156851642,
        "strand": "+",
        "biochemical_axis": "TrkA neurotrophin receptor signaling through NGF, RAS-MAPK, PI3K-AKT, PLC-gamma, pain pathways, and oncogenic fusions",
        "gene_summary": "NTRK1 encodes TrkA, the high-affinity nerve growth factor receptor that controls neuronal survival, pain-sensory neuron biology, differentiation, and kinase signaling.",
        "clinical_context": "The local NTRK1 bundle spans neurodevelopmental/pain biology and somatic fusion oncology. Germline loss-of-function and tumor fusion contexts must be kept separate.",
        "variant_effect_overview": [
            "NTRK1 loss-of-function can be relevant to congenital insensitivity to pain contexts when biallelic and clinically supported.",
            "NTRK1 fusions are somatic oncology markers and require tumor assay context.",
        ],
        "conditions": ["NGF-TrkA neuronal survival and pain biology", "Congenital insensitivity to pain review context", "NTRK fusion oncology target context"],
        "visual_key": "receptor_signaling",
        "evidence": [_evidence("ClinVar NTRK1 variation set", "https://www.ncbi.nlm.nih.gov/clinvar/?term=NTRK1%5Bgene%5D")],
        "variants": [
            {"variant": "NTRK1 fusion model", "common_name": "NTRK1 oncogenic kinase-fusion model", "is_assayable_in_snp_vcf": False},
            {"variant": "NTRK1 loss-of-function model", "common_name": "NTRK1 congenital pain-insensitivity model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests an NTRK1 TrkA signaling thesis: the individual may carry context relevant to NGF receptor signaling, pain-sensory neuron development, neuronal survival, kinase-pathway activation, or somatic NTRK fusion review. Germline and tumor contexts must be separated.",
    },
    {
        "gene_name": "PPARG",
        "ensembl_id": "ENSG00000132170",
        "ncbi_id": "5468",
        "uniprot_id": "P37231",
        "cytoband": "3p25.2",
        "chromosome": "3",
        "start": 12328867,
        "end": 12475855,
        "strand": "+",
        "biochemical_axis": "nuclear receptor transcriptional control of adipogenesis, insulin sensitivity, lipid storage, inflammation, and thiazolidinedione pharmacology",
        "gene_summary": "PPARG encodes peroxisome proliferator-activated receptor gamma, a ligand-activated nuclear receptor that regulates adipocyte differentiation, lipid storage, insulin sensitivity, macrophage polarization, and metabolic inflammation.",
        "clinical_context": "The local PPARG bundle is metabolic and pharmacology oriented. Common PPARG markers are low-penetrance modifiers, while rare dominant-negative variants require clinical classification.",
        "variant_effect_overview": [
            "Pro12Ala is a widely studied PPARG functional/metabolic marker with context-dependent effects on insulin sensitivity and metabolic traits.",
            "Drug-response interpretation for thiazolidinediones should remain cautious and phenotype/treatment specific.",
        ],
        "conditions": ["Adipogenesis and insulin-sensitivity biology", "Type 2 diabetes and metabolic trait association research", "PPAR-gamma agonist pharmacology"],
        "visual_key": "lipid_metabolism",
        "evidence": [_evidence("PharmGKB PPARG gene summary", "https://www.pharmgkb.org/gene/PA281")],
        "variants": [
            {"variant": "rs1801282", "display_name": "rs1801282 (PPARG Pro12Ala)", "common_name": "PPARG adipogenesis/insulin-sensitivity marker"},
            {"variant": "PPARG dominant-negative model", "common_name": "PPARG severe insulin-resistance model", "is_assayable_in_snp_vcf": False},
        ],
        "concrete_variant_prediction": "The variant observed in this sample suggests a PPARG adipogenesis and insulin-sensitivity thesis: the individual may carry context relevant to lipid storage, adipocyte differentiation, metabolic inflammation, type 2 diabetes traits, or PPAR-gamma agonist pharmacology. Common markers remain probabilistic modifiers.",
    },
]


def _build_extended_biochemistry_gene_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expand compact gene metadata into the full biochemistry bundle schema."""
    gene_name = str(config["gene_name"])
    axis = str(config["biochemical_axis"])
    conditions = list(config["conditions"])
    bundle_focus = str(config.get("bundle_focus") or (conditions[0] if conditions else axis))
    function = str(config["function"])
    variant_note = str(
        config.get("variant_note")
        or "exact allele class, zygosity, ancestry, phenotype, and assay context"
    )
    expanded = dict(config)
    expanded["gene_summary"] = str(
        config.get("gene_summary")
        or f"{gene_name} encodes {function}. This bundle emphasizes {axis}."
    )
    expanded["clinical_context"] = str(
        config.get("clinical_context")
        or (
            f"The local {gene_name} bundle is {bundle_focus} oriented. "
            f"Interpret markers through {variant_note} rather than as stand-alone deterministic calls."
        )
    )
    expanded["variant_effect_overview"] = config.get("variant_effect_overview") or [
        f"{gene_name} variants can alter or tag {axis}; effect direction depends on allele class, tissue, dosage, and biological context.",
        "Common markers, pharmacogenetic alleles, structural events, and high-impact pathogenic variants should be kept separate during interpretation.",
    ]
    expanded["variants"] = config.get("variants") or [
        {
            "variant": f"{gene_name} functional variation model",
            "common_name": f"{gene_name} pathway-function model",
            "is_assayable_in_snp_vcf": False,
        }
    ]
    expanded["figure_focus"] = str(config.get("figure_focus") or f"{gene_name}: {axis}")
    expanded["concrete_variant_prediction"] = str(
        config.get("concrete_variant_prediction")
        or (
            f"The variant observed in this sample suggests a {gene_name} {bundle_focus} thesis: "
            f"the individual may carry context relevant to {axis} and {', '.join(conditions[:2])}. "
            "Treat this as pathway triage until the exact allele has external classification and phenotype support."
        )
    )
    return expanded


EXTENDED_BIOCHEMISTRY_GENE_CONFIGS: list[dict[str, Any]] = [
    {
        "gene_name": "KRAS",
        "ensembl_id": "ENSG00000133703",
        "ncbi_id": "3845",
        "uniprot_id": "P01116",
        "cytoband": "12p12.1",
        "chromosome": "12",
        "start": 25357723,
        "end": 25403870,
        "strand": "-",
        "function": "KRAS, a membrane-associated small GTPase that couples receptor input to RAF-MEK-ERK and PI3K signaling",
        "biochemical_axis": "RAS GTPase switching, MAPK pathway activation, PI3K crosstalk, proliferation, differentiation, and oncogenic signaling",
        "bundle_focus": "oncogenic RAS signaling",
        "conditions": ["Somatic oncology and targeted-therapy context", "RAS-MAPK pathway biology", "Developmental RASopathy review context"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "KRAS codon 12/13/61 model", "common_name": "KRAS activating hotspot model", "is_assayable_in_snp_vcf": False},
            {"variant": "rs61764370", "display_name": "rs61764370 (KRAS 3'UTR marker)", "common_name": "KRAS 3'UTR regulatory association marker"},
        ],
    },
    {
        "gene_name": "BRAF",
        "ensembl_id": "ENSG00000157764",
        "ncbi_id": "673",
        "uniprot_id": "P15056",
        "cytoband": "7q34",
        "chromosome": "7",
        "start": 140419127,
        "end": 140624564,
        "strand": "-",
        "function": "B-Raf serine/threonine kinase, a core RAF family MAPK pathway effector",
        "biochemical_axis": "RAF kinase activation, MEK-ERK signaling, proliferation, differentiation, and kinase-inhibitor response biology",
        "bundle_focus": "MAPK kinase signaling",
        "conditions": ["Somatic oncology and kinase-inhibitor context", "RASopathy and developmental signaling review", "MAPK pathway activation"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "BRAF V600E model", "common_name": "BRAF activating kinase hotspot model", "is_assayable_in_snp_vcf": False},
            {"variant": "BRAF non-V600 kinase-altered model", "common_name": "BRAF non-V600 signaling model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "PIK3CA",
        "ensembl_id": "ENSG00000121879",
        "ncbi_id": "5290",
        "uniprot_id": "P42336",
        "cytoband": "3q26.32",
        "chromosome": "3",
        "start": 178865902,
        "end": 178957881,
        "strand": "+",
        "function": "the p110-alpha catalytic subunit of class IA phosphoinositide 3-kinase",
        "biochemical_axis": "PI3K lipid kinase signaling, PIP3 production, AKT-mTOR activation, cell growth, metabolism, and survival",
        "bundle_focus": "PI3K pathway signaling",
        "conditions": ["PI3K-AKT-mTOR signaling and cancer context", "Overgrowth and mosaic activation review", "Targeted PI3K inhibitor biology"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "PIK3CA E542K/E545K/H1047R model", "common_name": "PIK3CA activating hotspot model", "is_assayable_in_snp_vcf": False},
            {"variant": "PIK3CA mosaic overgrowth model", "common_name": "PIK3CA-related overgrowth spectrum model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "PTEN",
        "ensembl_id": "ENSG00000171862",
        "ncbi_id": "5728",
        "uniprot_id": "P60484",
        "cytoband": "10q23.31",
        "chromosome": "10",
        "start": 89622870,
        "end": 89731687,
        "strand": "+",
        "function": "PTEN, a lipid and protein phosphatase that opposes PI3K signaling",
        "biochemical_axis": "PIP3 dephosphorylation, AKT pathway restraint, cell growth control, genomic stability, and tumor-suppressor biology",
        "bundle_focus": "PI3K restraint and tumor-suppressor review",
        "conditions": ["PTEN hamartoma tumor syndrome review context", "PI3K-AKT pathway regulation", "Somatic tumor-suppressor loss context"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "PTEN loss-of-function model", "common_name": "PTEN tumor-suppressor loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "PTEN promoter/regulatory loss model", "common_name": "PTEN expression-restraint model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "AKT1",
        "ensembl_id": "ENSG00000142208",
        "ncbi_id": "207",
        "uniprot_id": "P31749",
        "cytoband": "14q32.33",
        "chromosome": "14",
        "start": 105235686,
        "end": 105262088,
        "strand": "-",
        "function": "AKT1, a serine/threonine kinase that integrates PI3K input with survival and growth programs",
        "biochemical_axis": "AKT kinase signaling, glucose and growth-factor response, apoptosis restraint, mTOR activation, and mosaic activation biology",
        "bundle_focus": "AKT signaling",
        "conditions": ["PI3K-AKT-mTOR pathway biology", "Somatic oncology and mosaic overgrowth context", "Growth-factor survival signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "AKT1 E17K model", "common_name": "AKT1 activating pleckstrin-homology-domain model", "is_assayable_in_snp_vcf": False},
            {"variant": "AKT1 pathway-activation model", "common_name": "AKT1 signaling activation model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "ALK",
        "ensembl_id": "ENSG00000171094",
        "ncbi_id": "238",
        "uniprot_id": "Q9UM73",
        "cytoband": "2p23.1 p23.2",
        "chromosome": "2",
        "start": 29415640,
        "end": 30144432,
        "strand": "-",
        "function": "anaplastic lymphoma kinase, a receptor tyrosine kinase with developmental and oncogenic roles",
        "biochemical_axis": "ALK receptor kinase signaling, fusion activation, MAPK/PI3K/JAK pathway output, and kinase-inhibitor response",
        "bundle_focus": "ALK kinase and fusion oncology",
        "conditions": ["ALK fusion and somatic oncology context", "Neuroblastoma predisposition review context", "Receptor tyrosine kinase signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "ALK fusion model", "common_name": "ALK oncogenic kinase-fusion model", "is_assayable_in_snp_vcf": False},
            {"variant": "ALK activating kinase-domain model", "common_name": "ALK kinase-domain activation model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "ROS1",
        "ensembl_id": "ENSG00000047936",
        "ncbi_id": "6098",
        "uniprot_id": "P08922",
        "cytoband": "6q22.1",
        "chromosome": "6",
        "start": 117609463,
        "end": 117747018,
        "strand": "-",
        "function": "ROS1, a receptor tyrosine kinase best known for oncogenic fusion activation",
        "biochemical_axis": "ROS1 kinase fusion signaling, MAPK/PI3K/JAK pathway output, cell survival, and targeted-therapy biology",
        "bundle_focus": "ROS1 fusion kinase signaling",
        "conditions": ["ROS1 fusion oncology context", "Tyrosine kinase inhibitor response", "Receptor kinase signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "ROS1 fusion model", "common_name": "ROS1 oncogenic kinase-fusion model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "RET",
        "ensembl_id": "ENSG00000165731",
        "ncbi_id": "5979",
        "uniprot_id": "P07949",
        "cytoband": "10q11.21",
        "chromosome": "10",
        "start": 43572475,
        "end": 43625799,
        "strand": "+",
        "function": "RET, a receptor tyrosine kinase for GDNF-family neurotrophic signaling",
        "biochemical_axis": "RET receptor kinase signaling, neural-crest development, MAPK/PI3K activation, endocrine tumor biology, and fusion oncology",
        "bundle_focus": "RET receptor kinase biology",
        "conditions": ["MEN2 and medullary thyroid carcinoma review context", "Hirschsprung disease developmental context", "RET fusion oncology context"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "RET MEN2 activating model", "common_name": "RET endocrine tumor activation model", "is_assayable_in_snp_vcf": False},
            {"variant": "RET fusion model", "common_name": "RET oncogenic kinase-fusion model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "MET",
        "ensembl_id": "ENSG00000105976",
        "ncbi_id": "4233",
        "uniprot_id": "P08581",
        "cytoband": "7q31.2",
        "chromosome": "7",
        "start": 116312444,
        "end": 116438440,
        "strand": "+",
        "function": "MET, the hepatocyte growth factor receptor tyrosine kinase",
        "biochemical_axis": "HGF-MET receptor signaling, motility, invasion, epithelial repair, PI3K/MAPK activation, and exon-14 skipping oncology",
        "bundle_focus": "MET receptor signaling",
        "conditions": ["MET exon 14 skipping and oncology context", "HGF-driven motility and invasion biology", "Receptor tyrosine kinase signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "MET exon 14 skipping model", "common_name": "MET juxtamembrane regulation-loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "MET amplification/activation model", "common_name": "MET receptor activation model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "ERBB2",
        "ensembl_id": "ENSG00000141736",
        "ncbi_id": "2064",
        "uniprot_id": "P04626",
        "cytoband": "17q12",
        "chromosome": "17",
        "start": 37844167,
        "end": 37886679,
        "strand": "+",
        "function": "HER2/ERBB2, an ErbB family receptor tyrosine kinase dimerization partner",
        "biochemical_axis": "HER2 receptor dimer signaling, MAPK/PI3K activation, amplification biology, and anti-HER2 therapy response",
        "bundle_focus": "HER2 receptor signaling",
        "conditions": ["HER2-amplified oncology context", "ErbB receptor signaling", "Targeted antibody and kinase-inhibitor biology"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "ERBB2 amplification model", "common_name": "HER2 amplification expression model", "is_assayable_in_snp_vcf": False},
            {"variant": "ERBB2 activating kinase-domain model", "common_name": "HER2 kinase activation model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "ESR1",
        "ensembl_id": "ENSG00000091831",
        "ncbi_id": "2099",
        "uniprot_id": "P03372",
        "cytoband": "6q25.1",
        "chromosome": "6",
        "start": 151977826,
        "end": 152450754,
        "strand": "+",
        "function": "estrogen receptor alpha, a ligand-regulated nuclear receptor transcription factor",
        "biochemical_axis": "estrogen receptor transcriptional control, chromatin binding, endocrine signaling, and anti-estrogen response biology",
        "bundle_focus": "steroid hormone receptor signaling",
        "conditions": ["Endocrine-responsive breast cancer context", "Estrogen signaling and reproductive tissue biology", "Bone, vascular, and metabolic association research"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "rs2234693", "display_name": "rs2234693 (ESR1 PvuII marker)", "common_name": "ESR1 intronic association marker"},
            {"variant": "ESR1 ligand-binding-domain resistance model", "common_name": "ESR1 endocrine-resistance model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "AR",
        "ensembl_id": "ENSG00000169083",
        "ncbi_id": "367",
        "uniprot_id": "P10275",
        "cytoband": "Xq12",
        "chromosome": "X",
        "start": 66764465,
        "end": 66950461,
        "strand": "+",
        "function": "the androgen receptor, a ligand-regulated nuclear receptor transcription factor",
        "biochemical_axis": "androgen receptor transcriptional control, steroid signaling, CAG-repeat modulation, prostate biology, and endocrine therapy response",
        "bundle_focus": "androgen receptor signaling",
        "conditions": ["Androgen signaling and prostate biology", "Androgen insensitivity and endocrine review context", "Prostate cancer resistance biology"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "AR CAG-repeat model", "common_name": "AR polyglutamine-length modulation model", "is_assayable_in_snp_vcf": False},
            {"variant": "AR ligand-binding-domain resistance model", "common_name": "AR endocrine-resistance model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "VHL",
        "ensembl_id": "ENSG00000134086",
        "ncbi_id": "7428",
        "uniprot_id": "P40337",
        "cytoband": "3p25.3",
        "chromosome": "3",
        "start": 10182692,
        "end": 10193904,
        "strand": "+",
        "function": "VHL, the substrate-recognition component of an E3 ligase complex that targets HIF-alpha",
        "biochemical_axis": "oxygen sensing, HIF-alpha ubiquitination, hypoxia-response control, angiogenesis, and tumor-suppressor biology",
        "bundle_focus": "hypoxia and tumor-suppressor review",
        "conditions": ["Von Hippel-Lindau disease review context", "Renal and endocrine tumor biology", "Hypoxia-inducible factor pathway control"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "VHL loss-of-function model", "common_name": "VHL HIF-regulation loss model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "APC",
        "ensembl_id": "ENSG00000134982",
        "ncbi_id": "324",
        "uniprot_id": "P25054",
        "cytoband": "5q22.2",
        "chromosome": "5",
        "start": 112043195,
        "end": 112181936,
        "strand": "+",
        "function": "APC, a tumor-suppressor scaffold in the beta-catenin destruction complex",
        "biochemical_axis": "Wnt/beta-catenin restraint, cell adhesion, chromosome segregation, intestinal stem-cell control, and colorectal tumor biology",
        "bundle_focus": "Wnt pathway tumor-suppressor review",
        "conditions": ["Familial adenomatous polyposis review context", "Colorectal tumor-suppressor biology", "Wnt/beta-catenin signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "APC truncating-variant model", "common_name": "APC Wnt-restraint loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "APC I1307K model", "common_name": "APC colorectal-risk modifier model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "MLH1",
        "ensembl_id": "ENSG00000076242",
        "ncbi_id": "4292",
        "uniprot_id": "P40692",
        "cytoband": "3p22.2",
        "chromosome": "3",
        "start": 37034823,
        "end": 37107380,
        "strand": "+",
        "function": "MLH1, a DNA mismatch-repair factor that partners with PMS2",
        "biochemical_axis": "DNA mismatch repair, microsatellite stability, replication-error correction, Lynch syndrome biology, and promoter methylation context",
        "bundle_focus": "mismatch-repair review",
        "conditions": ["Lynch syndrome review context", "Microsatellite instability and colorectal/endometrial tumor biology", "Mismatch-repair pathway function"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "MLH1 loss-of-function model", "common_name": "MLH1 mismatch-repair loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "MLH1 promoter methylation model", "common_name": "MLH1 epigenetic silencing model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "MSH2",
        "ensembl_id": "ENSG00000095002",
        "ncbi_id": "4436",
        "uniprot_id": "P43246",
        "cytoband": "2p21",
        "chromosome": "2",
        "start": 47630108,
        "end": 47789450,
        "strand": "+",
        "function": "MSH2, a DNA mismatch-recognition factor that pairs with MSH6 or MSH3",
        "biochemical_axis": "DNA mismatch recognition, microsatellite stability, replication-error repair, and Lynch syndrome biology",
        "bundle_focus": "mismatch-repair review",
        "conditions": ["Lynch syndrome review context", "Microsatellite instability biology", "DNA mismatch-recognition function"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "MSH2 loss-of-function model", "common_name": "MSH2 mismatch-recognition loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "EPCAM-MSH2 silencing model", "common_name": "MSH2 epigenetic-silencing context model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "MSH6",
        "ensembl_id": "ENSG00000116062",
        "ncbi_id": "2956",
        "uniprot_id": "P52701",
        "cytoband": "2p16.3",
        "chromosome": "2",
        "start": 47922669,
        "end": 48037240,
        "strand": "+",
        "function": "MSH6, the mismatch-recognition partner in the MutS-alpha complex",
        "biochemical_axis": "single-base mismatch recognition, insertion/deletion-loop repair, microsatellite stability, and Lynch syndrome biology",
        "bundle_focus": "mismatch-repair review",
        "conditions": ["Lynch syndrome review context", "Mismatch-repair pathway function", "Microsatellite instability biology"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "MSH6 loss-of-function model", "common_name": "MSH6 mismatch-recognition loss model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "PMS2",
        "ensembl_id": "ENSG00000122512",
        "ncbi_id": "5395",
        "uniprot_id": "P54278",
        "cytoband": "7p22.1",
        "chromosome": "7",
        "start": 6012870,
        "end": 6048756,
        "strand": "-",
        "function": "PMS2, the endonuclease partner of MLH1 in mismatch repair",
        "biochemical_axis": "MutL-alpha mismatch repair, strand incision, replication-error correction, and Lynch syndrome biology",
        "bundle_focus": "mismatch-repair review",
        "conditions": ["Lynch syndrome review context", "Constitutional mismatch-repair deficiency review", "DNA mismatch-repair function"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "PMS2 loss-of-function model", "common_name": "PMS2 mismatch-repair loss model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "ATM",
        "ensembl_id": "ENSG00000149311",
        "ncbi_id": "472",
        "uniprot_id": "Q13315",
        "cytoband": "11q22.3",
        "chromosome": "11",
        "start": 108093211,
        "end": 108239829,
        "strand": "+",
        "function": "ATM, a DNA damage response kinase activated by double-strand breaks",
        "biochemical_axis": "DNA double-strand break signaling, checkpoint kinase cascades, p53 activation, radiosensitivity, and genome stability",
        "bundle_focus": "DNA damage response kinase review",
        "conditions": ["Ataxia-telangiectasia and carrier review context", "Hereditary cancer susceptibility context", "Radiation response and checkpoint biology"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "ATM loss-of-function model", "common_name": "ATM checkpoint-kinase loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "rs1801516", "display_name": "rs1801516 (ATM D1853N)", "common_name": "ATM radiosensitivity association marker"},
        ],
    },
    {
        "gene_name": "CHEK2",
        "ensembl_id": "ENSG00000183765",
        "ncbi_id": "11200",
        "uniprot_id": "O96017",
        "cytoband": "22q12.1",
        "chromosome": "22",
        "start": 29083731,
        "end": 29138410,
        "strand": "-",
        "function": "CHK2, a checkpoint kinase downstream of ATM",
        "biochemical_axis": "DNA damage checkpoint signaling, p53 and BRCA pathway crosstalk, cell-cycle arrest, and hereditary cancer modifier biology",
        "bundle_focus": "checkpoint kinase review",
        "conditions": ["Hereditary cancer susceptibility context", "DNA damage checkpoint biology", "Breast, colorectal, and prostate cancer association research"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "CHEK2 c.1100delC model", "common_name": "CHEK2 truncating founder model", "is_assayable_in_snp_vcf": False},
            {"variant": "rs17879961", "display_name": "rs17879961 (CHEK2 I157T)", "common_name": "CHEK2 missense modifier marker"},
        ],
    },
    {
        "gene_name": "PALB2",
        "ensembl_id": "ENSG00000083093",
        "ncbi_id": "79728",
        "uniprot_id": "Q86YC2",
        "cytoband": "16p12.2",
        "chromosome": "16",
        "start": 23614488,
        "end": 23652631,
        "strand": "-",
        "function": "PALB2, a BRCA2-localizing scaffold for homologous recombination",
        "biochemical_axis": "homologous recombination, BRCA1-BRCA2 coordination, DNA double-strand break repair, and hereditary cancer biology",
        "bundle_focus": "homologous recombination review",
        "conditions": ["Hereditary breast/pancreatic cancer review context", "Homologous recombination repair", "BRCA-pathway genome stability"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "PALB2 loss-of-function model", "common_name": "PALB2 homologous-recombination loss model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "BRCA2",
        "ensembl_id": "ENSG00000139618",
        "ncbi_id": "675",
        "uniprot_id": "P51587",
        "cytoband": "13q13.1",
        "chromosome": "13",
        "start": 32889611,
        "end": 32973805,
        "strand": "+",
        "function": "BRCA2, a homologous recombination mediator that loads RAD51 onto DNA",
        "biochemical_axis": "RAD51 loading, homologous recombination repair, replication-fork protection, and hereditary cancer biology",
        "bundle_focus": "homologous recombination review",
        "conditions": ["Hereditary breast/ovarian/pancreatic/prostate cancer context", "Homologous recombination deficiency", "Replication-fork protection"],
        "visual_key": "dna_repair",
        "variants": [
            {"variant": "BRCA2 c.5946delT / 6174delT model", "common_name": "BRCA2 frameshift founder model", "is_assayable_in_snp_vcf": False},
            {"variant": "BRCA2 loss-of-function model", "common_name": "BRCA2 homologous-recombination loss model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "RYR1",
        "ensembl_id": "ENSG00000196218",
        "ncbi_id": "6261",
        "uniprot_id": "P21817",
        "cytoband": "19q13.2",
        "chromosome": "19",
        "start": 38924339,
        "end": 39078204,
        "strand": "+",
        "function": "the skeletal-muscle ryanodine receptor calcium-release channel",
        "biochemical_axis": "sarcoplasmic-reticulum calcium release, excitation-contraction coupling, malignant hyperthermia biology, and myopathy context",
        "bundle_focus": "skeletal muscle calcium-channel review",
        "conditions": ["Malignant hyperthermia susceptibility review context", "Congenital myopathy context", "Excitation-contraction coupling"],
        "visual_key": "ion_channel",
        "variants": [
            {"variant": "RYR1 malignant-hyperthermia model", "common_name": "RYR1 calcium-release susceptibility model", "is_assayable_in_snp_vcf": False},
            {"variant": "RYR1 congenital myopathy model", "common_name": "RYR1 skeletal-muscle channelopathy model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "KCNQ1",
        "ensembl_id": "ENSG00000053918",
        "ncbi_id": "3784",
        "uniprot_id": "P51787",
        "cytoband": "11p15.4 p15.5",
        "chromosome": "11",
        "start": 2465914,
        "end": 2870339,
        "strand": "+",
        "function": "the Kv7.1 voltage-gated potassium channel alpha subunit",
        "biochemical_axis": "cardiac repolarization, IKs potassium current, beta-cell electrophysiology, imprinting-region context, and channel pharmacology",
        "bundle_focus": "potassium-channel review",
        "conditions": ["Long-QT syndrome review context", "Cardiac repolarization biology", "Type 2 diabetes association research"],
        "visual_key": "ion_channel",
        "variants": [
            {"variant": "KCNQ1 long-QT model", "common_name": "KCNQ1 IKs channelopathy model", "is_assayable_in_snp_vcf": False},
            {"variant": "rs2237892", "common_name": "KCNQ1 diabetes-association marker"},
        ],
    },
    {
        "gene_name": "KCNH2",
        "ensembl_id": "ENSG00000055118",
        "ncbi_id": "3757",
        "uniprot_id": "Q12809",
        "cytoband": "7q36.1",
        "chromosome": "7",
        "start": 150642049,
        "end": 150675403,
        "strand": "-",
        "function": "the hERG/Kv11.1 voltage-gated potassium channel",
        "biochemical_axis": "cardiac IKr current, ventricular repolarization, drug-induced QT sensitivity, trafficking, and arrhythmia biology",
        "bundle_focus": "hERG potassium-channel review",
        "conditions": ["Long-QT syndrome review context", "Drug-induced QT prolongation biology", "Cardiac repolarization and channel trafficking"],
        "visual_key": "ion_channel",
        "variants": [
            {"variant": "KCNH2 long-QT model", "common_name": "KCNH2 hERG channelopathy model", "is_assayable_in_snp_vcf": False},
            {"variant": "KCNH2 drug-sensitivity model", "common_name": "hERG pharmacology and QT-risk model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "APOB",
        "ensembl_id": "ENSG00000084674",
        "ncbi_id": "338",
        "uniprot_id": "P04114",
        "cytoband": "2p24.1",
        "chromosome": "2",
        "start": 21224301,
        "end": 21266945,
        "strand": "-",
        "function": "apolipoprotein B, the structural apolipoprotein of LDL and other atherogenic lipoproteins",
        "biochemical_axis": "LDL particle structure, apoB-LDLR binding, triglyceride-rich lipoprotein metabolism, and familial hypercholesterolemia biology",
        "bundle_focus": "atherogenic lipoprotein biology",
        "conditions": ["Familial hypercholesterolemia review context", "LDL cholesterol and atherosclerosis biology", "Hypobetalipoproteinemia context"],
        "visual_key": "lipid_metabolism",
        "variants": [
            {"variant": "rs693", "display_name": "rs693 (APOB lipid association marker)", "common_name": "APOB common lipid marker"},
            {"variant": "APOB familial-hypercholesterolemia model", "common_name": "APOB LDLR-binding disruption model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "F5",
        "ensembl_id": "ENSG00000198734",
        "ncbi_id": "2153",
        "uniprot_id": "P12259",
        "cytoband": "1q24.2",
        "chromosome": "1",
        "start": 169483404,
        "end": 169555826,
        "strand": "-",
        "function": "coagulation factor V, a cofactor for prothrombinase complex assembly",
        "biochemical_axis": "thrombin generation, activated protein C resistance, clotting cascade amplification, and venous thrombosis biology",
        "bundle_focus": "coagulation and thrombosis review",
        "conditions": ["Factor V Leiden thrombophilia context", "Venous thromboembolism association research", "Coagulation cascade biology"],
        "visual_key": "hematology",
        "variants": [
            {"variant": "rs6025", "display_name": "rs6025 (Factor V Leiden / R506Q)", "common_name": "Factor V Leiden APC-resistance marker"},
        ],
    },
    {
        "gene_name": "F2",
        "ensembl_id": "ENSG00000180210",
        "ncbi_id": "2147",
        "uniprot_id": "P00734",
        "cytoband": "11p11.2",
        "chromosome": "11",
        "start": 46740730,
        "end": 46761056,
        "strand": "+",
        "function": "prothrombin, the precursor of thrombin in the coagulation cascade",
        "biochemical_axis": "thrombin generation, fibrin clot formation, platelet activation, and prothrombotic regulatory biology",
        "bundle_focus": "coagulation and thrombosis review",
        "conditions": ["Prothrombin thrombophilia context", "Venous thromboembolism association research", "Coagulation cascade biology"],
        "visual_key": "hematology",
        "variants": [
            {"variant": "rs1799963", "display_name": "rs1799963 (F2 G20210A)", "common_name": "prothrombin 20210A thrombosis marker"},
        ],
    },
    {
        "gene_name": "SERPINA1",
        "ensembl_id": "ENSG00000197249",
        "ncbi_id": "5265",
        "uniprot_id": "P01009",
        "cytoband": "14q32.13",
        "chromosome": "14",
        "start": 94843084,
        "end": 94857030,
        "strand": "-",
        "function": "alpha-1 antitrypsin, a secreted serine-protease inhibitor",
        "biochemical_axis": "neutrophil elastase restraint, protease-antiprotease balance, liver polymerization stress, and lung inflammation biology",
        "bundle_focus": "protease inhibitor deficiency review",
        "conditions": ["Alpha-1 antitrypsin deficiency context", "COPD/emphysema risk biology", "Liver polymerization and secretion stress"],
        "visual_key": "hematology",
        "variants": [
            {"variant": "rs28929474", "display_name": "rs28929474 (SERPINA1 Pi*Z)", "common_name": "SERPINA1 Pi*Z deficiency marker"},
            {"variant": "rs17580", "display_name": "rs17580 (SERPINA1 Pi*S)", "common_name": "SERPINA1 Pi*S deficiency marker"},
        ],
    },
    {
        "gene_name": "G6PD",
        "ensembl_id": "ENSG00000160211",
        "ncbi_id": "2539",
        "uniprot_id": "P11413",
        "cytoband": "Xq28",
        "chromosome": "X",
        "start": 153759606,
        "end": 153775787,
        "strand": "-",
        "function": "glucose-6-phosphate dehydrogenase, the rate-limiting pentose phosphate pathway enzyme",
        "biochemical_axis": "NADPH production, red-cell oxidative stress defense, glutathione recycling, and hemolysis susceptibility",
        "bundle_focus": "red-cell oxidative metabolism",
        "conditions": ["G6PD deficiency review context", "Drug/food-triggered hemolysis susceptibility", "Pentose phosphate pathway and NADPH biology"],
        "visual_key": "hematology",
        "variants": [
            {"variant": "rs1050828", "display_name": "rs1050828 (G6PD A- marker)", "common_name": "G6PD A- deficiency marker"},
            {"variant": "G6PD deficiency class model", "common_name": "G6PD enzyme-deficiency model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "CYP2D6",
        "ensembl_id": "ENSG00000100197",
        "ncbi_id": "1565",
        "uniprot_id": "P10635",
        "cytoband": "22q13.2",
        "chromosome": "22",
        "start": 42522501,
        "end": 42526908,
        "strand": "-",
        "function": "cytochrome P450 2D6, a polymorphic drug-metabolizing monooxygenase",
        "biochemical_axis": "phase I drug oxidation, copy-number and star-allele pharmacogenetics, antidepressant/opioid/beta-blocker metabolism, and poor/ultrarapid metabolizer biology",
        "bundle_focus": "pharmacogenomic drug metabolism",
        "conditions": ["CYP2D6 pharmacogenomics context", "Drug exposure and adverse-response biology", "Star-allele and copy-number interpretation"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs3892097", "display_name": "rs3892097 (CYP2D6*4 splice marker)", "common_name": "CYP2D6 no-function allele marker"},
            {"variant": "CYP2D6 copy-number/star-allele model", "common_name": "CYP2D6 metabolizer phenotype model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "CYP2C19",
        "ensembl_id": "ENSG00000165841",
        "ncbi_id": "1557",
        "uniprot_id": "P33261",
        "cytoband": "10q23.33",
        "chromosome": "10",
        "start": 96447911,
        "end": 96613017,
        "strand": "+",
        "function": "cytochrome P450 2C19, a polymorphic drug-metabolizing monooxygenase",
        "biochemical_axis": "phase I drug oxidation, proton-pump inhibitor and antiplatelet pharmacogenetics, star-allele activity, and metabolizer phenotype biology",
        "bundle_focus": "pharmacogenomic drug metabolism",
        "conditions": ["CYP2C19 pharmacogenomics context", "Clopidogrel and proton-pump inhibitor response biology", "Star-allele metabolizer interpretation"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs4244285", "display_name": "rs4244285 (CYP2C19*2)", "common_name": "CYP2C19 no-function allele marker"},
            {"variant": "rs12248560", "display_name": "rs12248560 (CYP2C19*17)", "common_name": "CYP2C19 increased-function allele marker"},
        ],
    },
    {
        "gene_name": "CYP2C9",
        "ensembl_id": "ENSG00000138109",
        "ncbi_id": "1559",
        "uniprot_id": "P11712",
        "cytoband": "10q23.33",
        "chromosome": "10",
        "start": 96698415,
        "end": 96749147,
        "strand": "+",
        "function": "cytochrome P450 2C9, a polymorphic drug-metabolizing monooxygenase",
        "biochemical_axis": "phase I drug oxidation, warfarin/NSAID/phenytoin pharmacogenetics, star-allele activity, and exposure biology",
        "bundle_focus": "pharmacogenomic drug metabolism",
        "conditions": ["CYP2C9 pharmacogenomics context", "Warfarin and phenytoin dose-sensitivity biology", "Star-allele metabolizer interpretation"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs1799853", "display_name": "rs1799853 (CYP2C9*2)", "common_name": "CYP2C9 reduced-function allele marker"},
            {"variant": "rs1057910", "display_name": "rs1057910 (CYP2C9*3)", "common_name": "CYP2C9 reduced-function allele marker"},
        ],
    },
    {
        "gene_name": "SLCO1B1",
        "ensembl_id": "ENSG00000134538",
        "ncbi_id": "10599",
        "uniprot_id": "Q9Y6L6",
        "cytoband": "12p12.1 p12.2",
        "chromosome": "12",
        "start": 21284136,
        "end": 21392180,
        "strand": "+",
        "function": "OATP1B1, a hepatic organic anion uptake transporter",
        "biochemical_axis": "hepatic statin uptake, bile-acid and drug transport, transporter pharmacogenetics, and myopathy-risk biology",
        "bundle_focus": "transporter pharmacogenomics",
        "conditions": ["Statin-associated myopathy context", "Hepatic transporter pharmacogenomics", "Drug disposition and exposure biology"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs4149056", "display_name": "rs4149056 (SLCO1B1 c.521T>C)", "common_name": "SLCO1B1 reduced-transport statin marker"},
        ],
    },
    {
        "gene_name": "VKORC1",
        "ensembl_id": "ENSG00000167397",
        "ncbi_id": "79001",
        "uniprot_id": "Q9BQB6",
        "cytoband": "16p11.2",
        "chromosome": "16",
        "start": 31102163,
        "end": 31107301,
        "strand": "-",
        "function": "vitamin K epoxide reductase complex subunit 1, the warfarin target",
        "biochemical_axis": "vitamin K recycling, gamma-carboxylation support, warfarin sensitivity, and anticoagulant pharmacogenomics",
        "bundle_focus": "warfarin pharmacogenomics",
        "conditions": ["Warfarin dose-sensitivity context", "Vitamin K cycle biology", "Anticoagulation pharmacogenetics"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs9923231", "display_name": "rs9923231 (VKORC1 -1639G>A)", "common_name": "VKORC1 warfarin sensitivity marker"},
        ],
    },
    {
        "gene_name": "DPYD",
        "ensembl_id": "ENSG00000188641",
        "ncbi_id": "1806",
        "uniprot_id": "Q12882",
        "cytoband": "1p21.3",
        "chromosome": "1",
        "start": 97543299,
        "end": 98386605,
        "strand": "-",
        "function": "dihydropyrimidine dehydrogenase, the rate-limiting enzyme in fluoropyrimidine catabolism",
        "biochemical_axis": "pyrimidine catabolism, 5-fluorouracil/capecitabine clearance, toxicity susceptibility, and pharmacogenomic dosing biology",
        "bundle_focus": "fluoropyrimidine pharmacogenomics",
        "conditions": ["DPYD fluoropyrimidine toxicity context", "Pyrimidine catabolism", "Oncology pharmacogenomics"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs3918290", "display_name": "rs3918290 (DPYD*2A)", "common_name": "DPYD no-function fluoropyrimidine toxicity marker"},
            {"variant": "rs67376798", "display_name": "rs67376798 (DPYD D949V)", "common_name": "DPYD reduced-function toxicity marker"},
        ],
    },
    {
        "gene_name": "TPMT",
        "ensembl_id": "ENSG00000137364",
        "ncbi_id": "7172",
        "uniprot_id": "P51580",
        "cytoband": "6p22.3",
        "chromosome": "6",
        "start": 18128542,
        "end": 18155305,
        "strand": "-",
        "function": "thiopurine S-methyltransferase, a thiopurine inactivation enzyme",
        "biochemical_axis": "thiopurine methylation, active nucleotide accumulation, myelosuppression risk, and immunosuppressant pharmacogenomics",
        "bundle_focus": "thiopurine pharmacogenomics",
        "conditions": ["TPMT thiopurine toxicity context", "Mercaptopurine/azathioprine dosing biology", "Methyltransferase activity variation"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs1800460", "display_name": "rs1800460 (TPMT*3B)", "common_name": "TPMT reduced-function allele marker"},
            {"variant": "rs1142345", "display_name": "rs1142345 (TPMT*3C)", "common_name": "TPMT reduced-function allele marker"},
        ],
    },
    {
        "gene_name": "UGT1A1",
        "ensembl_id": "ENSG00000242366",
        "ncbi_id": "54658",
        "uniprot_id": "P22309",
        "cytoband": "2q37.1",
        "chromosome": "2",
        "start": 234668894,
        "end": 234681945,
        "strand": "+",
        "function": "bilirubin UDP-glucuronosyltransferase 1A1",
        "biochemical_axis": "bilirubin glucuronidation, irinotecan SN-38 clearance, promoter-repeat regulation, and glucuronidation pharmacogenomics",
        "bundle_focus": "glucuronidation pharmacogenomics",
        "conditions": ["Gilbert syndrome context", "Irinotecan toxicity pharmacogenomics", "Bilirubin conjugation biology"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "UGT1A1*28 promoter-repeat model", "common_name": "UGT1A1 reduced-expression promoter model", "is_assayable_in_snp_vcf": False},
            {"variant": "rs4148323", "display_name": "rs4148323 (UGT1A1*6)", "common_name": "UGT1A1 reduced-function allele marker"},
        ],
    },
    {
        "gene_name": "IFNG",
        "ensembl_id": "ENSG00000111537",
        "ncbi_id": "3458",
        "uniprot_id": "P01579",
        "cytoband": "12q15",
        "chromosome": "12",
        "start": 68548548,
        "end": 68553527,
        "strand": "-",
        "function": "interferon gamma, a type II interferon cytokine",
        "biochemical_axis": "Th1 cytokine signaling, macrophage activation, antigen presentation, JAK-STAT1 activation, and host-defense inflammation",
        "bundle_focus": "cytokine signaling",
        "conditions": ["Th1 immune activation context", "Infectious disease and autoimmunity association research", "Macrophage activation and antigen presentation"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "rs2430561", "display_name": "rs2430561 (IFNG +874 T>A)", "common_name": "IFNG expression association marker"},
        ],
    },
    {
        "gene_name": "TNF",
        "ensembl_id": "ENSG00000232810",
        "ncbi_id": "7124",
        "uniprot_id": "P01375",
        "cytoband": "6p21.33",
        "chromosome": "6",
        "start": 31543344,
        "end": 31546113,
        "strand": "+",
        "function": "tumor necrosis factor, a central inflammatory cytokine",
        "biochemical_axis": "TNF receptor signaling, NF-kappaB activation, apoptosis/necroptosis crosstalk, inflammation, and anti-TNF pharmacology",
        "bundle_focus": "inflammatory cytokine signaling",
        "conditions": ["Autoimmune and inflammatory disease context", "TNF-blocker pharmacology", "NF-kappaB and cell-death signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "rs1800629", "display_name": "rs1800629 (TNF -308G>A)", "common_name": "TNF promoter association marker"},
        ],
    },
    {
        "gene_name": "IL6",
        "ensembl_id": "ENSG00000136244",
        "ncbi_id": "3569",
        "uniprot_id": "P05231",
        "cytoband": "7p15.3",
        "chromosome": "7",
        "start": 22765503,
        "end": 22771621,
        "strand": "+",
        "function": "interleukin-6, a pleiotropic inflammatory cytokine",
        "biochemical_axis": "IL-6 receptor/gp130 signaling, JAK-STAT3 activation, acute-phase response, inflammation, and metabolic immune crosstalk",
        "bundle_focus": "inflammatory cytokine signaling",
        "conditions": ["Inflammation and acute-phase response context", "Autoimmune and infection association research", "JAK-STAT3 cytokine signaling"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "rs1800795", "display_name": "rs1800795 (IL6 -174G>C)", "common_name": "IL6 promoter association marker"},
        ],
    },
    {
        "gene_name": "CRP",
        "ensembl_id": "ENSG00000132693",
        "ncbi_id": "1401",
        "uniprot_id": "P02741",
        "cytoband": "1q23.2",
        "chromosome": "1",
        "start": 159682079,
        "end": 159684379,
        "strand": "-",
        "function": "C-reactive protein, a pentraxin acute-phase protein",
        "biochemical_axis": "acute-phase inflammation, complement engagement, pentraxin biology, cardiometabolic biomarker context, and hepatic IL-6 response",
        "bundle_focus": "acute-phase inflammatory biomarker review",
        "conditions": ["Inflammation biomarker context", "Cardiometabolic association research", "Hepatic acute-phase response"],
        "visual_key": "hematology",
        "variants": [
            {"variant": "rs1205", "display_name": "rs1205 (CRP 3'UTR marker)", "common_name": "CRP level association marker"},
        ],
    },
    {
        "gene_name": "VDR",
        "ensembl_id": "ENSG00000111424",
        "ncbi_id": "7421",
        "uniprot_id": "P11473",
        "cytoband": "12q13.11",
        "chromosome": "12",
        "start": 48235320,
        "end": 48336831,
        "strand": "-",
        "function": "the vitamin D receptor, a ligand-regulated nuclear receptor",
        "biochemical_axis": "vitamin D hormone signaling, calcium/phosphate homeostasis, immune modulation, bone biology, and nuclear receptor transcription",
        "bundle_focus": "vitamin D receptor signaling",
        "conditions": ["Bone mineral and calcium homeostasis context", "Immune and infection association research", "Nuclear receptor transcriptional control"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "rs2228570", "display_name": "rs2228570 (VDR FokI)", "common_name": "VDR translation-start marker"},
            {"variant": "rs731236", "display_name": "rs731236 (VDR TaqI)", "common_name": "VDR association marker"},
        ],
    },
    {
        "gene_name": "FTO",
        "ensembl_id": "ENSG00000140718",
        "ncbi_id": "79068",
        "uniprot_id": "Q9C0B1",
        "cytoband": "16q12.2",
        "chromosome": "16",
        "start": 53737875,
        "end": 54155853,
        "strand": "+",
        "function": "FTO, an Fe(II)/2-oxoglutarate-dependent RNA demethylase",
        "biochemical_axis": "RNA demethylation, appetite and adiposity association biology, m6A regulation, energy balance, and IRX3/IRX5 regulatory context",
        "bundle_focus": "adiposity and RNA-modification review",
        "conditions": ["Obesity and BMI association context", "Energy balance and appetite biology", "RNA methylation regulation"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "rs9939609", "display_name": "rs9939609 (FTO obesity association marker)", "common_name": "FTO adiposity association marker"},
        ],
    },
    {
        "gene_name": "LEP",
        "ensembl_id": "ENSG00000174697",
        "ncbi_id": "3952",
        "uniprot_id": "P41159",
        "cytoband": "7q32.1",
        "chromosome": "7",
        "start": 127881337,
        "end": 127897681,
        "strand": "+",
        "function": "leptin, an adipocyte-derived endocrine hormone",
        "biochemical_axis": "adipose endocrine signaling, hypothalamic appetite regulation, energy expenditure, puberty/fertility signaling, and inflammatory crosstalk",
        "bundle_focus": "leptin endocrine signaling",
        "conditions": ["Energy balance and obesity context", "Leptin deficiency review context", "Adipokine and inflammatory signaling"],
        "visual_key": "lipid_metabolism",
        "variants": [
            {"variant": "rs7799039", "display_name": "rs7799039 (LEP promoter marker)", "common_name": "LEP expression association marker"},
            {"variant": "LEP deficiency model", "common_name": "leptin deficiency endocrine model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "MC4R",
        "ensembl_id": "ENSG00000166603",
        "ncbi_id": "4160",
        "uniprot_id": "P32245",
        "cytoband": "18q21.32",
        "chromosome": "18",
        "start": 58038564,
        "end": 58040001,
        "strand": "-",
        "function": "melanocortin 4 receptor, a hypothalamic G protein-coupled receptor",
        "biochemical_axis": "melanocortin GPCR signaling, appetite suppression, energy balance, obesity biology, and receptor trafficking",
        "bundle_focus": "melanocortin receptor signaling",
        "conditions": ["Monogenic obesity review context", "Appetite and energy-balance biology", "GPCR signaling and receptor trafficking"],
        "visual_key": "receptor_signaling",
        "variants": [
            {"variant": "rs17782313", "display_name": "rs17782313 (near MC4R obesity marker)", "common_name": "MC4R-region adiposity association marker"},
            {"variant": "MC4R loss-of-function model", "common_name": "MC4R monogenic obesity model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "HNF1A",
        "ensembl_id": "ENSG00000135100",
        "ncbi_id": "6927",
        "uniprot_id": "P20823",
        "cytoband": "12q24.31",
        "chromosome": "12",
        "start": 121416346,
        "end": 121440315,
        "strand": "+",
        "function": "hepatocyte nuclear factor 1-alpha, a homeobox transcription factor",
        "biochemical_axis": "pancreatic beta-cell transcription, hepatic gene regulation, MODY biology, glycemic control, and CRP/lipid trait associations",
        "bundle_focus": "diabetes transcription-factor review",
        "conditions": ["MODY3 review context", "Beta-cell function and glycemic trait biology", "Hepatic transcriptional regulation"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "HNF1A MODY3 loss-of-function model", "common_name": "HNF1A beta-cell transcription loss model", "is_assayable_in_snp_vcf": False},
            {"variant": "rs1169288", "display_name": "rs1169288 (HNF1A I27L)", "common_name": "HNF1A cardiometabolic association marker"},
        ],
    },
    {
        "gene_name": "GCK",
        "ensembl_id": "ENSG00000106633",
        "ncbi_id": "2645",
        "uniprot_id": "Q12851",
        "cytoband": "7p13",
        "chromosome": "7",
        "start": 44183872,
        "end": 44237769,
        "strand": "-",
        "function": "glucokinase, the glucose sensor hexokinase in beta cells and liver",
        "biochemical_axis": "glucose phosphorylation, beta-cell glucose sensing, hepatic glycogen synthesis, MODY biology, and congenital hyperinsulinism context",
        "bundle_focus": "glucose-sensing enzyme review",
        "conditions": ["GCK-MODY review context", "Beta-cell glucose sensing", "Congenital hyperinsulinism and glycemic set-point biology"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "GCK MODY2 loss-of-function model", "common_name": "GCK glucose-sensing set-point model", "is_assayable_in_snp_vcf": False},
            {"variant": "GCK activating hyperinsulinism model", "common_name": "GCK activating enzyme model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "KCNJ11",
        "ensembl_id": "ENSG00000187486",
        "ncbi_id": "3767",
        "uniprot_id": "Q14654",
        "cytoband": "11p15.1",
        "chromosome": "11",
        "start": 17407406,
        "end": 17410878,
        "strand": "-",
        "function": "Kir6.2, the pore-forming subunit of the ATP-sensitive potassium channel",
        "biochemical_axis": "ATP-sensitive potassium channel gating, beta-cell insulin secretion, neonatal diabetes biology, and sulfonylurea pharmacology",
        "bundle_focus": "KATP channel and insulin-secretion review",
        "conditions": ["Neonatal diabetes review context", "Type 2 diabetes association research", "Sulfonylurea response and beta-cell electrophysiology"],
        "visual_key": "ion_channel",
        "variants": [
            {"variant": "rs5219", "display_name": "rs5219 (KCNJ11 E23K)", "common_name": "KCNJ11 diabetes association marker"},
            {"variant": "KCNJ11 neonatal-diabetes model", "common_name": "KCNJ11 KATP gain-of-function model", "is_assayable_in_snp_vcf": False},
        ],
    },
    {
        "gene_name": "TTR",
        "ensembl_id": "ENSG00000118271",
        "ncbi_id": "7276",
        "uniprot_id": "P02766",
        "cytoband": "18q12.1",
        "chromosome": "18",
        "start": 29171689,
        "end": 29178974,
        "strand": "+",
        "function": "transthyretin, a tetrameric transporter of thyroxine and retinol-binding protein",
        "biochemical_axis": "transthyretin tetramer stability, amyloid fibril formation, neuropathy/cardiomyopathy context, and protein-stabilizer therapy biology",
        "bundle_focus": "amyloid protein stability review",
        "conditions": ["Hereditary transthyretin amyloidosis context", "Amyloid cardiomyopathy and neuropathy biology", "Tetramer stabilization pharmacology"],
        "visual_key": "metabolic_pathway",
        "variants": [
            {"variant": "TTR Val50Met / V30M model", "common_name": "TTR amyloidogenic variant model", "is_assayable_in_snp_vcf": False},
            {"variant": "TTR tetramer-destabilization model", "common_name": "TTR amyloid stability model", "is_assayable_in_snp_vcf": False},
        ],
    },
]


TOPIC_BIOLOGY_DEFAULTS: dict[str, dict[str, Any]] = {
    "allergy_inflammation": {
        "label": "allergy and inflammation",
        "axis": "IgE receptor biology, mast-cell and eosinophil activation, cytokine and chemokine signaling, NF-kappaB control, and JAK-STAT inflammatory regulation",
        "conditions": [
            "Allergy, asthma, atopy, and inflammatory disease context",
            "Mast-cell, eosinophil, cytokine, and chemokine pathway biology",
            "Immune signaling, barrier inflammation, and therapeutic target review",
        ],
        "visual_key": "allergic_inflammation",
        "variant_label": "immune-response",
    },
    "pigmentation_hair": {
        "label": "hair pigmentation and graying",
        "axis": "melanocyte development, melanin synthesis and trafficking, hair-follicle pigmentation, melanocyte stem-cell maintenance, and graying biology",
        "conditions": [
            "Hair pigmentation, skin pigmentation, and graying research context",
            "Melanocyte development, survival, and melanosome biology",
            "Pigmentary trait, albinism, melanoma-lineage, and follicle biology",
        ],
        "visual_key": "pigmentation_hair",
        "variant_label": "pigmentation",
    },
    "aging_maintenance": {
        "label": "aging and cellular maintenance",
        "axis": "telomere and genome maintenance, stress resilience, NAD-dependent signaling, proteostasis, oxidative response, and senescence biology",
        "conditions": [
            "Aging, longevity, and cellular maintenance research context",
            "Telomere biology, DNA repair, oxidative stress, and senescence pathways",
            "Progeroid, degenerative, cancer-risk, and stress-resilience review",
        ],
        "visual_key": "aging_telomere",
        "variant_label": "aging-maintenance",
    },
    "neurobiology": {
        "label": "neurological and neurodegenerative biology",
        "axis": "synaptic signaling, neurotransmission, protein aggregation, neuronal survival, neurodevelopment, and neurodegeneration biology",
        "conditions": [
            "Neurological, neurodevelopmental, and neurodegenerative disease context",
            "Synaptic signaling, excitability, neurotransmitter, and protein-homeostasis biology",
            "Cognitive, movement-disorder, epilepsy, and psychiatric association review",
        ],
        "visual_key": "neurodegeneration",
        "variant_label": "neurobiology",
    },
    "digestion_gut": {
        "label": "digestion and gut barrier biology",
        "axis": "intestinal epithelial barrier function, digestion and nutrient transport, mucus and microbial interface biology, bile-acid handling, and gut immune signaling",
        "conditions": [
            "Digestion, malabsorption, gut barrier, and microbiome interaction context",
            "Inflammatory bowel disease and mucosal immune pathway biology",
            "Nutrient transport, digestive enzyme, and epithelial repair review",
        ],
        "visual_key": "gut_barrier",
        "variant_label": "gut-digestion",
    },
}


def _build_topic_biology_gene_config(row: tuple[str, str, str, str, str, str, int, int, str, str, str]) -> dict[str, Any]:
    """Expand compact topic biology rows into the full biochemistry config schema."""
    (
        gene_name,
        ensembl_id,
        ncbi_id,
        uniprot_id,
        cytoband,
        chromosome,
        start,
        end,
        strand,
        topic,
        role,
    ) = row
    topic_defaults = TOPIC_BIOLOGY_DEFAULTS[topic]
    topic_label = str(topic_defaults["label"])
    axis = str(topic_defaults["axis"])
    manifest_start = max(1, int(start) - 100000)
    manifest_end = int(end) + 100000
    return _build_extended_biochemistry_gene_config(
        {
            "gene_name": gene_name,
            "ensembl_id": ensembl_id,
            "ncbi_id": ncbi_id,
            "uniprot_id": uniprot_id,
            "cytoband": cytoband,
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "strand": strand,
            "manifest_filter_region": f"{chromosome}:{manifest_start}-{manifest_end}",
            "function": role,
            "biochemical_axis": axis,
            "bundle_focus": topic_label,
            "conditions": list(topic_defaults["conditions"]),
            "visual_key": str(topic_defaults["visual_key"]),
            "figure_focus": f"{gene_name}: {topic_label} via {axis}",
            "gene_summary": (
                f"{gene_name} encodes {role}. This topic bundle emphasizes {topic_label}, "
                f"with interpretation centered on {axis}."
            ),
            "clinical_context": (
                f"The local {gene_name} bundle is oriented toward {topic_label}. "
                "Use it for pathway triage, phenotype review, and candidate-gene context; "
                "external variant classification is required before clinical interpretation."
            ),
            "variant_effect_overview": [
                f"{gene_name} variants can alter, tag, or contextualize {axis}.",
                "Effect direction depends on exact allele class, dosage, tissue, ancestry, phenotype definition, and assay context.",
            ],
            "variants": [
                {
                    "variant": f"{gene_name} {topic_defaults['variant_label']} variation model",
                    "common_name": f"{gene_name} {topic_label} pathway model",
                    "is_assayable_in_snp_vcf": False,
                },
                {
                    "variant": f"{gene_name} regulatory expression context model",
                    "common_name": f"{gene_name} expression and methylation-context model",
                    "is_assayable_in_snp_vcf": False,
                },
            ],
            "concrete_variant_prediction": (
                f"The variant observed in this sample suggests a {gene_name} {topic_label} thesis: "
                f"the individual may carry context relevant to {axis}. "
                "Treat the result as candidate pathway evidence until exact allele, zygosity, phenotype, and external classification are reviewed."
            ),
        }
    )


TOPIC_BIOLOGY_GENE_ROWS: list[tuple[str, str, str, str, str, str, int, int, str, str, str]] = [
    ("FCER1A", "ENSG00000179639", "2205", "P12319", "1q23.2", "1", 159259504, 159278014, "+", "allergy_inflammation", "the high-affinity IgE receptor alpha chain that binds IgE on mast cells and basophils"),
    ("FCER1G", "ENSG00000158869", "2207", "P30273", "1q23.3", "1", 161185024, 161190489, "+", "allergy_inflammation", "the Fc receptor gamma signaling chain used by IgE, IgG, and innate immune receptor complexes"),
    ("FCER2", "ENSG00000104921", "2208", "P06734", "19p13.2", "19", 7753644, 7767032, "-", "allergy_inflammation", "CD23, the low-affinity IgE receptor that regulates IgE biology and B-cell/allergic responses"),
    ("MS4A2", "ENSG00000149534", "2206", "Q01362", "11q12.1", "11", 59855734, 59863444, "+", "allergy_inflammation", "the beta chain of the high-affinity IgE receptor complex"),
    ("TPSAB1", "ENSG00000172236", "7177", "Q15661", "16p13.3", "16", 1290697, 1292555, "+", "allergy_inflammation", "alpha/beta tryptase, a mast-cell granule protease linked to mediator release biology"),
    ("TPSB2", "ENSG00000197253", "64499", "P20231", "16p13.3", "16", 1277272, 1280214, "-", "allergy_inflammation", "beta-2 tryptase, a mast-cell serine protease relevant to allergic inflammation"),
    ("CMA1", "ENSG00000092009", "1215", "P23946", "14q12", "14", 24974559, 24977471, "-", "allergy_inflammation", "mast-cell chymase, a granule protease involved in tissue remodeling and inflammatory mediator processing"),
    ("CPA3", "ENSG00000163751", "1359", "P15088", "3q24", "3", 148583043, 148614983, "+", "allergy_inflammation", "mast-cell carboxypeptidase A3, a granule enzyme used as a mast-cell activation and differentiation marker"),
    ("KIT", "ENSG00000157404", "3815", "P10721", "4q12", "4", 55524085, 55606881, "+", "allergy_inflammation", "the KIT receptor tyrosine kinase that supports mast-cell development, survival, and activation thresholds"),
    ("IL5", "ENSG00000113525", "3567", "P05113", "5q31.1", "5", 131877136, 131892530, "-", "allergy_inflammation", "interleukin-5, a type 2 cytokine central to eosinophil maturation and survival"),
    ("IL5RA", "ENSG00000091181", "3568", "Q01344", "3p26.2", "3", 3111233, 3168297, "-", "allergy_inflammation", "the interleukin-5 receptor alpha chain that mediates eosinophil-targeted cytokine signaling"),
    ("CCL11", "ENSG00000172156", "6356", "P51671", "17q12", "17", 32612687, 32615353, "+", "allergy_inflammation", "eotaxin-1, a chemokine that recruits CCR3-positive eosinophils"),
    ("CCR3", "ENSG00000183625", "1232", "P51677", "3p21.31", "3", 46205096, 46308197, "+", "allergy_inflammation", "the eosinophil-biased chemokine receptor for eotaxin-family ligands"),
    ("CCL17", "ENSG00000102970", "6361", "Q92583", "16q21", "16", 57438679, 57449974, "+", "allergy_inflammation", "TARC/CCL17, a chemokine that recruits CCR4-positive type 2 immune cells"),
    ("CCL22", "ENSG00000102962", "6367", "O00626", "16q13 q21", "16", 57392684, 57400102, "+", "allergy_inflammation", "MDC/CCL22, a CCR4 ligand involved in Th2 and regulatory T-cell trafficking"),
    ("CCR4", "ENSG00000183813", "1233", "Q9UK39", "3p22.3", "3", 32993066, 32997841, "+", "allergy_inflammation", "a chemokine receptor used by Th2, Treg, and skin-homing inflammatory cells"),
    ("IL8", "ENSG00000169429", "3576", "P10145", "4q13.3", "4", 74606223, 74609433, "+", "allergy_inflammation", "CXCL8/IL-8, a neutrophil chemoattractant chemokine in acute inflammation"),
    ("CXCR2", "ENSG00000180871", "3579", "P25025", "2q35", "2", 218990012, 219001976, "+", "allergy_inflammation", "a CXCL8-family chemokine receptor that drives neutrophil recruitment and inflammatory trafficking"),
    ("NFKB1", "ENSG00000109320", "4790", "P19838", "4q24", "4", 103422486, 103538459, "+", "allergy_inflammation", "the NF-kappaB p105/p50 transcription factor subunit central to inflammatory gene control"),
    ("NFKBIA", "ENSG00000100906", "4792", "P25963", "14q13.2", "14", 35870717, 35873955, "-", "allergy_inflammation", "IkappaB-alpha, an inhibitor that restrains NF-kappaB inflammatory activation"),
    ("RELA", "ENSG00000173039", "5970", "Q04206", "11q13.1", "11", 65421067, 65430565, "-", "allergy_inflammation", "the NF-kappaB p65 transcriptional activation subunit"),
    ("JAK1", "ENSG00000162434", "3716", "P23458", "1p31.3", "1", 65298912, 65432187, "-", "allergy_inflammation", "Janus kinase 1, a core cytokine receptor signaling kinase"),
    ("JAK2", "ENSG00000096968", "3717", "O60674", "9p24.1", "9", 4985033, 5128183, "+", "allergy_inflammation", "Janus kinase 2, a cytokine and hematopoietic growth-factor signaling kinase"),
    ("TYK2", "ENSG00000105397", "7297", "P29597", "19p13.2", "19", 10461209, 10491352, "-", "allergy_inflammation", "tyrosine kinase 2, a JAK-family kinase used by interferon, IL-12, and IL-23 signaling"),
    ("STAT3", "ENSG00000168610", "6774", "P40763", "17q21.2", "17", 40465342, 40540586, "-", "allergy_inflammation", "STAT3, a cytokine-responsive transcription factor linking IL-6/IL-23 signaling to inflammatory differentiation"),
    ("MC1R", "ENSG00000258839", "4157", "Q01726", "16q24.3", "16", 89978527, 89987385, "+", "pigmentation_hair", "the melanocortin 1 receptor that switches melanocytes toward eumelanin production"),
    ("TYR", "ENSG00000077498", "7299", "P14679", "11q14.3", "11", 88910620, 89028927, "+", "pigmentation_hair", "tyrosinase, the copper enzyme that catalyzes key early steps in melanin synthesis"),
    ("TYRP1", "ENSG00000107165", "7306", "P17643", "9p23", "9", 12685439, 12710290, "+", "pigmentation_hair", "tyrosinase-related protein 1, a melanosomal enzyme involved in eumelanin maturation"),
    ("DCT", "ENSG00000080166", "1638", "P40126", "13q32.1", "13", 95089558, 95131936, "-", "pigmentation_hair", "dopachrome tautomerase, a melanin-pathway enzyme active in melanocytes"),
    ("OCA2", "ENSG00000104044", "4948", "Q04671", "15q12 q13.1", "15", 28000021, 28344504, "-", "pigmentation_hair", "a melanosomal membrane protein affecting melanosome pH and pigment production"),
    ("SLC45A2", "ENSG00000164175", "51151", "Q9UMX9", "5p13.2", "5", 33944721, 33984835, "-", "pigmentation_hair", "a melanosomal solute carrier required for normal pigmentation"),
    ("SLC24A5", "ENSG00000188467", "283652", "Q71RS6", "15q21.1", "15", 48413169, 48434869, "+", "pigmentation_hair", "a melanosomal sodium/potassium/calcium exchanger shaping pigmentation traits"),
    ("IRF4", "ENSG00000137265", "3662", "Q15306", "6p25.3", "6", 391739, 411447, "+", "pigmentation_hair", "an interferon regulatory factor with melanocyte regulatory and immune differentiation roles"),
    ("MITF", "ENSG00000187098", "4286", "O75030", "3p13 p14.1", "3", 69788586, 70017488, "+", "pigmentation_hair", "the master melanocyte lineage transcription factor controlling pigment and survival programs"),
    ("PAX3", "ENSG00000135903", "5077", "P23760", "2q36.1", "2", 223064607, 223163715, "-", "pigmentation_hair", "a paired-box transcription factor involved in neural crest and melanocyte development"),
    ("SOX10", "ENSG00000100146", "6663", "P56693", "22q13.1", "22", 38366693, 38383429, "-", "pigmentation_hair", "a neural-crest transcription factor required for melanocyte and glial lineage development"),
    ("KITLG", "ENSG00000049130", "4254", "P21583", "12q21.32", "12", 88886570, 88974628, "-", "pigmentation_hair", "stem cell factor, the KIT ligand that supports melanocyte development and hair-follicle pigment cells"),
    ("EDNRB", "ENSG00000136160", "1910", "P24530", "13q22.3", "13", 78469616, 78493903, "-", "pigmentation_hair", "endothelin receptor type B, a GPCR needed for melanocyte and enteric neural crest development"),
    ("EDN3", "ENSG00000124205", "1908", "P14138", "20q13.32", "20", 57875482, 57901047, "+", "pigmentation_hair", "endothelin-3, a neural-crest ligand supporting melanocyte and enteric neuron development"),
    ("ASIP", "ENSG00000101440", "434", "P42127", "20q11.22", "20", 32782375, 32857150, "+", "pigmentation_hair", "agouti signaling protein, an MC1R antagonist that shifts pigment toward pheomelanin"),
    ("BCL2", "ENSG00000171791", "596", "P10415", "18q21.33", "18", 60790579, 60987361, "-", "pigmentation_hair", "an anti-apoptotic mitochondrial protein relevant to melanocyte survival and stem-cell maintenance"),
    ("PMEL", "ENSG00000185664", "6490", "P40967", "12q13.2", "12", 56347889, 56367101, "-", "pigmentation_hair", "a premelanosome matrix protein needed for melanosome structure"),
    ("GPR143", "ENSG00000101850", "4935", "P51810", "Xp22.2", "X", 9693386, 9754337, "-", "pigmentation_hair", "an intracellular melanosomal GPCR implicated in ocular albinism and melanosome biology"),
    ("MLANA", "ENSG00000120215", "2315", "Q16655", "9p24.1", "9", 5890802, 5910606, "+", "pigmentation_hair", "melan-A, a melanocyte-lineage protein involved in melanosome protein processing"),
    ("SLC24A4", "ENSG00000140090", "123041", "Q8NFF2", "14q32.12", "14", 92788925, 92962596, "+", "pigmentation_hair", "a sodium/potassium/calcium exchanger associated with pigmentation and tooth/hair-related traits"),
    ("SIRT1", "ENSG00000096717", "23411", "Q96EB6", "10q21.3", "10", 69644427, 69678147, "+", "aging_maintenance", "NAD-dependent deacetylase SIRT1, a stress-response and metabolic regulator"),
    ("SIRT2", "ENSG00000068903", "22933", "Q8IXJ6", "19q13.2", "19", 39369197, 39390502, "-", "aging_maintenance", "NAD-dependent deacetylase SIRT2, a cytosolic and nuclear regulator of microtubules, metabolism, and stress responses"),
    ("SIRT3", "ENSG00000142082", "23410", "Q9NTG7", "11p15.5", "11", 215458, 236931, "-", "aging_maintenance", "mitochondrial NAD-dependent deacetylase SIRT3, a regulator of oxidative metabolism and stress resilience"),
    ("SIRT7", "ENSG00000187531", "51547", "Q9NRC8", "17q25.3", "17", 79869815, 79879199, "-", "aging_maintenance", "nucleolar sirtuin SIRT7, a chromatin, ribosome-biogenesis, and stress-response regulator"),
    ("KL", "ENSG00000133116", "9365", "Q9UEF7", "13q13.1", "13", 33590207, 33640282, "+", "aging_maintenance", "klotho, a membrane and secreted endocrine factor linked to phosphate, FGF23, and aging biology"),
    ("WRN", "ENSG00000165392", "7486", "Q14191", "8p12", "8", 30891317, 31031285, "+", "aging_maintenance", "the Werner syndrome RecQ helicase with exonuclease activity in genome maintenance"),
    ("LMNA", "ENSG00000160789", "4000", "P02545", "1q22", "1", 156052364, 156109880, "+", "aging_maintenance", "lamin A/C, a nuclear lamina scaffold protein involved in genome organization and progeroid disease"),
    ("TERF1", "ENSG00000147601", "7013", "P54274", "8q21.11", "8", 73921099, 73960357, "+", "aging_maintenance", "telomeric repeat-binding factor 1, a shelterin component regulating telomere length and replication"),
    ("POT1", "ENSG00000128513", "25913", "Q9NUX5", "7q31.33", "7", 124462440, 124570037, "-", "aging_maintenance", "protection of telomeres 1, a shelterin single-stranded telomeric DNA-binding protein"),
    ("RTEL1", "ENSG00000258366", "51750", "Q9NZ71", "20q13.33", "20", 62289163, 62328416, "+", "aging_maintenance", "a helicase that resolves telomere and genome replication structures"),
    ("PARN", "ENSG00000140694", "5073", "O95453", "16p13.12", "16", 14529558, 14726585, "-", "aging_maintenance", "poly(A)-specific ribonuclease, an RNA-processing factor linked to telomerase RNA maturation"),
    ("DKC1", "ENSG00000130826", "1736", "O60832", "Xq28", "X", 153991031, 154005964, "+", "aging_maintenance", "dyskerin, an H/ACA ribonucleoprotein and telomerase-complex component"),
    ("TINF2", "ENSG00000092330", "26277", "Q9BSI4", "14q12", "14", 24708849, 24711880, "-", "aging_maintenance", "TIN2, a shelterin adaptor connecting TRF1/TRF2 with telomere protection complexes"),
    ("ACD", "ENSG00000102977", "65057", "Q96AP0", "16q22.1", "16", 67691415, 67694713, "-", "aging_maintenance", "TPP1, a shelterin component that recruits telomerase and protects telomeres"),
    ("PARP1", "ENSG00000143799", "142", "P09874", "1q42.12", "1", 226548392, 226595780, "-", "aging_maintenance", "poly(ADP-ribose) polymerase 1, a DNA damage sensor and chromatin regulator"),
    ("NFE2L2", "ENSG00000116044", "4780", "Q16236", "2q31.2", "2", 178092323, 178257425, "-", "aging_maintenance", "NRF2, a transcription factor controlling antioxidant and detoxification stress responses"),
    ("KEAP1", "ENSG00000079999", "9817", "Q14145", "19p13.2", "19", 10596796, 10614417, "-", "aging_maintenance", "KEAP1, the redox-sensitive adaptor that restrains NRF2 signaling"),
    ("FOXO1", "ENSG00000150907", "2308", "Q12778", "13q14.11", "13", 41129804, 41240734, "-", "aging_maintenance", "FOXO1, a forkhead transcription factor linking insulin signaling, stress resistance, and metabolism"),
    ("FOXO4", "ENSG00000184481", "4303", "P98177", "Xq13.1", "X", 70316047, 70323385, "+", "aging_maintenance", "FOXO4, a forkhead transcription factor linked to stress response and senescence biology"),
    ("CDKN1A", "ENSG00000124762", "1026", "P38936", "6p21.2", "6", 36644305, 36655116, "+", "aging_maintenance", "p21/CIP1, a cyclin-dependent kinase inhibitor induced by p53 and senescence programs"),
    ("APP", "ENSG00000142192", "351", "P05067", "21q21.3", "21", 27252861, 27543446, "-", "neurobiology", "amyloid precursor protein, the source of amyloid-beta peptides in Alzheimer disease biology"),
    ("PSEN1", "ENSG00000080815", "5663", "P49768", "14q24.2", "14", 73603126, 73690399, "+", "neurobiology", "presenilin-1, the catalytic component of gamma-secretase in amyloid precursor processing"),
    ("PSEN2", "ENSG00000143801", "5664", "P49810", "1q42.13", "1", 227057885, 227083806, "+", "neurobiology", "presenilin-2, a gamma-secretase complex component involved in amyloid precursor processing"),
    ("MAPT", "ENSG00000186868", "4137", "P10636", "17q21.31", "17", 43971748, 44105700, "+", "neurobiology", "tau, a microtubule-associated protein central to tauopathy and neuronal cytoskeleton biology"),
    ("SNCA", "ENSG00000145335", "6622", "P37840", "4q22.1", "4", 90645250, 90759466, "-", "neurobiology", "alpha-synuclein, a presynaptic protein involved in synucleinopathy and vesicle biology"),
    ("LRRK2", "ENSG00000188906", "120892", "Q5S007", "12q12", "12", 40590546, 40763087, "+", "neurobiology", "leucine-rich repeat kinase 2, a kinase/GTPase involved in vesicle, lysosome, and Parkinson disease biology"),
    ("PINK1", "ENSG00000158828", "65018", "Q9BXM7", "1p36.12", "1", 20959948, 20978004, "+", "neurobiology", "PTEN-induced kinase 1, a mitochondrial damage sensor in mitophagy"),
    ("PARK2", "ENSG00000185345", "5071", "O60260", "6q26", "6", 161768452, 163148803, "-", "neurobiology", "parkin/PRKN, an E3 ubiquitin ligase that cooperates with PINK1 in mitophagy"),
    ("GBA", "ENSG00000177628", "2629", "P04062", "1q22", "1", 155204243, 155214490, "-", "neurobiology", "glucocerebrosidase/GBA1, a lysosomal enzyme linking Gaucher and Parkinson disease biology"),
    ("GRIN2A", "ENSG00000183454", "2903", "Q12879", "16p13.2", "16", 9852376, 10276611, "-", "neurobiology", "the GluN2A NMDA receptor subunit involved in excitatory synaptic signaling"),
    ("GRIN2B", "ENSG00000273079", "2904", "Q13224", "12p13.1", "12", 13693165, 14133053, "-", "neurobiology", "the GluN2B NMDA receptor subunit involved in synaptic plasticity and neurodevelopment"),
    ("SLC6A4", "ENSG00000108576", "6532", "P31645", "17q11.2", "17", 28521337, 28563020, "-", "neurobiology", "the serotonin transporter that clears serotonin from synapses"),
    ("MAOA", "ENSG00000189221", "4128", "P21397", "Xp11.3", "X", 43515467, 43606068, "+", "neurobiology", "monoamine oxidase A, a mitochondrial enzyme that degrades serotonin, norepinephrine, and dopamine"),
    ("HTR2A", "ENSG00000102468", "3356", "P28223", "13q14.2", "13", 47405685, 47471169, "-", "neurobiology", "the serotonin 2A receptor, a GPCR involved in cortical signaling and neuropsychiatric pharmacology"),
    ("GABRA1", "ENSG00000022355", "2554", "P14867", "5q34", "5", 161274197, 161326975, "+", "neurobiology", "the GABA-A receptor alpha-1 subunit involved in inhibitory synaptic signaling"),
    ("SHANK3", "ENSG00000251322", "85358", "Q9BYB0", "22q13.33", "22", 51112843, 51171726, "+", "neurobiology", "a postsynaptic density scaffold protein important for excitatory synapse organization"),
    ("MECP2", "ENSG00000169057", "4204", "P51608", "Xq28", "X", 153287024, 153363212, "-", "neurobiology", "methyl-CpG-binding protein 2, an epigenetic regulator central to Rett syndrome biology"),
    ("FMR1", "ENSG00000102081", "2332", "Q06787", "Xq27.3", "X", 146993469, 147032645, "+", "neurobiology", "fragile X messenger ribonucleoprotein 1, an RNA-binding regulator of synaptic protein translation"),
    ("HTT", "ENSG00000197386", "3064", "P42858", "4p16.3", "4", 3076408, 3245676, "+", "neurobiology", "huntingtin, a large scaffold protein whose polyglutamine expansion causes Huntington disease"),
    ("SCN1A", "ENSG00000144285", "6323", "P35498", "2q24.3", "2", 166845670, 166984523, "-", "neurobiology", "the Nav1.1 voltage-gated sodium channel alpha subunit important for inhibitory interneuron excitability"),
    ("FUT2", "ENSG00000176920", "2524", "Q10981", "19q13.33", "19", 49199228, 49209207, "+", "digestion_gut", "alpha(1,2)fucosyltransferase 2, the secretor-status enzyme shaping mucosal glycans and microbiome interaction"),
    ("NOD2", "ENSG00000167207", "64127", "Q9HC29", "16q12.1", "16", 50727514, 50766988, "+", "digestion_gut", "an intracellular muramyl dipeptide sensor involved in Crohn disease and bacterial-response biology"),
    ("ATG16L1", "ENSG00000085978", "55054", "Q676U5", "2q37.1", "2", 234118697, 234204320, "+", "digestion_gut", "an autophagy complex component linked to Paneth-cell and Crohn disease biology"),
    ("CARD9", "ENSG00000187796", "64170", "Q9H257", "9q34.3", "9", 139256355, 139268133, "-", "digestion_gut", "an adaptor for fungal and bacterial innate immune signaling in mucosal defense"),
    ("IL23R", "ENSG00000162594", "149233", "Q5VWK5", "1p31.3", "1", 67632083, 67725662, "+", "digestion_gut", "the interleukin-23 receptor, a Th17/IL-23 pathway receptor important in IBD genetics"),
    ("TNFSF15", "ENSG00000181634", "9966", "O95150", "9q32", "9", 117546915, 117568406, "-", "digestion_gut", "TL1A, a TNF-family cytokine ligand implicated in gut inflammation and fibrosis"),
    ("PNLIP", "ENSG00000175535", "5406", "P16233", "10q25.3", "10", 118305443, 118327367, "+", "digestion_gut", "pancreatic triglyceride lipase, a digestive enzyme required for dietary fat hydrolysis"),
    ("AMY1A", "ENSG00000237763", "276", "P0DUB6", "1p21.1", "1", 104197912, 104207173, "+", "digestion_gut", "salivary alpha-amylase, a starch-digestion enzyme with copy-number and diet-adaptation relevance"),
    ("SI", "ENSG00000090402", "6476", "P14410", "3q26.1", "3", 164696686, 164796283, "-", "digestion_gut", "sucrase-isomaltase, an intestinal brush-border enzyme for sucrose and starch-derived disaccharide digestion"),
    ("ALPI", "ENSG00000163295", "248", "P09923", "2q37.1", "2", 233320833, 233324742, "+", "digestion_gut", "intestinal alkaline phosphatase, a brush-border enzyme involved in barrier and microbial-lipid detoxification biology"),
    ("MUC2", "ENSG00000198788", "4583", "Q02817", "11p15.5", "11", 1074875, 1104419, "+", "digestion_gut", "mucin-2, the major gel-forming mucin of the intestinal mucus barrier"),
    ("TFF3", "ENSG00000160180", "7033", "Q07654", "21q22.3", "21", 43731777, 43735761, "-", "digestion_gut", "trefoil factor 3, a secreted epithelial repair peptide abundant in intestinal goblet cells"),
    ("SLC5A1", "ENSG00000100170", "6523", "P13866", "22q12.3", "22", 32439019, 32509016, "+", "digestion_gut", "SGLT1, the sodium-glucose cotransporter responsible for intestinal glucose and galactose uptake"),
    ("SLC2A2", "ENSG00000163581", "6514", "P11168", "3q26.2", "3", 170714137, 170744539, "-", "digestion_gut", "GLUT2, a glucose transporter in liver, pancreas, kidney, and intestinal nutrient handling"),
    ("SLC10A2", "ENSG00000125255", "6555", "Q12908", "13q33.1", "13", 103696350, 103719196, "-", "digestion_gut", "ASBT, the ileal sodium-bile acid cotransporter required for enterohepatic bile-acid recycling"),
]


BIOCHEMISTRY_GENE_CONFIGS.extend(
    _build_extended_biochemistry_gene_config(config)
    for config in EXTENDED_BIOCHEMISTRY_GENE_CONFIGS
)

BIOCHEMISTRY_GENE_CONFIGS.extend(
    _build_topic_biology_gene_config(row)
    for row in TOPIC_BIOLOGY_GENE_ROWS
)


GENE_DEFINITIONS.extend(
    _build_allergic_inflammation_gene_definition(config)
    for config in ALLERGIC_INFLAMMATION_GENE_CONFIGS
)

GENE_DEFINITIONS.extend(
    _build_biochemistry_gene_definition(config)
    for config in BIOCHEMISTRY_GENE_CONFIGS
)


def _build_promoter_region(meta: dict[str, Any]) -> dict[str, int | str]:
    if meta.get("promoter_review_region"):
        return dict(meta["promoter_review_region"])

    chrom = str(meta["chromosome"])
    start = int(meta["start"])
    end = int(meta["end"])
    if meta["strand"] == "-":
        promoter_start = end + 1
        promoter_end = end + 1000
        definition = (
            "A practical 1 kb upstream window relative to the reverse-strand transcription start, "
            "used to flag promoter-adjacent coverage before the canonical gene interval."
        )
    else:
        promoter_start = max(1, start - 1000)
        promoter_end = start - 1
        definition = (
            "A practical 1 kb upstream window used by this app to flag promoter-adjacent coverage "
            "before the canonical gene interval begins."
        )

    return {
        "label": "Operational promoter review window",
        "start": promoter_start,
        "end": promoter_end,
        "definition": definition,
    }


def _select_relevant_probe_ids(subset_df: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    if meta.get("curated_methylation_probe_ids") is not None:
        return list(meta["curated_methylation_probe_ids"])
    if subset_df.empty or "MAPINFO" not in subset_df.columns or "IlmnID" not in subset_df.columns:
        return []

    tss_coordinate = int(meta["end"] if meta["strand"] == "-" else meta["start"])
    prioritized = subset_df.copy()
    prioritized["distance_to_tss"] = (prioritized["MAPINFO"] - tss_coordinate).abs()

    if "UCSC_RefGene_Group" in prioritized.columns:
        group_text = prioritized["UCSC_RefGene_Group"].fillna("").astype(str)
        tss_mask = group_text.str.contains("TSS|5'UTR|1stExon", case=False, regex=True)
        if tss_mask.any():
            prioritized = prioritized[tss_mask].copy()
            prioritized["distance_to_tss"] = (prioritized["MAPINFO"] - tss_coordinate).abs()

    return (
        prioritized.sort_values(["distance_to_tss", "MAPINFO"])["IlmnID"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(10)
        .tolist()
    )


def _build_hotspot_region(subset_df: pd.DataFrame, meta: dict[str, Any], promoter_region: dict[str, Any]) -> dict[str, Any]:
    if meta.get("promoter_hotspot_region"):
        return dict(meta["promoter_hotspot_region"])
    if subset_df.empty or "MAPINFO" not in subset_df.columns:
        return {
            "label": "No EPIC methylation probe window",
            "start": int(promoter_region["start"]),
            "end": int(promoter_region["end"]),
            "definition": (
                "No bundled EPIC methylation probes are available for this gene; methylation review should use a custom assay rather than the nuclear CpG manifest subset."
            ),
        }

    candidate = subset_df.copy()
    if "UCSC_RefGene_Group" in candidate.columns:
        group_text = candidate["UCSC_RefGene_Group"].fillna("").astype(str)
        tss_mask = group_text.str.contains("TSS|5'UTR|1stExon", case=False, regex=True)
        if tss_mask.any():
            candidate = candidate[tss_mask].copy()

    if "Relation_to_UCSC_CpG_Island" in candidate.columns:
        island_text = candidate["Relation_to_UCSC_CpG_Island"].fillna("").astype(str)
        island_mask = island_text.str.contains("Island|Shore", case=False, regex=True)
        if island_mask.any():
            candidate = candidate[island_mask].copy()

    if candidate.empty:
        hotspot_start = int(promoter_region["start"])
        hotspot_end = int(promoter_region["end"])
        definition = (
            "Promoter-proximal review span derived from the operational promoter window because the bundled EPIC subset does not mark a smaller CpG-focused hotspot."
        )
    else:
        hotspot_start = int(candidate["MAPINFO"].min())
        hotspot_end = int(candidate["MAPINFO"].max())
        definition = (
            "Promoter-proximal CpG-focused span represented by the bundled EPIC subset and used here as the main methylation-review hotspot."
        )

    return {
        "label": "Promoter-associated CpG review window",
        "start": hotspot_start,
        "end": hotspot_end,
        "definition": definition,
    }


def _build_interpretation_database(meta: dict[str, Any], subset_df: pd.DataFrame) -> dict[str, Any]:
    assembly = str(meta.get("assembly") or ASSEMBLY)
    promoter_region = _build_promoter_region(meta)
    hotspot_region = _build_hotspot_region(subset_df, meta, promoter_region)
    probe_ids = _select_relevant_probe_ids(subset_df, meta)

    combined_start = min(
        int(meta["start"]),
        int(meta["end"]),
        int(promoter_region["start"]),
        int(promoter_region["end"]),
    )
    combined_end = max(
        int(meta["start"]),
        int(meta["end"]),
        int(promoter_region["start"]),
        int(promoter_region["end"]),
    )
    recommended_region = f"{meta['chromosome']}:{combined_start}-{combined_end}"
    gene_context = {
        "gene_name": meta["gene_name"],
        "assembly": assembly,
        "cytoband": meta["cytoband"],
        "chromosome": meta["chromosome"],
        "gene_region": {
            "label": f"{meta['gene_name']} transcribed interval",
            "start": int(meta["start"]),
            "end": int(meta["end"]),
            "definition": f"Canonical {meta['gene_name']} genomic interval on {assembly} from {meta.get('coordinate_source', 'NCBI Gene')}.",
        },
        "promoter_review_region": promoter_region,
        "promoter_hotspot_region": hotspot_region,
        "recommended_promoter_plus_gene_region": recommended_region,
        "gene_summary": meta["gene_summary"],
        "clinical_context": meta["clinical_context"],
        "variant_effect_overview": meta["variant_effect_overview"],
        "condition_research_overview": meta["condition_research_overview"],
        "relevant_methylation_probe_ids": probe_ids,
        "methylation_interpretation": meta["methylation_interpretation"],
        "methylation_effects": meta["methylation_effects"],
        "methylation_condition_research": meta["methylation_condition_research"],
        "evidence": meta["evidence"],
    }
    if meta.get("biorender_visuals"):
        gene_context["biorender_visuals"] = meta["biorender_visuals"]
    if meta.get("concrete_variant_prediction"):
        gene_context["concrete_variant_prediction"] = meta["concrete_variant_prediction"]

    variant_records: list[dict[str, Any]] = []
    for raw_variant in meta["variants"]:
        record = deepcopy(raw_variant)
        record["gene_name"] = meta["gene_name"]
        record["chromosome"] = meta["chromosome"]
        record.setdefault("display_name", record["variant"])
        record.setdefault("lookup_keys", [record["variant"], f"{meta['gene_name'].lower()}:{record['variant'].lower()}"])
        record["relevant_methylation_probe_ids"] = probe_ids
        variant_records.append(record)

    return {
        "database_name": f"NophiGene {meta['gene_name']} Interpretation Database",
        "version": VERSION,
        "gene_context": gene_context,
        "variant_records": variant_records,
    }


def _build_population_database(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "database_name": f"NophiGene {meta['gene_name']} Population Database",
        "version": VERSION,
        "assembly": ASSEMBLY,
        "coverage_note": meta["population_coverage_note"],
        "gene_population_patterns_intro": meta["population_intro"],
        "population_categories": COMMON_POPULATION_CATEGORIES,
        "sources": meta["population_sources"],
        "variant_population_records": meta.get("variant_population_records", []),
        "gene_population_patterns": meta["gene_population_patterns"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_empty_manifest_subset(meta: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
    """Write an empty manifest subset for genes without EPIC probe coverage."""
    subset_path = GENE_DATA_DIR / f"{meta['gene_name']}_epigenetics_hg19.csv"
    subset_df = pd.DataFrame(columns=EMPTY_MANIFEST_COLUMNS)
    subset_df.to_csv(subset_path, index=False)
    return subset_path, subset_df


def _write_manifest_subset_from_loaded_manifest(
    manifest_df: pd.DataFrame,
    meta: dict[str, Any],
    region: str,
) -> tuple[Path, pd.DataFrame]:
    """Write one gene-specific manifest subset using a preloaded manifest."""
    chrom, start, end = parse_region_string(region)
    subset_df = filter_probes_by_region(manifest_df, chrom, start, end, "hg19")
    if subset_df.empty:
        if not meta.get("allow_empty_manifest_subset"):
            raise ValueError(f"No manifest probes were found in {chrom}:{start}-{end} for genome build hg19.")
        subset_df = manifest_df.iloc[0:0].copy()

    subset_path = GENE_DATA_DIR / f"{meta['gene_name']}_epigenetics_hg19.csv"
    subset_df.to_csv(subset_path, index=False)
    return subset_path, subset_df


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Required manifest not found: {MANIFEST_PATH}")

    GENE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_df = load_manifest(str(MANIFEST_PATH))

    generated: list[str] = []
    for meta in GENE_DEFINITIONS:
        if meta.get("skip_manifest_subset"):
            subset_path, subset_df = _write_empty_manifest_subset(meta)
        else:
            region = str(meta.get("manifest_filter_region") or f"{meta['chromosome']}:{meta['start']}-{meta['end']}")
            subset_path, subset_df = _write_manifest_subset_from_loaded_manifest(manifest_df, meta, region)

        interpretation_path = GENE_DATA_DIR / f"{meta['gene_name'].lower()}_interpretation_db.json"
        population_path = GENE_DATA_DIR / f"{meta['gene_name'].lower()}_population_db.json"

        _write_json(interpretation_path, _build_interpretation_database(meta, subset_df))
        _write_json(population_path, _build_population_database(meta))

        generated.extend(
            [
                subset_path.name,
                interpretation_path.name,
                population_path.name,
            ]
        )

    print("Generated knowledge-base artifacts:")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
