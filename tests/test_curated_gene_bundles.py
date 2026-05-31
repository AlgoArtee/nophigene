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


BIOCHEMISTRY_GENES = {
    "TP53": 7565097,
    "BRCA1": 41196312,
    "EGFR": 55086714,
    "APOE": 45409011,
    "ACE2": 15579156,
    "CFTR": 117105838,
    "PCSK9": 55505221,
    "LDLR": 11200038,
    "HBB": 5246694,
    "HFE": 26087509,
    "ALDH2": 112204691,
    "MTHFR": 11845780,
    "LCT": 136545410,
    "BDNF": 27676440,
    "COMT": 19929130,
    "CACNA1C": 2079952,
    "SCN5A": 38589548,
    "MYH7": 23881947,
    "NTRK1": 156785432,
    "PPARG": 12328867,
    "KRAS": 25357723,
    "BRAF": 140419127,
    "PIK3CA": 178865902,
    "PTEN": 89622870,
    "AKT1": 105235686,
    "ALK": 29415640,
    "ROS1": 117609463,
    "RET": 43572475,
    "MET": 116312444,
    "ERBB2": 37844167,
    "ESR1": 151977826,
    "AR": 66764465,
    "VHL": 10182692,
    "APC": 112043195,
    "MLH1": 37034823,
    "MSH2": 47630108,
    "MSH6": 47922669,
    "PMS2": 6012870,
    "ATM": 108093211,
    "CHEK2": 29083731,
    "PALB2": 23614488,
    "BRCA2": 32889611,
    "RYR1": 38924339,
    "KCNQ1": 2465914,
    "KCNH2": 150642049,
    "APOB": 21224301,
    "F5": 169483404,
    "F2": 46740730,
    "SERPINA1": 94843084,
    "G6PD": 153759606,
    "CYP2D6": 42522501,
    "CYP2C19": 96447911,
    "CYP2C9": 96698415,
    "SLCO1B1": 21284136,
    "VKORC1": 31102163,
    "DPYD": 97543299,
    "TPMT": 18128542,
    "UGT1A1": 234668894,
    "IFNG": 68548548,
    "TNF": 31543344,
    "IL6": 22765503,
    "CRP": 159682079,
    "VDR": 48235320,
    "FTO": 53737875,
    "LEP": 127881337,
    "MC4R": 58038564,
    "HNF1A": 121416346,
    "GCK": 44183872,
    "KCNJ11": 17407406,
    "TTR": 29171689,
    "FCER1A": 159259504,
    "FCER1G": 161185024,
    "FCER2": 7753644,
    "MS4A2": 59855734,
    "TPSAB1": 1290697,
    "TPSB2": 1277272,
    "CMA1": 24974559,
    "CPA3": 148583043,
    "KIT": 55524085,
    "IL5": 131877136,
    "IL5RA": 3111233,
    "CCL11": 32612687,
    "CCR3": 46205096,
    "CCL17": 57438679,
    "CCL22": 57392684,
    "CCR4": 32993066,
    "IL8": 74606223,
    "CXCR2": 218990012,
    "NFKB1": 103422486,
    "NFKBIA": 35870717,
    "RELA": 65421067,
    "JAK1": 65298912,
    "JAK2": 4985033,
    "TYK2": 10461209,
    "STAT3": 40465342,
    "MC1R": 89978527,
    "TYR": 88910620,
    "TYRP1": 12685439,
    "DCT": 95089558,
    "OCA2": 28000021,
    "SLC45A2": 33944721,
    "SLC24A5": 48413169,
    "IRF4": 391739,
    "MITF": 69788586,
    "PAX3": 223064607,
    "SOX10": 38366693,
    "KITLG": 88886570,
    "EDNRB": 78469616,
    "EDN3": 57875482,
    "ASIP": 32782375,
    "BCL2": 60790579,
    "PMEL": 56347889,
    "GPR143": 9693386,
    "MLANA": 5890802,
    "SLC24A4": 92788925,
    "SIRT1": 69644427,
    "SIRT2": 39369197,
    "SIRT3": 215458,
    "SIRT7": 79869815,
    "KL": 33590207,
    "WRN": 30891317,
    "LMNA": 156052364,
    "TERF1": 73921099,
    "POT1": 124462440,
    "RTEL1": 62289163,
    "PARN": 14529558,
    "DKC1": 153991031,
    "TINF2": 24708849,
    "ACD": 67691415,
    "PARP1": 226548392,
    "NFE2L2": 178092323,
    "KEAP1": 10596796,
    "FOXO1": 41129804,
    "FOXO4": 70316047,
    "CDKN1A": 36644305,
    "APP": 27252861,
    "PSEN1": 73603126,
    "PSEN2": 227057885,
    "MAPT": 43971748,
    "SNCA": 90645250,
    "LRRK2": 40590546,
    "PINK1": 20959948,
    "PARK2": 161768452,
    "GBA": 155204243,
    "GRIN2A": 9852376,
    "GRIN2B": 13693165,
    "SLC6A4": 28521337,
    "MAOA": 43515467,
    "HTR2A": 47405685,
    "GABRA1": 161274197,
    "SHANK3": 51112843,
    "MECP2": 153287024,
    "FMR1": 146993469,
    "HTT": 3076408,
    "SCN1A": 166845670,
    "FUT2": 49199228,
    "NOD2": 50727514,
    "ATG16L1": 234118697,
    "CARD9": 139256355,
    "IL23R": 67632083,
    "TNFSF15": 117546915,
    "PNLIP": 118305443,
    "AMY1A": 104197912,
    "SI": 164696686,
    "ALPI": 233320833,
    "MUC2": 1074875,
    "TFF3": 43731777,
    "SLC5A1": 32439019,
    "SLC2A2": 170714137,
    "SLC10A2": 103696350,
}


OBSCURE_SURPRISING_GENES = {
    "TRPA1": 72932152,
    "SCN9A": 167051695,
    "SCN10A": 38738293,
    "SCN11A": 38887260,
    "PRDM12": 133539981,
    "NTRK2": 87283466,
    "KCNK18": 118957000,
    "KCNT1": 138594031,
    "KCNMA1": 78629359,
    "SLC6A3": 1392909,
    "TACR1": 75273590,
    "TAS2R38": 141672431,
    "TAS1R3": 1266694,
    "OR7D4": 9324526,
    "ABCC11": 48200821,
    "CHRNB2": 154540257,
    "CHRNA4": 61975420,
    "GABRB3": 26788693,
    "CACNA1H": 1203241,
    "HCN4": 73612200,
    "PNPLA3": 44319619,
    "TM6SF2": 19375173,
    "APOA5": 116660083,
    "ANGPTL3": 63063158,
    "ANGPTL4": 8428173,
    "GIPR": 46171502,
    "GHRL": 10327359,
    "NPY": 24323782,
    "POMC": 25383722,
    "UCP1": 141480588,
    "ADRA2A": 112836790,
    "ADRB3": 37820516,
    "CPS1": 211342406,
    "SLC22A12": 64358113,
    "SLC2A9": 9772777,
    "UMOD": 20344374,
    "HSD11B2": 67464555,
    "CYP1A2": 75041185,
    "NAT2": 18248755,
    "BCHE": 165490692,
    "MCM6": 136597196,
    "DEFA5": 6912831,
    "DEFB1": 6728097,
    "REG3A": 79384132,
    "AQP8": 25227052,
    "SLC26A3": 107405912,
    "SLC9A3": 473425,
    "CFB": 31895475,
    "CLDN2": 106143394,
    "DUOX2": 45384848,
    "DUOXA2": 45406519,
    "TGM2": 36756863,
    "CEL": 135937365,
    "PRSS1": 142457319,
    "SPINK1": 147204131,
    "CLEC7A": 10269376,
    "TLR7": 12885202,
    "TLR8": 12924739,
    "MBL2": 54525140,
    "FCGR2A": 161475220,
    "FCGR3A": 161511549,
    "NLRP3": 247579458,
    "NLRC4": 32449522,
    "AIRE": 45705721,
    "DOCK8": 214854,
    "LRBA": 151185594,
    "CTLA4": 204732509,
    "PTPN22": 114356433,
    "IFIH1": 163123589,
    "TNFAIP3": 138188351,
    "TYROBP": 36395303,
    "IRAK4": 44152747,
    "MYD88": 38179969,
    "C5": 123714616,
    "C7": 40909354,
    "AIFM1": 129263337,
    "OPA1": 193310933,
    "MFN2": 12040238,
    "POLG": 89859534,
    "NDUFS4": 52856463,
    "SUCLA2": 48510622,
    "ETHE1": 44010871,
    "SLC25A4": 186064395,
    "HSPA1A": 31783291,
    "HSPB1": 75931861,
    "BAG3": 121410882,
    "SQSTM1": 179233388,
    "VCP": 35056061,
    "OPTN": 13141449,
    "TBK1": 64845660,
    "CHCHD10": 24108021,
    "SPG7": 89557325,
    "ALDH3A2": 19551449,
    "SLC30A8": 117962512,
    "FMO3": 171060018,
    "ALPL": 21835858,
    "TMEM173": 138855119,
    "PLCG2": 81772702,
    "CX3CR1": 39304985,
    "GPR35": 241544848,
    "BHLHE41": 26272959,
    "HCRTR2": 55039050,
    "ADCYAP1": 904944,
    "VIP": 153071933,
    "KCNQ2": 62037542,
    "GRIA3": 122318006,
    "SLC17A7": 49932658,
    "SLC1A2": 35272753,
    "TREM2": 41126244,
    "CSF1R": 149432854,
    "DNASE1L3": 58177984,
    "TREX1": 48506445,
    "SAMHD1": 35518632,
    "RNASEH2A": 12917394,
    "IL31": 122656577,
    "IL31RA": 55147207,
    "OSMR": 38845960,
    "SIGLEC8": 51954101,
    "PLA2G7": 46671938,
    "ASGR1": 7076750,
    "CLOCK": 56294070,
    "ARNTL": 13298199,
    "PER1": 8043790,
    "PER2": 239152679,
    "PER3": 7844380,
    "CRY1": 107385142,
    "CRY2": 45868669,
    "NR1D1": 38249040,
    "RORA": 60780483,
    "NPAS2": 101436614,
    "OXTR": 8792094,
    "OXT": 3052266,
    "AVPR1A": 63539014,
    "AVPR1B": 206223976,
    "AVP": 3063202,
    "HTR1A": 63256183,
    "HTR2C": 113818551,
    "SLC6A2": 55689516,
    "SLC18A2": 119000604,
    "DBH": 136501482,
    "TH": 2185159,
    "DDC": 50526134,
    "GAD1": 171669723,
    "GAD2": 26505236,
    "GABRA2": 46250444,
    "CHRM2": 136553416,
    "CHRM3": 239549865,
    "DRD2": 113280318,
    "DRD3": 113847499,
    "DRD1": 174867042,
    "FAAH": 46859937,
    "CNR1": 88849583,
    "CNR2": 24197016,
    "TRPV1": 3468738,
    "TRPV3": 3413796,
    "PIEZO1": 88781751,
    "PIEZO2": 10666480,
    "SCN2A": 166095912,
    "SCN8A": 51984050,
    "KCNA1": 5019071,
    "FGF21": 49258816,
    "FGF23": 4477393,
    "LEPR": 65886248,
    "PYY": 42030106,
    "SST": 187386694,
    "INSR": 7112266,
    "IRS1": 227596033,
    "AKT2": 40736224,
    "GCG": 162999392,
    "DPP4": 162848751,
    "SLC5A2": 31494323,
    "PCK1": 56136136,
    "PFKM": 48498922,
    "GYS1": 49471382,
    "CPT1A": 68522088,
    "ACADM": 76190036,
    "ACADVL": 7120444,
    "CPT2": 53662101,
    "GPX1": 49394609,
    "SOD2": 160090089,
    "CAT": 34460472,
    "PRDX1": 45976708,
    "GPX4": 1103936,
    "IL1B": 113587328,
    "IL1RN": 113864791,
    "IL10": 206940947,
    "IL10RA": 117857063,
    "TGFB1": 41807492,
    "IFNAR1": 34696734,
    "IFNAR2": 34602206,
    "CXCL10": 76942273,
    "CCR5": 46411633,
    "CXCR4": 136871919,
    "CCR2": 46395225,
    "TLR3": 186990306,
    "TLR4": 120466610,
    "TLR9": 52255096,
    "NOD1": 30464143,
    "AIM2": 159032274,
    "CASP1": 104896170,
    "GSDMD": 144635377,
    "IL25": 23842018,
    "IL17A": 52051185,
    "IL17F": 52101479,
    "IL22": 68642022,
    "AHR": 17338246,
    "SPINK5": 147405246,
    "KRT10": 38974369,
    "KRT14": 39738531,
    "COL7A1": 48601506,
    "EDAR": 109510927,
    "EDA": 68835911,
    "WNT10A": 219745085,
    "WNT10B": 49359123,
    "ALDOB": 104182860,
    "SLC2A5": 9095166,
    "SLC15A1": 99336055,
    "FADS1": 61567099,
    "TCN2": 31002825,
    "CUBN": 16865963,
}


BIOCHEMISTRY_GENES.update(OBSCURE_SURPRISING_GENES)


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
    "CCDC66": 56591184,
    "TYW5": 200793636,
    "ELOVL7": 60047618,
    "SH3PXD2B": 171752185,
    "FRMD3": 85857907,
    "TMEM218": 124964285,
    "FAM170A": 118965253,
    "SYCE3": 50989541,
    "BLTP3B": 100430850,
    "CIROP": 23568271,
    "IL33": 6215805,
    "IL1RL1": 102927962,
    "ORMDL3": 38077294,
    "GSDMB": 38060848,
    "HLA-DQA1": 32595956,
    "HLA-DQB1": 32627244,
    "TSLP": 110405760,
    "IL4R": 27324989,
    "STAT6": 57489191,
    "IL13": 131991955,
    "IL4": 132009678,
    "FLG": 152274651,
    "TLR10": 38773860,
    "TNFRSF8": 12123434,
    "CD30": 12123434,
    "MUC5AC": 1151580,
    "SMAD3": 67356101,
    "IL18R1": 102927989,
    "IL18RAP": 103035149,
}
CURATED_GENES.update(BIOCHEMISTRY_GENES)


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


@pytest.mark.parametrize("gene_name", BIOCHEMISTRY_GENES)
def test_biochemistry_gene_bundle_exposes_biorender_visual_starter(gene_name: str) -> None:
    """New biochemistry-focused bundles should carry BioRender template/icon metadata."""
    knowledge_base = load_gene_interpretation_database(gene_name)
    assert knowledge_base is not None

    visuals = knowledge_base["gene_context"].get("biorender_visuals")
    assert visuals is not None
    assert visuals["provider"] == "BioRender"
    assert visuals["template_url"].startswith("https://app.biorender.com/biorender-templates/details/")
    assert visuals["recommended_icons"]
    assert visuals["icon_search_terms"]
    assert knowledge_base["gene_context"]["concrete_variant_prediction"]


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


def test_il33_rs1342326_catalog_has_literature_grounded_allele_context() -> None:
    """IL33 rs1342326 should explain the A/C marker, strand notation, and upstream locus context."""
    knowledge_base = load_gene_interpretation_database("IL33")
    population_database = load_gene_population_database("IL33")
    synthesis_database = load_gene_synthesis_database("IL33")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None

    rs1342326_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "rs1342326"
    )
    assert "9:6190076:A>C" in rs1342326_record["lookup_keys"]
    assert "reverse-complement" in rs1342326_record["clinical_interpretation"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "9",
                "id": "rs1342326",
                "pos": 6190076,
                "ref": "A",
                "alt": "C",
                "gt_raw": "0/1",
                "ad": [17, 12],
                "dp": 29,
                "gq": 67,
                "qual": 99.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="9:6188000-6192000",
    )

    assert interpretation["matched_records"][0]["variant"] == "rs1342326"
    assert "upstream/flanking A/C" in interpretation["matched_records"][0]["clinical_interpretation"]

    catalog_marker = next(
        item
        for item in interpretation["curated_named_markers"]
        if item["variant"] == "rs1342326"
    )
    assert catalog_marker["genome_location"] == "GRCh37 / hg19 chr9:6,190,076"
    assert catalog_marker["nucleotide_change"] == "A>C"
    assert catalog_marker["nucleotide_change_basis"] == "genomic REF/ALT alias"
    assert catalog_marker["reference_allele"] == "A"
    assert catalog_marker["alternate_allele"] == "C"
    assert catalog_marker["marker_type"] == "Single-nucleotide variant"
    assert catalog_marker["rsids"] == ["rs1342326"]
    assert "25.7 kb upstream" in " ".join(rs1342326_record["research_context"])
    assert any("Moffatt" in link["label"] for link in catalog_marker["research_links"])
    assert any("Ensembl GRCh37" in link["label"] for link in catalog_marker["research_links"])
    assert any("T/G" in link["label"] for link in catalog_marker["research_links"])


def test_coordinate_only_rs_marker_catalog_is_not_labeled_structural() -> None:
    """Coordinate-only rsID markers should say the local bundle lacks alleles, not imply CNV/model biology."""
    knowledge_base = {
        "database_name": "Synthetic marker database",
        "gene_context": {
            "gene_name": "TEST",
            "assembly": "GRCh37 / hg19",
            "cytoband": "1p36",
            "chromosome": "1",
            "gene_region": {
                "start": 1000,
                "end": 2000,
                "definition": "Synthetic gene interval.",
            },
            "promoter_review_region": {
                "start": 900,
                "end": 999,
                "definition": "Synthetic promoter interval.",
            },
            "promoter_hotspot_region": {
                "start": 900,
                "end": 999,
                "definition": "Synthetic promoter hotspot.",
            },
        },
        "variant_records": [
            {
                "variant": "rs12345",
                "display_name": "rs12345",
                "position": 950,
                "chromosome": "1",
                "lookup_keys": ["rs12345", "1:950"],
                "clinical_significance": "Synthetic research marker.",
                "is_assayable_in_snp_vcf": True,
            }
        ],
    }

    interpretation = build_variant_interpretations(
        pd.DataFrame(columns=["chrom", "id", "pos", "ref", "alt"]),
        knowledge_base,
        region="1:900-2000",
    )

    catalog_marker = interpretation["curated_named_markers"][0]
    assert catalog_marker["marker_type"] == "Single-nucleotide marker (alleles not bundled)"
    assert catalog_marker["nucleotide_change"] == "Exact REF/ALT not bundled for this rsID"
    assert catalog_marker["nucleotide_change_basis"] == "rsID and coordinate marker; exact REF/ALT not bundled locally"


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


def test_ccdc66_bundle_covers_retinal_ciliary_and_high_myopia_context() -> None:
    """CCDC66 should load as a retinal/ciliary microtubule and high-myopia bundle."""
    knowledge_base = load_gene_interpretation_database("CCDC66")
    population_database = load_gene_population_database("CCDC66")
    synthesis_database = load_gene_synthesis_database("CCDC66")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "3"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 56591184
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 56655865
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "3:56590184-56655865"
    assert "microtubule, ciliary-transition-zone, and retinal-development thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg26457421",
        "cg02129834",
        "cg26943727",
        "cg24569399",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "CCDC66 c.C172T / p.Q58X" in variant_ids
    assert "CCDC66 retinal degeneration/ciliary loss model" in variant_ids

    q58x_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "CCDC66 c.C172T / p.Q58X"
    )
    assert "CCDC66 p.Q58X" in q58x_record["lookup_keys"]
    assert "Research-level high-myopia marker" in q58x_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "3",
                "id": "CCDC66 p.Q58X",
                "pos": 56591241,
                "ref": "C",
                "alt": "T",
                "gt_raw": "0/1",
                "ad": [14, 10],
                "dp": 24,
                "gq": 61,
                "qual": 94.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg02129834",
                "beta": 0.64,
                "chrom": "3",
                "pos": 56591125,
                "GencodeBasicV12_NAME": "CCDC66",
                "UCSC_RefGene_Group": "TSS200",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="3:56590184-56655865",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "CCDC66 c.C172T / p.Q58X"
    assert "Research-level high-myopia marker" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.64
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "high-myopia and retinal-development thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_tyw5_bundle_covers_trna_modification_and_schizophrenia_context() -> None:
    """TYW5 should load as a tRNA hydroxylase and schizophrenia regulatory-expression bundle."""
    knowledge_base = load_gene_interpretation_database("TYW5")
    population_database = load_gene_population_database("TYW5")
    synthesis_database = load_gene_synthesis_database("TYW5")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "2"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 200793636
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 200820214
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "2:200793636-200821214"
    assert "tRNA(Phe) hydroxywybutosine and schizophrenia regulatory-expression thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg03599729",
        "cg17075961",
        "cg03947362",
        "cg06278833",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "rs796364 / rs281759" in variant_ids
    assert "rs203772" in variant_ids
    assert "TYW5 enzymatic wybutosine-hydroxylase model" in variant_ids

    rs203772_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "rs203772"
    )
    assert "TYW5 rs203772" in rs203772_record["lookup_keys"]
    assert "Research-level schizophrenia eQTL" in rs203772_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "2",
                "id": "rs203772",
                "pos": 200800000,
                "ref": "A",
                "alt": "G",
                "gt_raw": "0/1",
                "ad": [16, 9],
                "dp": 25,
                "gq": 62,
                "qual": 95.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg17075961",
                "beta": 0.63,
                "chrom": "2",
                "pos": 200820165,
                "GencodeBasicV12_NAME": "TYW5",
                "UCSC_RefGene_Group": "5'UTR;1stExon;Body;Body;Body;Body",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="2:200793636-200821214",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "rs203772 (TYW5 schizophrenia eQTL/MRI marker)"
    assert "Research-level schizophrenia eQTL" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.63
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "integrative schizophrenia eQTL and neuroimaging thesis" in row["prediction"]
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


def test_elovl7_bundle_covers_vlcfa_and_msa_locus_context() -> None:
    """ELOVL7 should load as a VLCFA-remodeling and MSA locus-context research bundle."""
    knowledge_base = load_gene_interpretation_database("ELOVL7")
    population_database = load_gene_population_database("ELOVL7")
    synthesis_database = load_gene_synthesis_database("ELOVL7")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "5"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 60047618
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 60140096
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "5:60047618-60141096"
    assert "ER fatty-acid elongase and VLCFA-remodeling thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg06320606",
        "cg13581847",
        "cg07716486",
        "cg22553062",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "rs7715147" in variant_ids
    assert "ELOVL7 functional lipid-elongation model" in variant_ids

    rs_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "rs7715147"
    )
    assert "ELOVL7 rs7715147" in rs_record["lookup_keys"]
    assert "Research-level MSA locus marker" in rs_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "5",
                "id": "rs7715147",
                "pos": 60090000,
                "ref": "A",
                "alt": "G",
                "gt_raw": "0/1",
                "ad": [14, 12],
                "dp": 26,
                "gq": 58,
                "qual": 91.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg06320606",
                "beta": 0.67,
                "chrom": "5",
                "pos": 60139941,
                "GencodeBasicV12_NAME": "ELOVL7",
                "UCSC_RefGene_Group": "5'UTR",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="5:60047618-60141096",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "rs7715147 (ELOVL7 intronic MSA locus marker)"
    assert "Research-level MSA locus marker" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.67
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "MSA GWAS-interest lipid-dyshomeostasis locus thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_sh3pxd2b_bundle_covers_tks4_and_recessive_fths_context() -> None:
    """SH3PXD2B should load as a TKS4 podosome and recessive FTHS/BDCS bundle."""
    knowledge_base = load_gene_interpretation_database("SH3PXD2B")
    population_database = load_gene_population_database("SH3PXD2B")
    synthesis_database = load_gene_synthesis_database("SH3PXD2B")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "5"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 171752185
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 171881529
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "5:171752185-171882529"
    assert "SH3PXD2B/TKS4 podosome-adaptor" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg08854128",
        "cg05224707",
        "cg26528541",
        "cg00591781",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "SH3PXD2B c.76-2A>C" in variant_ids
    assert "SH3PXD2B loss-of-function/deletion model" in variant_ids

    splice_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "SH3PXD2B c.76-2A>C"
    )
    assert "rs775217258" in splice_record["lookup_keys"]
    assert "Pathogenic Frank-ter Haar syndrome splice-acceptor variant" in splice_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "5",
                "id": "rs775217258",
                "pos": 171849502,
                "ref": "T",
                "alt": "G",
                "gt_raw": "0/1",
                "ad": [13, 12],
                "dp": 25,
                "gq": 61,
                "qual": 94.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg05224707",
                "beta": 0.64,
                "chrom": "5",
                "pos": 171881549,
                "GencodeBasicV12_NAME": "SH3PXD2B",
                "UCSC_RefGene_Group": "TSS200",
                "Relation_to_UCSC_CpG_Island": "S_Shore",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="5:171752185-171882529",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "SH3PXD2B c.76-2A>C splice acceptor"
    assert "Pathogenic Frank-ter Haar syndrome" in interpretation["matched_records"][0]["clinical_significance"]
    catalog_marker = next(
        item
        for item in interpretation["curated_named_markers"]
        if item["variant"] == "SH3PXD2B c.76-2A>C splice acceptor"
    )
    assert catalog_marker["genome_location"] == "GRCh37 / hg19 chr5:171,849,502"
    assert catalog_marker["nucleotide_change"] == "T>G"
    assert catalog_marker["reference_allele"] == "T"
    assert catalog_marker["alternate_allele"] == "G"
    assert catalog_marker["coding_change"] == "c.76-2A>C"
    assert catalog_marker["rsids"] == ["rs775217258"]
    assert catalog_marker["marker_type"] == "Single-nucleotide variant"
    assert any("ClinVar" in link["label"] for link in catalog_marker["research_links"])
    assert methylation_insights["whitelist_mean_beta"] == 0.64
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "high-priority Frank-ter Haar syndrome splice-acceptor thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_frmd3_bundle_covers_dkd_regulatory_and_cytoskeletal_context() -> None:
    """FRMD3 should load as a DKD regulatory and protein 4.1O cytoskeletal bundle."""
    knowledge_base = load_gene_interpretation_database("FRMD3")
    population_database = load_gene_population_database("FRMD3")
    synthesis_database = load_gene_synthesis_database("FRMD3")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "9"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 85857907
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 86153316
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "9:85857907-86154316"
    assert "FRMD3/protein 4.1O FERM-domain" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg21197678",
        "cg01681498",
        "cg16643109",
        "cg04008954",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "rs1888747" in variant_ids
    assert "FRMD3 tumor-suppressor/cytoskeletal model" in variant_ids

    rs_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "rs1888747"
    )
    assert "FRMD3 rs1888747" in rs_record["lookup_keys"]
    assert "Research-level diabetic kidney disease association marker" in rs_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "9",
                "id": "rs1888747",
                "pos": 86153558,
                "ref": "C",
                "alt": "G",
                "gt_raw": "0/1",
                "ad": [16, 11],
                "dp": 27,
                "gq": 59,
                "qual": 92.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg21197678",
                "beta": 0.62,
                "chrom": "9",
                "pos": 86153558,
                "GencodeBasicV12_NAME": "FRMD3",
                "UCSC_RefGene_Group": "TSS1500",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="9:85857907-86154316",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "rs1888747 (FRMD3 diabetic kidney disease regulatory locus marker)"
    assert "Research-level diabetic kidney disease" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.62
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "diabetic-kidney-disease regulatory-locus thesis" in row["prediction"]
        for row in predictive_theses["variant_prediction_rows"]
    )


def test_tmem218_bundle_covers_ciliary_transition_zone_and_jbts_context() -> None:
    """TMEM218 should load as a ciliary transition-zone/Joubert-Meckel bundle."""
    knowledge_base = load_gene_interpretation_database("TMEM218")
    population_database = load_gene_population_database("TMEM218")
    synthesis_database = load_gene_synthesis_database("TMEM218")

    assert knowledge_base is not None
    assert population_database is not None
    assert synthesis_database is not None
    assert knowledge_base["gene_context"]["chromosome"] == "11"
    assert knowledge_base["gene_context"]["gene_region"]["start"] == 124964285
    assert knowledge_base["gene_context"]["gene_region"]["end"] == 124981522
    assert knowledge_base["gene_context"]["recommended_promoter_plus_gene_region"] == "11:124964285-124982522"
    assert "ciliary-transition-zone and Joubert-Meckel ciliopathy thesis" in synthesis_database["concrete_variant_prediction"]

    relevant_probe_ids = knowledge_base["gene_context"]["relevant_methylation_probe_ids"]
    assert {
        "cg19645210",
        "cg17569390",
        "cg09410014",
        "cg01963620",
    } <= set(relevant_probe_ids)

    variant_ids = {record["variant"] for record in knowledge_base["variant_records"]}
    assert "TMEM218 c.111G>T / p.Arg37Ser" in variant_ids
    assert "TMEM218 biallelic Joubert-Meckel ciliopathy model" in variant_ids

    r37s_record = next(
        record
        for record in knowledge_base["variant_records"]
        if record["variant"] == "TMEM218 c.111G>T / p.Arg37Ser"
    )
    assert "TMEM218 p.Arg37Ser" in r37s_record["lookup_keys"]
    assert "Likely pathogenic Joubert syndrome 39 marker" in r37s_record["clinical_significance"]

    variants = pd.DataFrame(
        [
            {
                "chrom": "11",
                "id": "TMEM218 c.111G>T",
                "pos": 124971199,
                "ref": "C",
                "alt": "A",
                "gt_raw": "0/1",
                "ad": [15, 11],
                "dp": 26,
                "gq": 60,
                "qual": 93.0,
                "filter_status": "PASS",
                "filter_pass": True,
            }
        ]
    )
    methylation = pd.DataFrame(
        [
            {
                "probe_id": "cg09410014",
                "beta": 0.66,
                "chrom": "11",
                "pos": 124981664,
                "GencodeBasicV12_NAME": "TMEM218",
                "UCSC_RefGene_Group": "TSS200",
                "Relation_to_UCSC_CpG_Island": "Island",
            }
        ]
    )

    interpretation = build_variant_interpretations(
        variants,
        knowledge_base,
        region="11:124964285-124982522",
    )
    methylation_insights = build_methylation_insights(methylation, knowledge_base)
    predictive_theses = build_predictive_theses(
        variant_interpretations=interpretation,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    assert interpretation["matched_records"][0]["variant"] == "TMEM218 c.111G>T / p.Arg37Ser"
    assert "Likely pathogenic Joubert syndrome 39 marker" in interpretation["matched_records"][0]["clinical_significance"]
    assert methylation_insights["whitelist_mean_beta"] == 0.66
    assert predictive_theses["matched_case_count"] >= 1
    assert any(
        "high-priority Joubert syndrome 39" in row["prediction"]
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
