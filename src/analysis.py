#!/usr/bin/env python3
"""
DRD4 Gene Analysis Pipeline

Usage examples:
    python src/analysis.py \
        --vcf data/GFXC926398.filtered.snp.vcf.gz \
        --idat data/202277800037_R01C01 \
        --out results/drd4_report.html \
        --region 11:636269-640706
"""

from __future__ import annotations

import argparse
import gzip
import html
import io
import json
import logging
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import allel
import pandas as pd
from methylprep import run_pipeline

try:
    from .helper_functions.filter_manifest_region import (
        filter_probes_by_region,
        load_manifest,
        parse_region_string,
        sanitize_gene_name_for_filename,
        save_filtered_manifest,
    )
    from .variant_knowledge.merger import load_dynamic_knowledge_base, merge_dynamic_knowledge_base
except ImportError:
    from helper_functions.filter_manifest_region import (
        filter_probes_by_region,
        load_manifest,
        parse_region_string,
        sanitize_gene_name_for_filename,
        save_filtered_manifest,
    )
    from variant_knowledge.merger import load_dynamic_knowledge_base, merge_dynamic_knowledge_base

DEFAULT_REGION = "11:636269-640706"
DEFAULT_REPORT_NAME = "drd4_report.html"
DEFAULT_GENE_NAME = "DRD4"
DEFAULT_ANALYSIS_SCOPE = "promoter_plus_gene"
ANALYSIS_SCOPE_OPTIONS = {
    "promoter_plus_gene": {
        "label": "Promoter + gene",
        "slug": "promoter_plus_gene",
        "description": "Standard full context: upstream promoter review window plus the transcribed gene body.",
    },
    "promoter_only": {
        "label": "Promoter only",
        "slug": "promoter_only",
        "description": "Focused report for the upstream promoter review window only.",
    },
    "gene_only": {
        "label": "Gene only",
        "slug": "gene_only",
        "description": "Focused report for the canonical transcribed gene interval only.",
    },
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENE_DATA_DIR = Path(__file__).resolve().parent / "gene_data"
GENE_DATA_BUNDLE_PATH = GENE_DATA_DIR / "gene_data_bundle.zip"
GENE_DATA_INDEX_PATH = GENE_DATA_DIR / "gene_data_index.json"
INTERPRETATION_DB_PATH = Path(__file__).resolve().parent / "gene_data" / "drd4_interpretation_db.json"
POPULATION_DB_PATH = Path(__file__).resolve().parent / "gene_data" / "drd4_population_db.json"
SYNTHESIS_DB_PATH = Path(__file__).resolve().parent / "gene_data" / "drd4_synthesis.json"
GENERAL_ANALYSIS_DATABASE_PATH = PROJECT_ROOT / "results" / "general_gene_analysis_database.csv"
GENERAL_ANALYSIS_DATABASE_COLUMNS = [
    "gene",
    "sample",
    "variant key",
    "observed gene variant",
    "gene variant label",
    "change",
    "genotype",
    "zygosity",
    "allele dosage",
    "chromosome",
    "position",
    "variant location",
    "gene location",
    "source",
    "(VCF) quality (qual)",
    "VCF filter",
    "VCF depth (DP)",
    "VCF allele depths (AD)",
    "VCF genotype quality (GQ)",
    "genotype confidence",
    "genotype QC flags",
    "matched curated marker",
    "variant interpretation scope",
    "curated biological significance",
    "functional effects",
    "associated conditions",
    "methylation-linked probes",
    "mean beta whitelist",
    "mean beta related to gene",
    "mean beta on found probes in the area (numerical rows)",
]


@dataclass(frozen=True)
class GenotypeQCConfig:
    """Configurable thresholds for sample-level VCF genotype confidence."""

    min_depth: int = 10
    low_depth: int = 20
    min_gq: float = 20.0
    min_qual: float = 20.0
    heterozygous_balance_mild: float = 0.30
    heterozygous_balance_severe: float = 0.15
    homozygous_contamination_fraction: float = 0.10
    dp_ad_mismatch_fraction: float = 0.20
    min_pl_delta: float = 20.0
    min_gp_called_probability: float = 0.80
    min_gp_probability_margin: float = 0.20


DEFAULT_GENOTYPE_QC_CONFIG = GenotypeQCConfig()
GENOTYPE_OUTPUT_COLUMNS = [
    "sample",
    "gt_raw",
    "phased",
    "genotype",
    "genotype_alleles",
    "zygosity",
    "allele_dosage_per_alt",
    "filter_status",
    "dp",
    "ad",
    "sample_af",
    "sample_af_source",
    "gq",
    "pl_or_gp_summary",
    "sb",
    "f1r2",
    "f2r1",
    "strand_bias_summary",
    "qc_flags",
    "confidence_score",
    "confidence_explanation",
]
BASE_VARIANT_COLUMNS = [
    "sample",
    "chrom",
    "id",
    "pos",
    "ref",
    "alt",
    "alt_alleles",
    "qual",
    "filter_pass",
    "filter_status",
]

# Configure the root logger once so both CLI and web runs stream progress.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def normalize_analysis_scope(scope: str | None) -> str:
    """Return a supported report focus key."""
    normalized_scope = str(scope or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_scope in ANALYSIS_SCOPE_OPTIONS:
        return normalized_scope
    return DEFAULT_ANALYSIS_SCOPE


def get_analysis_scope_label(scope: str | None) -> str:
    """Return a user-facing report focus label."""
    return ANALYSIS_SCOPE_OPTIONS[normalize_analysis_scope(scope)]["label"]


def get_analysis_scope_slug(scope: str | None) -> str:
    """Return a filesystem-friendly report focus suffix."""
    return ANALYSIS_SCOPE_OPTIONS[normalize_analysis_scope(scope)]["slug"]


def _candidate_gene_database_paths(gene_name: str, suffix: str) -> list[Path]:
    """Return likely per-gene knowledge-base paths in priority order."""
    sanitized_gene_name = sanitize_gene_name_for_filename(gene_name)
    candidates = [
        GENE_DATA_DIR / f"{sanitized_gene_name.lower()}_{suffix}",
        GENE_DATA_DIR / f"{sanitized_gene_name}_{suffix}",
        GENE_DATA_DIR / f"{sanitized_gene_name.upper()}_{suffix}",
    ]

    unique_candidates: list[Path] = []
    seen_paths: set[str] = set()
    for path in candidates:
        path_key = str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        unique_candidates.append(path)
    return unique_candidates


def _candidate_gene_data_filenames(gene_name: str, suffix: str) -> list[str]:
    """Return likely bundled filenames for a per-gene artifact."""
    return [candidate.name for candidate in _candidate_gene_database_paths(gene_name, suffix)]


def _candidate_gene_manifest_paths(gene_name: str, genome_build: str = "hg19") -> list[Path]:
    """Return likely per-gene manifest subset paths in priority order."""
    return _candidate_gene_database_paths(gene_name, f"epigenetics_{genome_build}.csv")


@lru_cache(maxsize=8)
def _gene_data_bundle_members(bundle_path: str) -> frozenset[str]:
    """Return member names from the compressed gene-data bundle."""
    path = Path(bundle_path)
    if not path.exists():
        return frozenset()
    with zipfile.ZipFile(path) as bundle:
        return frozenset(info.filename for info in bundle.infolist() if not info.is_dir())


@lru_cache(maxsize=4096)
def _read_gene_data_bundle_member_bytes(bundle_path: str, member_name: str) -> bytes | None:
    """Read one member from the compressed gene-data bundle, if present."""
    if member_name not in _gene_data_bundle_members(bundle_path):
        return None
    with zipfile.ZipFile(bundle_path) as bundle:
        return bundle.read(member_name)


@lru_cache(maxsize=8)
def _gene_data_index(index_path: str) -> dict[str, Any]:
    """Return the sharded bulk gene-data index, or an empty index when absent."""
    path = Path(index_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Bulk gene-data index could not be read: %s", path, exc_info=True)
        return {}


@lru_cache(maxsize=8)
def _bulk_gene_data_members(index_path: str) -> frozenset[str]:
    """Return sharded bulk archive member names from the index."""
    files = _gene_data_index(index_path).get("files", {})
    if not isinstance(files, dict):
        return frozenset()
    return frozenset(str(filename) for filename in files)


def _bulk_gene_data_shard_path(index_path: str, gene_data_dir: str, filename: str) -> Path | None:
    """Return the shard path for one bulk archive member when indexed."""
    index = _gene_data_index(index_path)
    files = index.get("files", {})
    shards = index.get("shards", {})
    if not isinstance(files, dict) or not isinstance(shards, dict):
        return None
    shard_name = files.get(filename)
    if not isinstance(shard_name, str):
        return None
    shard_info = shards.get(shard_name, {})
    relative_path = shard_info.get("path") if isinstance(shard_info, dict) else None
    if not isinstance(relative_path, str):
        relative_path = f"bulk_gene_data_shards/{shard_name}"
    return Path(gene_data_dir) / relative_path


@lru_cache(maxsize=4096)
def _read_bulk_gene_data_member_bytes(index_path: str, gene_data_dir: str, member_name: str) -> bytes | None:
    """Read one member from the sharded bulk gene-data archives, if indexed."""
    shard_path = _bulk_gene_data_shard_path(index_path, gene_data_dir, member_name)
    if shard_path is None or not shard_path.exists():
        return None
    try:
        with zipfile.ZipFile(shard_path) as bundle:
            return bundle.read(member_name)
    except (OSError, zipfile.BadZipFile, KeyError):
        logger.warning("Bulk gene-data member could not be read: %s in %s", member_name, shard_path, exc_info=True)
        return None


def clear_gene_data_bundle_cache() -> None:
    """Clear cached bundle membership/content; useful after rebuilding the archive in-process."""
    _gene_data_bundle_members.cache_clear()
    _read_gene_data_bundle_member_bytes.cache_clear()
    _gene_data_index.cache_clear()
    _bulk_gene_data_members.cache_clear()
    _read_bulk_gene_data_member_bytes.cache_clear()


def list_gene_data_bundle_members(suffix: str | None = None) -> list[str]:
    """List bundled archive members, optionally filtered by filename suffix."""
    members = sorted(_gene_data_bundle_members(str(GENE_DATA_BUNDLE_PATH)))
    if suffix is None:
        return members
    return [member for member in members if member.endswith(suffix)]


def list_gene_data_bulk_members(suffix: str | None = None) -> list[str]:
    """List sharded bulk archive members, optionally filtered by filename suffix."""
    members = sorted(_bulk_gene_data_members(str(GENE_DATA_INDEX_PATH)))
    if suffix is None:
        return members
    return [member for member in members if member.endswith(suffix)]


def list_available_gene_data_files(suffix: str | None = None) -> list[str]:
    """List gene-data filenames available either loose on disk or inside the archive."""
    loose_files = {path.name for path in GENE_DATA_DIR.iterdir() if path.is_file()} if GENE_DATA_DIR.exists() else set()
    bundled_files = set(_gene_data_bundle_members(str(GENE_DATA_BUNDLE_PATH)))
    bulk_files = set(_bulk_gene_data_members(str(GENE_DATA_INDEX_PATH)))
    filenames = sorted(loose_files | bundled_files | bulk_files)
    if suffix is None:
        return filenames
    return [filename for filename in filenames if filename.endswith(suffix)]


def _gene_data_bundle_has_member(filename: str) -> bool:
    """Return whether the compressed bundle contains the requested filename."""
    return filename in _gene_data_bundle_members(str(GENE_DATA_BUNDLE_PATH))


def _bulk_gene_data_has_member(filename: str) -> bool:
    """Return whether the sharded bulk archives contain the requested filename."""
    return filename in _bulk_gene_data_members(str(GENE_DATA_INDEX_PATH))


def _read_gene_data_text(filename: str) -> str | None:
    """Read a gene-data text artifact, preferring loose files over curated and bulk archives."""
    loose_path = GENE_DATA_DIR / filename
    if loose_path.exists():
        return loose_path.read_text(encoding="utf-8")
    bundled_bytes = _read_gene_data_bundle_member_bytes(str(GENE_DATA_BUNDLE_PATH), filename)
    if bundled_bytes is not None:
        return bundled_bytes.decode("utf-8")
    bulk_bytes = _read_bulk_gene_data_member_bytes(str(GENE_DATA_INDEX_PATH), str(GENE_DATA_DIR), filename)
    if bulk_bytes is not None:
        return bulk_bytes.decode("utf-8")
    return None


def find_gene_database_path(gene_name: str, suffix: str) -> Path | None:
    """Locate a bundled per-gene database when one is available."""
    for candidate in _candidate_gene_database_paths(gene_name, suffix):
        if candidate.exists():
            return candidate
    return None


def find_gene_manifest_path(gene_name: str, genome_build: str = "hg19") -> Path | None:
    """Locate a loose per-gene manifest subset when one is available."""
    for candidate in _candidate_gene_manifest_paths(gene_name, genome_build):
        if candidate.exists():
            return candidate
    return None


class AnalysisError(RuntimeError):
    """Raised when the DRD4 workflow cannot complete successfully."""


@dataclass
class AnalysisResult:
    """Structured result returned by :func:`run_analysis`.

    Attributes
    ----------
    variants : pd.DataFrame
        Regional VCF sample calls loaded from the source VCF with decoded GT.
    methylation : pd.DataFrame
        Probe-level methylation table after joining with the curated DRD4
        manifest subset.
    popstats : Any | None
        Optional population statistics payload loaded from a user-supplied CSV
        or JSON file.
    report_path : Path
        Path to the generated output report.
    methylation_output_path : Path
        Path to the exported methylation CSV companion file.
    region : str
        Genomic interval used for the run.
    analysis_scope : str
        Machine-readable report focus key.
    analysis_scope_label : str
        Human-readable report focus shown in the UI and exported report.
    vcf_path : Path
        Input VCF path used during execution.
    idat_base : Path
        Input IDAT prefix used during execution.
    variant_interpretations : dict[str, Any]
        Curated variant interpretations matched against loaded VCF calls using
        sample-level GT plus the local interpretation database.
    methylation_insights : dict[str, Any]
        Gene-level methylation interpretation assembled from the current probe
        table plus the local interpretation database.
    knowledge_base : dict[str, Any]
        Parsed local interpretation database used to build the biological and
        clinical insights shown in the UI.
    population_insights : dict[str, Any]
        Population-frequency and geography summaries assembled from the local
        DRD4 population database.
    population_database : dict[str, Any]
        Parsed local population database used to add location-based frequency
        context for common DRD4 variants.
    predictive_theses : dict[str, Any]
        Gene-level predictive synthesis payload assembled from the local
        synthesis database plus the current sample's variant and methylation
        results.
    general_database_path : Path
        Path to the central variant-level analysis database.
    general_database_status : str
        Human-readable status describing whether this run added, skipped, or
        overwrote central database rows.
    dynamic_knowledge_base_path : Path | None
        Optional dynamic knowledge-base artifact merged into the interpretation
        bundle before variant matching.
    dynamic_knowledge_base_status : str
        Human-readable status for the dynamic merge step.
    """

    variants: pd.DataFrame
    methylation: pd.DataFrame
    popstats: Any | None
    report_path: Path
    methylation_output_path: Path
    region: str
    analysis_scope: str
    analysis_scope_label: str
    vcf_path: Path
    idat_base: Path
    variant_interpretations: dict[str, Any]
    methylation_insights: dict[str, Any]
    knowledge_base: dict[str, Any]
    population_insights: dict[str, Any]
    population_database: dict[str, Any]
    predictive_theses: dict[str, Any]
    general_database_path: Path
    general_database_status: str
    dynamic_knowledge_base_path: Path | None
    dynamic_knowledge_base_status: str


@dataclass
class PreparedAnalysisResult:
    """In-memory analysis result that can be rendered into one or more formats."""

    variants: pd.DataFrame
    methylation: pd.DataFrame
    popstats: Any | None
    region: str
    analysis_scope: str
    analysis_scope_label: str
    variant_interpretations: dict[str, Any]
    methylation_insights: dict[str, Any]
    knowledge_base: dict[str, Any]
    population_insights: dict[str, Any]
    population_database: dict[str, Any]
    predictive_theses: dict[str, Any]
    general_database_path: Path
    general_database_status: str
    dynamic_knowledge_base_path: Path | None
    dynamic_knowledge_base_status: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the DRD4 analysis workflow.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse. When omitted, argparse reads directly from
        ``sys.argv``. Accepting an explicit list lets the Docker launcher reuse
        the CLI parser without shelling out to a subprocess.

    Returns
    -------
    argparse.Namespace
        Parsed arguments ready to be consumed by :func:`main`.
    """
    parser = argparse.ArgumentParser(
        description="DRD4 gene analysis: variants, methylation, population stats, and report generation."
    )
    parser.add_argument(
        "--vcf",
        required=True,
        help="Tabix-indexed VCF file with filtered SNPs (for example, *.vcf.gz).",
    )
    parser.add_argument(
        "--idat",
        required=True,
        help="Path prefix to Illumina IDATs, without the _Grn/_Red suffixes.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path for the output report (HTML, CSV, or JSON).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Genomic region in chr:start-end format. Defaults to the DRD4 promoter+gene GRCh37 interval.",
    )
    parser.add_argument(
        "--analysis-scope",
        choices=sorted(ANALYSIS_SCOPE_OPTIONS),
        default=DEFAULT_ANALYSIS_SCOPE,
        help="Report focus label for this run: promoter_plus_gene, promoter_only, or gene_only.",
    )
    parser.add_argument(
        "--popstats",
        default=None,
        help="Optional JSON or CSV with population-frequency data.",
    )
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="Optional manifest file passed through to methylprep.",
    )

    return parser.parse_args(argv)


def _serialize_popstats(popstats: Any | None) -> Any:
    """Convert population statistics to a JSON-friendly representation."""
    if popstats is None:
        return None
    if isinstance(popstats, pd.DataFrame):
        return popstats.to_dict(orient="records")
    return popstats


def _derive_methylation_output_path(output_path: str | Path) -> Path:
    """Derive the companion methylation CSV path from the requested report path."""
    report_path = Path(output_path)
    stem = report_path.stem if report_path.suffix else report_path.name
    return report_path.with_name(f"{stem}_methylation.csv")


def _load_json_database(
    database_path: str | Path,
    *,
    missing_label: str,
    invalid_label: str,
) -> dict[str, Any]:
    """Load a JSON-backed local database with a consistent error surface."""
    payload_path = Path(database_path)
    payload_text: str | None
    if payload_path.exists():
        try:
            payload_text = payload_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AnalysisError(f"{missing_label}: {payload_path}") from exc
    elif payload_path.parent == GENE_DATA_DIR:
        payload_text = _read_gene_data_text(payload_path.name)
    else:
        payload_text = None

    if payload_text is None:
        raise AnalysisError(f"{missing_label}: {payload_path}")

    try:
        return json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{invalid_label}: {payload_path}") from exc


def load_interpretation_database(database_path: str | Path = INTERPRETATION_DB_PATH) -> dict[str, Any]:
    """Load the curated local DRD4 interpretation database.

    Parameters
    ----------
    database_path : str | Path, optional
        Path to the JSON-backed local knowledge base that stores DRD4 variant
        and methylation interpretations.

    Returns
    -------
    dict[str, Any]
        Parsed JSON payload describing gene context plus curated variant
        records.

    Raises
    ------
    AnalysisError
        Raised when the JSON file cannot be found or parsed.
    """
    return _load_json_database(
        database_path,
        missing_label="Interpretation database not found",
        invalid_label="Interpretation database is not valid JSON",
    )


def load_population_database(database_path: str | Path = POPULATION_DB_PATH) -> dict[str, Any]:
    """Load the curated local DRD4 population database."""
    return _load_json_database(
        database_path,
        missing_label="Population database not found",
        invalid_label="Population database is not valid JSON",
    )


def load_synthesis_database(database_path: str | Path = SYNTHESIS_DB_PATH) -> dict[str, Any]:
    """Load the curated local predictive synthesis database."""
    return _load_json_database(
        database_path,
        missing_label="Synthesis database not found",
        invalid_label="Synthesis database is not valid JSON",
    )


def load_gene_interpretation_database(gene_name: str) -> dict[str, Any] | None:
    """Load a bundled gene-specific interpretation database when one exists."""
    for database_path in _candidate_gene_database_paths(gene_name, "interpretation_db.json"):
        if (
            database_path.exists()
            or _gene_data_bundle_has_member(database_path.name)
            or _bulk_gene_data_has_member(database_path.name)
        ):
            return load_interpretation_database(database_path)
    return None


def load_gene_population_database(gene_name: str) -> dict[str, Any] | None:
    """Load a bundled gene-specific population database when one exists."""
    for database_path in _candidate_gene_database_paths(gene_name, "population_db.json"):
        if (
            database_path.exists()
            or _gene_data_bundle_has_member(database_path.name)
            or _bulk_gene_data_has_member(database_path.name)
        ):
            return load_population_database(database_path)
    return None


def load_gene_synthesis_database(gene_name: str) -> dict[str, Any] | None:
    """Load a bundled gene-specific predictive synthesis database when one exists."""
    for database_path in _candidate_gene_database_paths(gene_name, "synthesis.json"):
        if (
            database_path.exists()
            or _gene_data_bundle_has_member(database_path.name)
            or _bulk_gene_data_has_member(database_path.name)
        ):
            return load_synthesis_database(database_path)
    return None


def load_gene_epigenetics_manifest(gene_name: str, genome_build: str = "hg19") -> pd.DataFrame | None:
    """Load a bundled gene-specific epigenetics manifest subset when one exists."""
    for manifest_path in _candidate_gene_manifest_paths(gene_name, genome_build):
        if manifest_path.exists():
            return pd.read_csv(manifest_path)
        bundled_bytes = _read_gene_data_bundle_member_bytes(str(GENE_DATA_BUNDLE_PATH), manifest_path.name)
        if bundled_bytes is not None:
            return pd.read_csv(io.BytesIO(bundled_bytes))
        bulk_bytes = _read_bulk_gene_data_member_bytes(str(GENE_DATA_INDEX_PATH), str(GENE_DATA_DIR), manifest_path.name)
        if bulk_bytes is not None:
            return pd.read_csv(io.BytesIO(bulk_bytes))
    return None


def _decode_scalar(value: Any) -> Any:
    """Decode byte-valued VCF fields into plain Python strings when needed."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _is_missing_value(value: Any) -> bool:
    """Return true for scalar missing values without treating lists as ambiguous."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_vcf_text(value: Any) -> str:
    """Convert VCF-ish scalar values into clean display text."""
    if _is_missing_value(value):
        return ""
    decoded = _decode_scalar(value)
    return str(decoded).strip()


def _as_clean_list(value: Any) -> list[Any]:
    """Convert scalar/list/array values into a clean Python list."""
    if _is_missing_value(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [item for item in value if not _is_missing_value(item)]
    text = _clean_vcf_text(value)
    if not text or text == ".":
        return []
    return [part.strip() for part in text.split(",") if part.strip() and part.strip() != "."]


def _safe_float(value: Any) -> float | None:
    """Parse a VCF scalar as a float when possible."""
    if _is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        text = _clean_vcf_text(value)
        if not text or text == ".":
            return None
        try:
            return float(text)
        except ValueError:
            return None


def _safe_int(value: Any) -> int | None:
    """Parse a VCF scalar as an integer when possible."""
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_numeric_list(value: Any, *, as_int: bool = False) -> list[float | int]:
    """Parse comma-separated or array-valued FORMAT fields."""
    parsed: list[float | int] = []
    for item in _as_clean_list(value):
        number = _safe_float(item)
        if number is None:
            continue
        parsed.append(int(number) if as_int else number)
    return parsed


def _normalize_alt_alleles(value: Any) -> list[str]:
    """Return non-empty ALT alleles from a VCF ALT field."""
    alleles: list[str] = []
    for allele in _as_clean_list(value):
        text = _clean_vcf_text(allele).upper()
        if not text or text in {".", "NAN", "NONE"}:
            continue
        alleles.append(text)
    return alleles


def _format_alt_alleles(alt_alleles: list[str]) -> str:
    """Render ALT alleles as a stable VCF-style string."""
    return ",".join(alt_alleles)


def _format_gt_from_codes(codes: list[int | None], *, phased: bool) -> str:
    """Render numeric genotype codes as a VCF GT string."""
    separator = "|" if phased else "/"
    rendered = ["." if code is None or int(code) < 0 else str(int(code)) for code in codes]
    return separator.join(rendered) if rendered else "./."


def _parse_gt_raw(gt_raw: Any) -> tuple[list[int | None], bool, str]:
    """Parse phased or unphased VCF GT text into allele codes."""
    text = _clean_vcf_text(gt_raw)
    if not text:
        return [], False, ""
    phased = "|" in text
    separator = "|" if phased else "/"
    parts = text.replace("|", separator).replace("/", separator).split(separator)
    codes: list[int | None] = []
    for part in parts:
        cleaned = part.strip()
        if cleaned in {"", "."}:
            codes.append(None)
            continue
        try:
            codes.append(int(cleaned))
        except ValueError:
            codes.append(None)
    return codes, phased, text


def _extract_gt_codes(row: pd.Series | dict[str, Any]) -> tuple[list[int | None], bool, str]:
    """Read GT from text or numeric-code columns and preserve phasing."""
    gt_raw = row.get("gt_raw") if hasattr(row, "get") else None
    if _clean_vcf_text(gt_raw):
        return _parse_gt_raw(gt_raw)

    gt_raw = row.get("GT") if hasattr(row, "get") else None
    if _clean_vcf_text(gt_raw):
        return _parse_gt_raw(gt_raw)

    for key in ("gt_codes", "genotype_codes"):
        codes_value = row.get(key) if hasattr(row, "get") else None
        codes = _as_clean_list(codes_value)
        if not codes:
            continue
        parsed_codes: list[int | None] = []
        for code in codes:
            if _clean_vcf_text(code) in {"", "."}:
                parsed_codes.append(None)
                continue
            parsed_codes.append(_safe_int(code))
        phased = bool(row.get("phased", False)) if hasattr(row, "get") else False
        return parsed_codes, phased, _format_gt_from_codes(parsed_codes, phased=phased)

    return [], False, ""


def _genotype_allele_label(code: int | None, ref: str, alt_alleles: list[str]) -> str:
    """Map a numeric GT allele code back to REF/ALT sequence text."""
    if code is None or code < 0:
        return "."
    if code == 0:
        return ref
    alt_index = code - 1
    if 0 <= alt_index < len(alt_alleles):
        return alt_alleles[alt_index]
    return f"?{code}"


def _classify_zygosity(codes: list[int | None]) -> str:
    """Classify sample genotype state from decoded GT allele codes."""
    if not codes or any(code is None or code < 0 for code in codes):
        return "missing"
    if len(codes) != 2:
        if len(codes) == 1:
            return "hemizygous_reference" if codes[0] == 0 else "hemizygous_alternate"
        return "non_diploid"
    first, second = codes
    if first == 0 and second == 0:
        return "homozygous_reference"
    if first == second and first > 0:
        return "homozygous_alternate"
    if first != second and (first == 0 or second == 0):
        return "heterozygous"
    return "compound_heterozygous"


def _format_genotype_display(alleles: list[str], *, phased: bool) -> str:
    """Render decoded alleles using the original GT phasing delimiter."""
    if not alleles:
        return "Unavailable"
    separator = "|" if phased else "/"
    return separator.join(alleles)


def _allele_dosage_per_alt(codes: list[int | None], alt_alleles: list[str]) -> dict[str, int]:
    """Count how many copies of each ALT allele are present in the sample GT."""
    dosage = {allele: 0 for allele in alt_alleles}
    for code in codes:
        if code is None or code <= 0:
            continue
        alt_index = code - 1
        if 0 <= alt_index < len(alt_alleles):
            dosage[alt_alleles[alt_index]] += 1
    return dosage


def _format_allele_dosage(dosage: dict[str, int] | Any) -> str:
    """Render ALT dosage dicts in a compact table-friendly form."""
    if not isinstance(dosage, dict) or not dosage:
        return "Unavailable"
    return "; ".join(f"{allele}:{count}" for allele, count in dosage.items())


def _genotype_has_alt_dosage(genotype: dict[str, Any] | pd.Series | dict[str, Any]) -> bool:
    """Return whether decoded GT contains at least one ALT allele."""
    dosage = genotype.get("allele_dosage_per_alt", {}) if hasattr(genotype, "get") else {}
    return isinstance(dosage, dict) and any(int(count or 0) > 0 for count in dosage.values())


def _format_filter_status(row: pd.Series | dict[str, Any]) -> str:
    """Render VCF FILTER status while keeping PASS distinct from unknown."""
    raw_status = row.get("filter_status") if hasattr(row, "get") else None
    status_text = _clean_vcf_text(raw_status)
    if status_text:
        return "PASS" if status_text in {".", "PASS", "True", "true"} else status_text
    if hasattr(row, "get") and "filter_pass" in row:
        return "PASS" if bool(row.get("filter_pass")) else "Non-PASS"
    return "Unknown"


def _diploid_pl_order(max_allele_index: int) -> list[tuple[int, int]]:
    """Return the VCF-standard diploid PL/GP genotype ordering."""
    order: list[tuple[int, int]] = []
    for second in range(max_allele_index + 1):
        for first in range(second + 1):
            order.append((first, second))
    return order


def _called_diploid_index(codes: list[int | None], max_allele_index: int) -> int | None:
    """Return the PL/GP index for the called diploid genotype."""
    if len(codes) != 2 or any(code is None or code < 0 for code in codes):
        return None
    ordered_call = tuple(sorted(int(code) for code in codes))
    for index, genotype in enumerate(_diploid_pl_order(max_allele_index)):
        if genotype == ordered_call:
            return index
    return None


def _summarize_likelihood_support(
    *,
    codes: list[int | None],
    alt_alleles: list[str],
    pl_values: list[float | int],
    gp_values: list[float | int],
    qc_flags: list[str],
    qc_config: GenotypeQCConfig,
) -> str:
    """Summarize PL or GP support for the reported sample genotype."""
    max_allele_index = len(alt_alleles)
    called_index = _called_diploid_index(codes, max_allele_index)
    if called_index is None:
        return "Unavailable"

    if pl_values and called_index < len(pl_values):
        called_pl = float(pl_values[called_index])
        other_pl = [float(value) for index, value in enumerate(pl_values) if index != called_index]
        best_other = min(other_pl) if other_pl else None
        if best_other is None:
            return f"PL called={called_pl:g}; no alternate likelihood available"
        delta = best_other - called_pl
        if delta < qc_config.min_pl_delta:
            qc_flags.append("pl_weak_support")
        return f"PL called={called_pl:g}; next-best delta={delta:g}"

    if gp_values and called_index < len(gp_values):
        called_gp = float(gp_values[called_index])
        other_gp = [float(value) for index, value in enumerate(gp_values) if index != called_index]
        best_other = max(other_gp) if other_gp else None
        if best_other is None:
            return f"GP called={called_gp:.3g}; no alternate posterior available"
        margin = called_gp - best_other
        if called_gp < qc_config.min_gp_called_probability or margin < qc_config.min_gp_probability_margin:
            qc_flags.append("gp_weak_support")
        return f"GP called={called_gp:.3g}; next-best margin={margin:.3g}"

    return "Unavailable"


def _compute_sample_af(
    *,
    explicit_af: Any,
    ad_values: list[int | float],
) -> tuple[float | list[float] | None, str]:
    """Return sample-level alternate read fraction from FORMAT/AF or AD."""
    parsed_af = _parse_numeric_list(explicit_af)
    if parsed_af:
        if len(parsed_af) == 1:
            return float(parsed_af[0]), "FORMAT/AF"
        return [float(value) for value in parsed_af], "FORMAT/AF"

    if len(ad_values) >= 2:
        total_depth = sum(float(value) for value in ad_values if value is not None)
        if total_depth > 0:
            fractions = [round(float(value) / total_depth, 6) for value in ad_values[1:]]
            return fractions[0] if len(fractions) == 1 else fractions, "AD"
    return None, ""


def _summarize_bias_fields(
    *,
    sb_values: list[int | float],
    f1r2_values: list[int | float],
    f2r1_values: list[int | float],
    qc_flags: list[str],
) -> str:
    """Summarize strand/orientation evidence when callers provide it."""
    summaries: list[str] = []
    if len(sb_values) >= 4:
        ref_forward, ref_reverse, alt_forward, alt_reverse = [float(value) for value in sb_values[:4]]
        alt_total = alt_forward + alt_reverse
        if alt_total > 0:
            alt_minor_fraction = min(alt_forward, alt_reverse) / alt_total
            summaries.append(f"SB alt forward/reverse={alt_forward:g}/{alt_reverse:g}")
            if alt_minor_fraction < 0.10:
                qc_flags.append("strand_bias_possible")
        ref_total = ref_forward + ref_reverse
        if ref_total > 0:
            summaries.append(f"SB ref forward/reverse={ref_forward:g}/{ref_reverse:g}")

    if f1r2_values and f2r1_values:
        f1r2_total = sum(float(value) for value in f1r2_values)
        f2r1_total = sum(float(value) for value in f2r1_values)
        orientation_total = f1r2_total + f2r1_total
        if orientation_total > 0:
            orientation_minor_fraction = min(f1r2_total, f2r1_total) / orientation_total
            summaries.append(f"orientation F1R2/F2R1={f1r2_total:g}/{f2r1_total:g}")
            if orientation_minor_fraction < 0.10:
                qc_flags.append("orientation_bias_possible")

    return "; ".join(summaries) if summaries else "Unavailable"


def _sample_af_for_alt(sample_af: float | list[float] | None, alt_index: int) -> float | None:
    """Return sample AF for one ALT allele index."""
    if sample_af is None:
        return None
    if isinstance(sample_af, list):
        if 0 <= alt_index < len(sample_af):
            return float(sample_af[alt_index])
        return None
    return float(sample_af) if alt_index == 0 else None


def _score_genotype_confidence(
    *,
    zygosity: str,
    gt_raw: str,
    filter_status: str,
    qual: float | None,
    dp: int | None,
    ad_values: list[int | float],
    gq: float | None,
    sample_af: float | list[float] | None,
    alt_alleles: list[str],
    allele_dosage: dict[str, int],
    qc_flags: list[str],
    qc_config: GenotypeQCConfig,
) -> tuple[float, str]:
    """Apply conservative genotype-call QC heuristics and return score + explanation."""
    score = 1.0
    notes: list[str] = []

    if "pl_weak_support" in qc_flags:
        notes.append("PL values do not strongly separate the reported genotype from the next-best genotype.")
        score -= 0.12
    if "gp_weak_support" in qc_flags:
        notes.append("GP values do not strongly support the reported genotype posterior.")
        score -= 0.12
    if "strand_bias_possible" in qc_flags:
        notes.append("SB suggests possible strand imbalance.")
        score -= 0.08
    if "orientation_bias_possible" in qc_flags:
        notes.append("F1R2/F2R1 suggests possible read-orientation imbalance.")
        score -= 0.08

    if not gt_raw or zygosity == "missing":
        qc_flags.append("missing_gt")
        notes.append("GT is missing, so the person's genotype cannot be inferred from REF/ALT.")
        score -= 0.65
    if filter_status != "PASS":
        qc_flags.append("filter_non_pass")
        notes.append(f"FILTER is {filter_status}, not PASS.")
        score -= 0.25
    if qual is not None and qual < qc_config.min_qual:
        qc_flags.append("low_qual")
        notes.append(f"QUAL {qual:g} is below the configured {qc_config.min_qual:g} threshold.")
        score -= 0.15
    if dp is None:
        qc_flags.append("missing_dp")
        notes.append("FORMAT/DP is unavailable.")
        score -= 0.08
    elif dp < qc_config.min_depth:
        qc_flags.append("very_low_dp")
        notes.append(f"DP {dp} is below the configured minimum depth {qc_config.min_depth}.")
        score -= 0.30
    elif dp < qc_config.low_depth:
        qc_flags.append("low_dp")
        notes.append(f"DP {dp} is below the preferred depth {qc_config.low_depth}.")
        score -= 0.10
    if gq is not None and gq < qc_config.min_gq:
        qc_flags.append("low_gq")
        notes.append(f"GQ {gq:g} is below the configured {qc_config.min_gq:g} threshold.")
        score -= 0.25
    if gq is None:
        notes.append("GQ is unavailable, so confidence relies on depth and allele balance.")

    if dp is not None and ad_values:
        ad_sum = int(sum(float(value) for value in ad_values))
        allowed_delta = max(2, int(round(dp * qc_config.dp_ad_mismatch_fraction)))
        if abs(dp - ad_sum) > allowed_delta:
            qc_flags.append("dp_ad_inconsistent")
            notes.append(f"DP {dp} differs from sum(AD) {ad_sum}.")
            score -= 0.08

    if zygosity == "heterozygous":
        called_alt_indices = [
            index
            for index, allele in enumerate(alt_alleles)
            if int(allele_dosage.get(allele, 0)) > 0
        ]
        fractions = [
            _sample_af_for_alt(sample_af, index)
            for index in called_alt_indices
        ]
        fractions = [fraction for fraction in fractions if fraction is not None]
        for fraction in fractions:
            minor_fraction = min(fraction, 1.0 - fraction)
            if minor_fraction < qc_config.heterozygous_balance_severe:
                qc_flags.append("heterozygous_allelic_imbalance_severe")
                notes.append(f"Heterozygous allele balance is severe at alt fraction {fraction:.2f}.")
                score -= 0.22
            elif minor_fraction < qc_config.heterozygous_balance_mild:
                qc_flags.append("heterozygous_allelic_imbalance_mild")
                notes.append(f"Heterozygous allele balance is mildly imbalanced at alt fraction {fraction:.2f}.")
                score -= 0.08

    if zygosity in {"homozygous_alternate", "homozygous_reference"} and sample_af is not None:
        alt_fractions = sample_af if isinstance(sample_af, list) else [sample_af]
        if zygosity == "homozygous_alternate":
            called_fractions = [
                float(alt_fractions[index])
                for index, allele in enumerate(alt_alleles)
                if index < len(alt_fractions) and int(allele_dosage.get(allele, 0)) > 0
            ]
            if called_fractions and min(called_fractions) < (1.0 - qc_config.homozygous_contamination_fraction):
                qc_flags.append("homozygous_alt_has_reference_support")
                notes.append("Homozygous ALT call retains notable reference-read support.")
                score -= 0.12
        if zygosity == "homozygous_reference" and max(float(value) for value in alt_fractions) > qc_config.homozygous_contamination_fraction:
            qc_flags.append("homozygous_ref_has_alt_support")
            notes.append("Homozygous REF call retains notable alternate-read support.")
            score -= 0.12

    if zygosity in {"non_diploid", "hemizygous_reference", "hemizygous_alternate"}:
        qc_flags.append("unexpected_ploidy")
        notes.append(f"GT ploidy produced {zygosity}; diploid trait rules may not apply directly.")
        score -= 0.15

    score = max(0.0, min(1.0, round(score, 3)))
    deduped_flags = _dedupe_text_items(qc_flags)
    qc_flags[:] = deduped_flags

    if not notes:
        notes.append("GT, FILTER, depth, allele balance, and available quality fields support this genotype call.")
    return score, " ".join(notes)


def build_canonical_genotype(
    row: pd.Series | dict[str, Any],
    *,
    qc_config: GenotypeQCConfig = DEFAULT_GENOTYPE_QC_CONFIG,
) -> dict[str, Any]:
    """Build the canonical sample-level genotype object for one VCF row.

    REF and ALT describe the possible alleles at a site. GT is the authoritative
    sample-level field used here to decode the person's genotype state.
    """
    ref = _clean_vcf_text(row.get("ref") if hasattr(row, "get") else "").upper()
    alt_alleles = _normalize_alt_alleles(row.get("alt_alleles") if hasattr(row, "get") else None)
    if not alt_alleles:
        alt_alleles = _normalize_alt_alleles(row.get("alt") if hasattr(row, "get") else None)

    codes, phased, gt_raw = _extract_gt_codes(row)
    alleles = [_genotype_allele_label(code, ref, alt_alleles) for code in codes]
    zygosity = _classify_zygosity(codes)
    allele_dosage = _allele_dosage_per_alt(codes, alt_alleles)

    ad_values = _parse_numeric_list(row.get("ad") if hasattr(row, "get") else None, as_int=True)
    dp = _safe_int(row.get("dp") if hasattr(row, "get") else None)
    gq = _safe_float(row.get("gq") if hasattr(row, "get") else None)
    qual = _safe_float(row.get("qual") if hasattr(row, "get") else None)
    sample_af, sample_af_source = _compute_sample_af(
        explicit_af=row.get("sample_af") if hasattr(row, "get") else None,
        ad_values=ad_values,
    )

    qc_flags: list[str] = []
    sb_values = _parse_numeric_list(row.get("sb") if hasattr(row, "get") else None, as_int=True)
    f1r2_values = _parse_numeric_list(row.get("f1r2") if hasattr(row, "get") else None, as_int=True)
    f2r1_values = _parse_numeric_list(row.get("f2r1") if hasattr(row, "get") else None, as_int=True)
    strand_bias_summary = _summarize_bias_fields(
        sb_values=sb_values,
        f1r2_values=f1r2_values,
        f2r1_values=f2r1_values,
        qc_flags=qc_flags,
    )
    pl_values = _parse_numeric_list(row.get("pl") if hasattr(row, "get") else None)
    gp_values = _parse_numeric_list(row.get("gp") if hasattr(row, "get") else None)
    pl_or_gp_summary = _summarize_likelihood_support(
        codes=codes,
        alt_alleles=alt_alleles,
        pl_values=pl_values,
        gp_values=gp_values,
        qc_flags=qc_flags,
        qc_config=qc_config,
    )
    filter_status = _format_filter_status(row)
    confidence_score, confidence_explanation = _score_genotype_confidence(
        zygosity=zygosity,
        gt_raw=gt_raw,
        filter_status=filter_status,
        qual=qual,
        dp=dp,
        ad_values=ad_values,
        gq=gq,
        sample_af=sample_af,
        alt_alleles=alt_alleles,
        allele_dosage=allele_dosage,
        qc_flags=qc_flags,
        qc_config=qc_config,
    )

    return {
        "sample": _clean_vcf_text(row.get("sample") if hasattr(row, "get") else ""),
        "rsid": _format_variant_label(row.get("id") if hasattr(row, "get") else None),
        "chrom": _clean_vcf_text(row.get("chrom") if hasattr(row, "get") else ""),
        "pos": row.get("pos") if hasattr(row, "get") else None,
        "ref": ref,
        "alt": _format_alt_alleles(alt_alleles),
        "alt_alleles": alt_alleles,
        "gt_raw": gt_raw or "./.",
        "phased": phased,
        "genotype_alleles": alleles,
        "genotype": _format_genotype_display(alleles, phased=phased),
        "zygosity": zygosity,
        "allele_dosage_per_alt": allele_dosage,
        "filter_status": filter_status,
        "qual": qual,
        "dp": dp,
        "ad": ad_values,
        "sample_af": sample_af,
        "sample_af_source": sample_af_source,
        "gq": gq,
        "pl_or_gp_summary": pl_or_gp_summary,
        "sb": sb_values,
        "f1r2": f1r2_values,
        "f2r1": f2r1_values,
        "strand_bias_summary": strand_bias_summary,
        "qc_flags": qc_flags,
        "confidence_score": confidence_score,
        "confidence_explanation": confidence_explanation,
    }


def _ensure_variant_genotype_annotations(variants: pd.DataFrame) -> pd.DataFrame:
    """Ensure a variant table contains canonical genotype/QC output columns."""
    annotated = variants.copy()
    if annotated.empty:
        for column in GENOTYPE_OUTPUT_COLUMNS:
            if column not in annotated.columns:
                annotated[column] = pd.Series(dtype="object")
        return annotated

    genotype_rows = [build_canonical_genotype(row) for _, row in annotated.iterrows()]
    for column in GENOTYPE_OUTPUT_COLUMNS:
        annotated[column] = [genotype.get(column) for genotype in genotype_rows]
    return annotated


def _empty_variant_dataframe() -> pd.DataFrame:
    """Return an empty variant table with the columns downstream reports expect."""
    dtype_by_column = {
        "pos": "int64",
        "qual": "float64",
        "filter_pass": "bool",
    }
    df = pd.DataFrame(
        {
            column: pd.Series(dtype=dtype_by_column.get(column, "object"))
            for column in BASE_VARIANT_COLUMNS
        }
    )
    return _ensure_variant_genotype_annotations(df)


def _open_text_or_gzip(path: str | Path):
    """Open plain VCF or gzip/bgzip VCF as text."""
    path_text = str(path)
    if path_text.lower().endswith(".gz"):
        return gzip.open(path_text, "rt", encoding="utf-8", errors="replace")
    return open(path_text, "rt", encoding="utf-8", errors="replace")


def _sample_field_lookup_key(
    *,
    chrom: Any,
    pos: Any,
    ref: Any,
    alt: Any,
    sample: Any,
) -> tuple[str, int, str, str, str]:
    """Build a stable key for overlaying raw FORMAT/sample fields."""
    chrom_text = _clean_vcf_text(chrom).removeprefix("chr")
    pos_value = _safe_int(pos) or 0
    return (
        chrom_text,
        pos_value,
        _clean_vcf_text(ref).upper(),
        _clean_vcf_text(alt).upper(),
        _clean_vcf_text(sample),
    )


def _load_raw_vcf_sample_fields(vcf_path: str, region: str) -> dict[tuple[str, int, str, str, str], dict[str, Any]]:
    """Read raw FORMAT/sample strings so GT phasing and sample AF fields are preserved."""
    try:
        parsed_region = _parse_region_string(region)
    except AnalysisError:
        return {}

    target_chrom = str(parsed_region["chrom"]).removeprefix("chr")
    target_start = int(parsed_region["start"])
    target_end = int(parsed_region["end"])
    sample_names: list[str] = []
    raw_rows: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}

    try:
        with _open_text_or_gzip(vcf_path) as handle:
            for line in handle:
                if not line:
                    continue
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    header = line.rstrip("\n").split("\t")
                    sample_names = header[9:]
                    continue
                if line.startswith("#"):
                    continue

                parts = line.rstrip("\n").split("\t")
                if len(parts) < 8:
                    continue
                chrom, pos_text, raw_id, ref, alt, qual, filter_status, _info = parts[:8]
                chrom_normalized = chrom.removeprefix("chr")
                pos = _safe_int(pos_text)
                if chrom_normalized != target_chrom or pos is None or pos < target_start or pos > target_end:
                    continue
                if len(parts) < 10 or not sample_names:
                    continue

                format_keys = parts[8].split(":")
                for sample_name, sample_value in zip(sample_names, parts[9:]):
                    sample_values = sample_value.split(":")
                    sample_map = dict(zip(format_keys, sample_values))
                    raw_rows[
                        _sample_field_lookup_key(
                            chrom=chrom,
                            pos=pos,
                            ref=ref,
                            alt=alt,
                            sample=sample_name,
                        )
                    ] = {
                        "id": None if raw_id in {"", "."} else raw_id,
                        "filter_status": "PASS" if filter_status in {"", ".", "PASS"} else filter_status,
                        "gt_raw": sample_map.get("GT", "./."),
                        "ad": sample_map.get("AD"),
                        "dp": sample_map.get("DP"),
                        "gq": sample_map.get("GQ"),
                        "pl": sample_map.get("PL"),
                        "gp": sample_map.get("GP"),
                        "sample_af": sample_map.get("AF"),
                        "sb": sample_map.get("SB"),
                        "f1r2": sample_map.get("F1R2"),
                        "f2r1": sample_map.get("F2R1"),
                    }
    except OSError:
        return {}

    return raw_rows


def _normalize_lookup_key(value: str) -> str:
    """Canonicalize lookup keys so IDs and coordinate aliases can match reliably."""
    normalized = value.strip().lower().replace(" ", "")
    if normalized.startswith("chr"):
        normalized = normalized[3:]
    return normalized


def _normalize_allele_change(value: Any) -> str:
    """Canonicalize allele-change strings such as ``A -> G`` and ``A>G``."""
    text = str(value or "").strip().upper().replace(" ", "")
    return text.replace("->", ">")


def _extract_alt_allele_from_change(value: Any) -> str:
    """Return the ALT allele from a display change such as ``A -> G`` when available."""
    normalized = _normalize_allele_change(value)
    if ">" not in normalized:
        return ""
    return normalized.rsplit(">", 1)[-1].strip()


def _parse_region_string(region: str) -> dict[str, Any]:
    """Parse a ``chr:start-end`` style region string into normalized components."""
    cleaned_region = region.strip().replace(",", "")
    match = re.fullmatch(r"(?:chr)?(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)", cleaned_region)
    if match is None:
        raise AnalysisError(
            f"Unsupported region format '{region}'. Use chr:start-end, for example 11:637269-640706."
        )

    start = int(match.group("start"))
    end = int(match.group("end"))
    if start > end:
        raise AnalysisError(f"Invalid region '{region}': start must be <= end.")

    return {
        "chrom": match.group("chrom"),
        "start": start,
        "end": end,
    }


def _format_interval(chrom: str, start: int, end: int) -> str:
    """Format a genomic interval for human-readable summaries."""
    return f"chr{chrom}:{start:,}-{end:,}"


def _format_plain_interval_union(chrom: str, records: Iterable[dict[str, Any]]) -> str:
    """Format the coordinate union for interval records without display commas."""
    intervals = [
        (int(record["start"]), int(record["end"]))
        for record in records
        if record and record.get("start") is not None and record.get("end") is not None
    ]
    if not intervals:
        return ""
    start = min(min(start, end) for start, end in intervals)
    end = max(max(start, end) for start, end in intervals)
    return f"{str(chrom).removeprefix('chr')}:{start}-{end}"


def _region_text_covers_records(region: str, records: Iterable[dict[str, Any]], *, chrom: str) -> bool:
    """Return whether a plain region string covers all supplied interval records."""
    try:
        parsed_region = _parse_region_string(region)
    except AnalysisError:
        return False
    if str(parsed_region["chrom"]).removeprefix("chr") != str(chrom).removeprefix("chr"):
        return False
    for record in records:
        record_start = int(record["start"])
        record_end = int(record["end"])
        if min(record_start, record_end) < int(parsed_region["start"]):
            return False
        if max(record_start, record_end) > int(parsed_region["end"]):
            return False
    return True


def _format_point(chrom: str, position: int) -> str:
    """Format a single genomic coordinate for human-readable summaries."""
    return f"chr{chrom}:{position:,}"


def _format_frequency(value: float | None) -> str:
    """Render an allele frequency as a percentage string for the UI."""
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether two intervals on the same chromosome overlap."""
    first_chrom = str(first["chrom"]).removeprefix("chr")
    second_chrom = str(second["chrom"]).removeprefix("chr")
    if first_chrom != second_chrom:
        return False
    return first["start"] <= second["end"] and second["start"] <= first["end"]


def _build_interval_record(label: str, chrom: str, start: int, end: int, definition: str) -> dict[str, Any]:
    """Create a normalized interval record used in the UI summaries."""
    return {
        "label": label,
        "chrom": chrom,
        "start": start,
        "end": end,
        "length_bp": end - start + 1,
        "display": _format_interval(chrom, start, end),
        "definition": definition,
    }


def _build_variant_lookup_keys(row: pd.Series) -> set[str]:
    """Create rsID and coordinate aliases for a single variant row."""
    keys: set[str] = set()

    raw_id = row.get("id")
    if pd.notna(raw_id) and str(raw_id).strip() and str(raw_id).strip() != ".":
        keys.update(_normalize_lookup_key(part) for part in str(raw_id).split(";") if part.strip())

    chrom = str(row.get("chrom", "")).strip()
    pos = row.get("pos")
    ref = str(row.get("ref", "") or "").strip().upper()
    alt_alleles = _normalize_alt_alleles(row.get("alt_alleles"))
    if not alt_alleles:
        alt_alleles = _normalize_alt_alleles(row.get("alt"))

    if chrom and pd.notna(pos):
        chrom = chrom[3:] if chrom.lower().startswith("chr") else chrom
        pos_text = str(int(pos))
        keys.add(_normalize_lookup_key(f"{chrom}:{pos_text}"))

        if ref and ref not in {"NAN", "NONE"}:
            for alt in alt_alleles:
                keys.add(_normalize_lookup_key(f"{chrom}:{pos_text}:{ref}>{alt}"))

    return keys


def _format_variant_display(row: pd.Series) -> str:
    """Render a human-readable label for a variant row in the UI."""
    raw_id = row.get("id")
    if pd.notna(raw_id) and str(raw_id).strip() and str(raw_id).strip() != ".":
        return str(raw_id)
    return f"{row['chrom']}:{int(row['pos'])} {row['ref']}>{row['alt']}"


def _format_variant_label(value: Any) -> str:
    """Render a compact variant label while preserving explicit missing IDs as ``None``."""
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text or text == ".":
        return "None"
    return text


def _format_variant_change(ref: Any, alt: Any) -> str:
    """Render the site-level REF -> ALT definition, not the person's genotype."""
    ref_text = "" if _is_missing_value(ref) else str(ref).strip()
    alt_text = "" if _is_missing_value(alt) else str(alt).strip()
    if not ref_text and not alt_text:
        return "Unavailable"
    if not ref_text or not alt_text:
        return f"{ref_text or '?'} -> {alt_text or '?'}"
    return f"{ref_text} -> {alt_text}"


def _dedupe_text_items(items: list[str]) -> list[str]:
    """Return non-empty text items while preserving the first occurrence order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        normalized = text.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(text)
    return deduped


def _build_specific_variant_link_summary(
    record: dict[str, Any] | None,
    *,
    fallback: str,
) -> str:
    """Return a compact, variant-specific association summary for sample tables."""
    if not record:
        return fallback

    link_items: list[str] = []
    usual_variant_note = str(record.get("usual_variant_note", "")).strip()
    common_name = str(record.get("common_name", "")).strip()
    if usual_variant_note:
        link_items.append(usual_variant_note)
    if common_name:
        link_items.append(common_name)

    associated_conditions = [str(item).strip() for item in record.get("associated_conditions", []) if str(item).strip()]
    if associated_conditions:
        link_items.append("Studied in " + ", ".join(associated_conditions))

    literature_findings = record.get("literature_findings", [])
    if not associated_conditions and literature_findings:
        first_finding = literature_findings[0]
        phenotype = str(first_finding.get("phenotype", "")).strip()
        if phenotype:
            link_items.append(phenotype)

    functional_effects = [str(item).strip() for item in record.get("functional_effects", []) if str(item).strip()]
    if not associated_conditions and not literature_findings and functional_effects:
        link_items.append(functional_effects[0])

    deduped_items = _dedupe_text_items(link_items)
    if deduped_items:
        return "; ".join(deduped_items[:2])

    clinical_significance = str(record.get("clinical_significance", "")).strip()
    if clinical_significance:
        return clinical_significance

    clinical_interpretation = str(record.get("clinical_interpretation", "")).strip()
    if clinical_interpretation:
        return clinical_interpretation

    return fallback


def annotate_known_variant_ids(
    variants: pd.DataFrame,
    knowledge_base: dict[str, Any] | None,
) -> pd.DataFrame:
    """Fill display-friendly IDs when the curated bundle can name an unlabeled variant."""
    labeled = _ensure_variant_genotype_annotations(variants)
    if labeled.empty:
        if "id_source" not in labeled.columns:
            labeled["id_source"] = pd.Series(dtype="object")
        return labeled

    variant_records = knowledge_base.get("variant_records", []) if knowledge_base else []
    resolved_ids: list[Any] = []
    id_sources: list[str] = []

    for _, row in labeled.iterrows():
        raw_id = row.get("id")
        raw_id_text = "" if pd.isna(raw_id) else str(raw_id).strip()
        matched_record = _match_variant_record(row, variant_records) if variant_records else None

        if raw_id_text and raw_id_text != ".":
            resolved_ids.append(raw_id_text)
            id_sources.append("Source VCF")
            continue

        if matched_record is not None:
            resolved_ids.append(
                str(matched_record.get("display_name", matched_record["variant"])).strip()
            )
            id_sources.append("Knowledge base match")
            continue

        resolved_ids.append(None)
        id_sources.append("Unlabeled in source VCF")

    labeled["id"] = resolved_ids
    labeled["id_source"] = id_sources
    return labeled


def _match_variant_record(row: pd.Series, variant_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first curated record that matches the observed variant row."""
    lookup_keys = _build_variant_lookup_keys(row)
    for record in variant_records:
        record_keys = {
            _normalize_lookup_key(candidate)
            for candidate in record.get("lookup_keys", [])
            if candidate
        }
        if lookup_keys & record_keys:
            return record
    return None


def _variant_record_text_candidates(record: dict[str, Any]) -> list[str]:
    """Collect searchable marker text without assuming every record uses the same schema."""
    candidates: list[str] = []
    for key in (
        "variant",
        "display_name",
        "common_name",
        "usual_variant_note",
        "clinical_significance",
        "interpretation_scope",
    ):
        value = str(record.get(key, "")).strip()
        if value:
            candidates.append(value)
    candidates.extend(str(value).strip() for value in record.get("lookup_keys", []) if str(value).strip())
    return candidates


def _extract_record_rsids(record: dict[str, Any]) -> list[str]:
    """Return stable rsID aliases mentioned by a curated marker record."""
    rsids: list[str] = []
    seen: set[str] = set()
    for text in _variant_record_text_candidates(record):
        for match in re.finditer(r"\brs\d+\b", text, flags=re.IGNORECASE):
            rsid = match.group(0).lower()
            if rsid in seen:
                continue
            seen.add(rsid)
            rsids.append(match.group(0))
    return rsids


def _dedupe_record_changes(changes: Iterable[str]) -> list[str]:
    """Return unique nucleotide or protein change strings while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for change in changes:
        text = str(change).strip().rstrip(".,;")
        if not text:
            continue
        normalized = text.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(text)
    return deduped


def _extract_transcript_nucleotide_changes(record: dict[str, Any]) -> list[str]:
    """Extract transcript or mitochondrial nucleotide changes such as c.76-2A>C."""
    changes: list[str] = []
    pattern = re.compile(
        r"(?P<change>[cm]\.[*+-]?\d+(?:[+-]\d+)?(?:[ACGT]+>[ACGT]+|delins[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)?)",
        flags=re.IGNORECASE,
    )
    for text in _variant_record_text_candidates(record):
        changes.extend(match.group("change") for match in pattern.finditer(text))
    return _dedupe_record_changes(changes)


def _extract_protein_changes(record: dict[str, Any]) -> list[str]:
    """Extract protein-level aliases such as p.Arg37Ser when bundled."""
    changes: list[str] = []
    pattern = re.compile(
        r"(?P<change>p\.[A-Za-z]{1,3}\d+(?:[A-Za-z]{1,3}|Ter|\*|X))",
        flags=re.IGNORECASE,
    )
    for text in _variant_record_text_candidates(record):
        changes.extend(match.group("change") for match in pattern.finditer(text))
    return _dedupe_record_changes(changes)


def _extract_record_genomic_changes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract exact genomic REF/ALT aliases from curated lookup keys when available."""
    changes: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, str, str]] = set()
    default_chrom = str(record.get("chromosome", "")).strip().removeprefix("chr")
    coordinate_pattern = re.compile(
        r"^(?:chr)?(?P<chrom>[0-9XYM]+|MT):(?P<position>\d+):(?P<ref>[ACGTN]+)>(?P<alt>[ACGTN]+)$",
        flags=re.IGNORECASE,
    )
    genomic_hgvs_pattern = re.compile(
        r"(?:^|[:(])g\.(?P<position>\d+)(?P<ref>[ACGTN]+)>(?P<alt>[ACGTN]+)",
        flags=re.IGNORECASE,
    )

    for text in _variant_record_text_candidates(record):
        candidate = text.strip()
        match = coordinate_pattern.fullmatch(candidate)
        chrom = ""
        position: int | None = None
        ref = ""
        alt = ""
        if match:
            chrom = match.group("chrom").removeprefix("chr")
            position = int(match.group("position"))
            ref = match.group("ref").upper()
            alt = match.group("alt").upper()
        else:
            match = genomic_hgvs_pattern.search(candidate)
            if match:
                chrom = default_chrom
                position = int(match.group("position"))
                ref = match.group("ref").upper()
                alt = match.group("alt").upper()
        if not ref or not alt:
            continue

        dedupe_key = (chrom or default_chrom, position, ref, alt)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        changes.append(
            {
                "chromosome": chrom or default_chrom,
                "position": position,
                "reference_allele": ref,
                "alternate_allele": alt,
                "change": f"{ref}>{alt}",
            }
        )

    return changes


def _extract_alleles_from_nucleotide_changes(changes: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return simple REF/ALT alleles from change strings where that is unambiguous."""
    refs: list[str] = []
    alts: list[str] = []
    for change in changes:
        match = re.search(r"(?P<ref>[ACGT]+)>(?P<alt>[ACGT]+)$", str(change).strip(), flags=re.IGNORECASE)
        if not match:
            continue
        refs.append(match.group("ref").upper())
        alts.append(match.group("alt").upper())
    return _dedupe_record_changes(refs), _dedupe_record_changes(alts)


def _build_marker_type(record: dict[str, Any], genomic_changes: list[dict[str, Any]]) -> str:
    """Classify a curated marker for compact catalog display."""
    marker_text = " ".join(_variant_record_text_candidates(record)).casefold()
    rsids = _extract_record_rsids(record)
    if "copy-number" in marker_text or "copy number" in marker_text or "cnv" in marker_text:
        return "Copy-number / structural marker"
    if "deletion" in marker_text or "duplication" in marker_text:
        return "Deletion / duplication marker"
    if genomic_changes:
        return "Single-nucleotide variant"
    if rsids and record.get("position") is not None:
        return "Single-nucleotide marker (alleles not bundled)"
    if "model" in marker_text:
        return "Functional or haplotype model"
    return "Curated sequence marker"


def _format_marker_genome_location(
    record: dict[str, Any],
    *,
    assembly: str,
    genomic_changes: list[dict[str, Any]],
) -> str:
    """Return the genome location for a curated marker, with explicit uncertainty when needed."""
    if genomic_changes:
        loci = []
        for change in genomic_changes:
            chrom = str(change.get("chromosome", record.get("chromosome", ""))).removeprefix("chr")
            position = change.get("position")
            if chrom and position is not None:
                loci.append(f"{assembly} {_format_point(chrom, int(position))}")
        loci = _dedupe_record_changes(loci)
        if loci:
            return "; ".join(loci)

    chrom = str(record.get("chromosome", "")).strip().removeprefix("chr")
    position = record.get("position")
    if chrom and position is not None and not pd.isna(position):
        return f"{assembly} {_format_point(chrom, int(position))}"
    if chrom:
        return f"{assembly} chr{chrom}; exact single-base coordinate not bundled"
    return f"{assembly}; genome location not bundled"


def _build_marker_clinical_parameter_summary(record: dict[str, Any], marker_type: str) -> str:
    """Condense clinically relevant marker parameters for table/report display."""
    parts: list[str] = []
    clinical_significance = str(record.get("clinical_significance", "")).strip()
    interpretation_scope = str(record.get("interpretation_scope", "")).strip()
    conditions = _dedupe_text_items([str(item) for item in record.get("associated_conditions", [])])
    functional_effects = _dedupe_text_items([str(item) for item in record.get("functional_effects", [])])
    probe_ids = _dedupe_text_items([str(item) for item in record.get("relevant_methylation_probe_ids", [])])

    if clinical_significance:
        parts.append(clinical_significance)
    if interpretation_scope:
        parts.append(f"Scope: {interpretation_scope}.")
    if marker_type:
        parts.append(f"Type: {marker_type}.")
    if conditions:
        parts.append(f"Studied conditions: {'; '.join(conditions[:4])}.")
    if functional_effects:
        parts.append(f"Functional themes: {'; '.join(functional_effects[:3])}.")
    if probe_ids:
        parts.append(f"Methylation probes: {', '.join(probe_ids[:6])}.")
    if not record.get("is_assayable_in_snp_vcf", True):
        parts.append("Assay note: not directly called by the SNP-oriented VCF preview.")
    return " ".join(parts).strip()


def _build_curated_marker_metadata(
    record: dict[str, Any],
    *,
    assembly: str = "GRCh37 / hg19",
) -> dict[str, Any]:
    """Build exact-position, allele, clinical, and research-link metadata for a marker."""
    genomic_changes = _extract_record_genomic_changes(record)
    transcript_changes = _extract_transcript_nucleotide_changes(record)
    protein_changes = _extract_protein_changes(record)
    rsids = _extract_record_rsids(record)
    marker_type = _build_marker_type(record, genomic_changes)

    if genomic_changes:
        nucleotide_changes = _dedupe_record_changes(change["change"] for change in genomic_changes)
        nucleotide_change = "; ".join(nucleotide_changes)
        refs = _dedupe_record_changes(change["reference_allele"] for change in genomic_changes)
        alts = _dedupe_record_changes(change["alternate_allele"] for change in genomic_changes)
        nucleotide_change_basis = "genomic REF/ALT alias"
    elif transcript_changes:
        nucleotide_change = "; ".join(transcript_changes)
        refs, alts = _extract_alleles_from_nucleotide_changes(transcript_changes)
        nucleotide_change_basis = "transcript or mitochondrial HGVS"
    elif rsids and record.get("position") is not None:
        nucleotide_change = "Exact REF/ALT not bundled for this rsID"
        refs = []
        alts = []
        nucleotide_change_basis = "rsID and coordinate marker; exact REF/ALT not bundled locally"
    else:
        nucleotide_change = "Not a bundled single-nucleotide change"
        refs = []
        alts = []
        nucleotide_change_basis = "structural, haplotype, or model marker"

    assayability = (
        "Directly assayable in SNP-oriented VCF preview"
        if record.get("is_assayable_in_snp_vcf", True)
        else "Not directly called by the current SNP-oriented VCF preview"
    )
    clinical_parameter_summary = _build_marker_clinical_parameter_summary(record, marker_type)

    return {
        "genome_assembly": assembly,
        "genome_location": _format_marker_genome_location(
            record,
            assembly=assembly,
            genomic_changes=genomic_changes,
        ),
        "genomic_changes": genomic_changes,
        "genomic_change": "; ".join(change["change"] for change in genomic_changes) or "Not bundled",
        "nucleotide_change": nucleotide_change,
        "nucleotide_change_basis": nucleotide_change_basis,
        "reference_allele": "/".join(refs) if refs else "Not specified",
        "alternate_allele": "/".join(alts) if alts else "Not specified",
        "coding_change": "; ".join(transcript_changes) if transcript_changes else "Not bundled",
        "protein_change": "; ".join(protein_changes) if protein_changes else "Not bundled",
        "rsids": rsids,
        "marker_type": marker_type,
        "assayability": assayability,
        "clinical_parameter_summary": clinical_parameter_summary,
        "research_links": _collect_variant_record_papers(record),
    }


def _build_known_variant_summary(
    record: dict[str, Any],
    *,
    assembly: str = "GRCh37 / hg19",
) -> dict[str, Any]:
    """Project a curated database record into a compact UI-friendly summary."""
    position = record.get("position")
    chrom = record.get("chromosome", "11")
    assay_note = None
    if not record.get("is_assayable_in_snp_vcf", True):
        assay_note = "Not directly called by the current SNP-oriented VCF preview."

    marker_metadata = _build_curated_marker_metadata(record, assembly=assembly)
    return {
        "variant": record.get("display_name", record["variant"]),
        "common_name": record.get("common_name"),
        "position": _format_point(chrom, int(position)) if position is not None else "Repeat / structural locus",
        **marker_metadata,
        "clinical_significance": record.get("clinical_significance", "Clinical significance not specified."),
        "summary": record.get("clinical_interpretation", ""),
        "interpretation_scope": record.get("interpretation_scope", "Research context"),
        "region_class": record.get("region_class", "research_marker"),
        "functional_effects": record.get("functional_effects", []),
        "associated_conditions": record.get("associated_conditions", []),
        "research_context": record.get("research_context", []),
        "literature_findings": _build_literature_findings(record),
        "assay_note": assay_note,
        "evidence": record.get("evidence", []),
    }


def _build_curated_named_marker_catalog(
    variants: pd.DataFrame,
    variant_records: list[dict[str, Any]],
    *,
    assembly: str = "GRCh37 / hg19",
) -> list[dict[str, Any]]:
    """Return all curated named markers with run-specific observation flags."""
    observed_marker_map: dict[str, list[str]] = {}
    for _, row in variants.iterrows():
        matched_record = _match_variant_record(row, variant_records)
        if matched_record is None:
            continue
        observed_marker_map.setdefault(matched_record["variant"], []).append(_format_variant_display(row))

    marker_catalog: list[dict[str, Any]] = []
    for record in variant_records:
        summary = _build_known_variant_summary(record, assembly=assembly)
        observed_variants = observed_marker_map.get(record["variant"], [])
        marker_catalog.append(
            {
                **summary,
                "observed_in_run": bool(observed_variants),
                "observed_variants": observed_variants,
            }
        )
    return marker_catalog


def _build_observed_variant_summary(
    row: pd.Series,
    matched_record: dict[str, Any] | None,
    *,
    gene_name: str = DEFAULT_GENE_NAME,
) -> dict[str, Any]:
    """Render an observed variant plus whatever curated significance is available."""
    chrom = str(row.get("chrom", "")).removeprefix("chr")
    position = int(row["pos"])
    quality = row.get("qual")
    quality_display = f"{float(quality):.2f}" if pd.notna(quality) else "n/a"
    genotype = build_canonical_genotype(row)
    genotype_summary = (
        f"VCF GT {genotype['gt_raw']} decodes to {genotype['genotype']} "
        f"({genotype['zygosity']}); ALT dosage { _format_allele_dosage(genotype['allele_dosage_per_alt']) }."
    )

    if matched_record is None:
        return {
            "display": _format_variant_display(row),
            "variant_label": _format_variant_label(row.get("id")),
            "change": _format_variant_change(row.get("ref"), row.get("alt")),
            "site_definition": _format_variant_change(row.get("ref"), row.get("alt")),
            "genotype": genotype["genotype"],
            "gt_raw": genotype["gt_raw"],
            "phased": genotype["phased"],
            "genotype_alleles": genotype["genotype_alleles"],
            "zygosity": genotype["zygosity"],
            "allele_dosage_per_alt": genotype["allele_dosage_per_alt"],
            "allele_dosage": _format_allele_dosage(genotype["allele_dosage_per_alt"]),
            "filter_status": genotype["filter_status"],
            "dp": genotype["dp"],
            "ad": genotype["ad"],
            "sample_af": genotype["sample_af"],
            "gq": genotype["gq"],
            "pl_or_gp_summary": genotype["pl_or_gp_summary"],
            "qc_flags": genotype["qc_flags"],
            "confidence_score": genotype["confidence_score"],
            "confidence_explanation": genotype["confidence_explanation"],
            "linked_to": f"No curated local {gene_name} link is bundled for this VCF call yet.",
            "position": _format_point(chrom, position),
            "quality": quality_display,
            "matched_variant": None,
            "clinical_significance": (
                f"No curated clinical significance available in the local {gene_name} database for this observed site."
            ),
            "summary": (
                f"Observed inside the selected {gene_name} search window, but not one of the seeded "
                f"common {gene_name} promoter or gene variants. {genotype_summary}"
            ),
            "functional_effects": [],
            "associated_conditions": [],
            "research_context": [],
            "literature_findings": [],
            "evidence": [],
        }

    return {
        "display": _format_variant_display(row),
        "variant_label": _format_variant_label(row.get("id")),
        "change": _format_variant_change(row.get("ref"), row.get("alt")),
        "site_definition": _format_variant_change(row.get("ref"), row.get("alt")),
        "genotype": genotype["genotype"],
        "gt_raw": genotype["gt_raw"],
        "phased": genotype["phased"],
        "genotype_alleles": genotype["genotype_alleles"],
        "zygosity": genotype["zygosity"],
        "allele_dosage_per_alt": genotype["allele_dosage_per_alt"],
        "allele_dosage": _format_allele_dosage(genotype["allele_dosage_per_alt"]),
        "filter_status": genotype["filter_status"],
        "dp": genotype["dp"],
        "ad": genotype["ad"],
        "sample_af": genotype["sample_af"],
        "gq": genotype["gq"],
        "pl_or_gp_summary": genotype["pl_or_gp_summary"],
        "qc_flags": genotype["qc_flags"],
        "confidence_score": genotype["confidence_score"],
        "confidence_explanation": genotype["confidence_explanation"],
        "linked_to": _build_specific_variant_link_summary(
            matched_record,
            fallback=f"No curated local {gene_name} link is bundled for this VCF call yet.",
        ),
        "position": _format_point(chrom, position),
        "quality": quality_display,
        "matched_variant": matched_record.get("display_name", matched_record["variant"]),
        "clinical_significance": matched_record.get(
            "clinical_significance", "Clinical significance not specified."
        ),
        "summary": f"{matched_record.get('clinical_interpretation', '')} {genotype_summary}".strip(),
        "functional_effects": matched_record.get("functional_effects", []),
        "associated_conditions": matched_record.get("associated_conditions", []),
        "research_context": matched_record.get("research_context", []),
        "literature_findings": _build_literature_findings(matched_record),
        "evidence": matched_record.get("evidence", []),
    }


def _build_literature_findings(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize literature findings so the UI can render paper-level bullets consistently."""
    findings: list[dict[str, Any]] = []
    for finding in record.get("literature_findings", []):
        paper = str(finding.get("paper", "")).strip()
        phenotype = str(finding.get("phenotype", "")).strip()
        finding_text = str(finding.get("finding", "")).strip()
        if not paper or not finding_text:
            continue
        findings.append(
            {
                "paper": paper,
                "genotypes": str(finding.get("genotypes", "")).strip(),
                "phenotype": phenotype,
                "finding": finding_text,
                "url": str(finding.get("url", "")).strip(),
            }
        )
    return findings


def _summarize_observed_region_variants(
    variants: pd.DataFrame,
    variant_records: list[dict[str, Any]],
    *,
    chrom: str,
    start: int,
    end: int,
    gene_name: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Summarize observed VCF calls in a given region for UI display."""
    observed = variants[
        (variants["chrom"].astype(str).str.removeprefix("chr") == chrom)
        & (variants["pos"] >= start)
        & (variants["pos"] <= end)
    ].sort_values("pos")

    summaries: list[dict[str, Any]] = []
    for _, row in observed.head(limit).iterrows():
        summaries.append(
            _build_observed_variant_summary(
                row,
                _match_variant_record(row, variant_records),
                gene_name=gene_name,
            )
        )
    return summaries


def _build_region_variant_analysis(
    *,
    region_record: dict[str, Any],
    search_region: dict[str, Any],
    variants: pd.DataFrame,
    curated_records: list[dict[str, Any]],
    gene_name: str,
    inclusion_hint: str,
    assembly: str = "GRCh37 / hg19",
) -> dict[str, Any]:
    """Build a structured analysis block for promoter or gene-body coverage."""
    included = _intervals_overlap(search_region, region_record)
    found_variants = (
        _summarize_observed_region_variants(
            variants,
            curated_records,
            chrom=region_record["chrom"],
            start=region_record["start"],
            end=region_record["end"],
            gene_name=gene_name,
        )
        if included
        else []
    )

    if included and found_variants:
        analysis_note = (
            f"The current search window overlaps {region_record['label'].lower()} "
            f"and found {len(found_variants)} VCF call(s) in the first preview slice."
        )
    elif included:
        analysis_note = (
            f"The current search window overlaps {region_record['label'].lower()}, "
            "but no VCF calls from that region were retained in the current preview."
        )
    else:
        analysis_note = (
            f"The current search window does not overlap {region_record['label'].lower()}. "
            f"{inclusion_hint}"
        )

    return {
        "label": region_record["label"],
        "window": region_record["display"],
        "length_bp": region_record["length_bp"],
        "definition": region_record["definition"],
        "included": included,
        "analysis_note": analysis_note,
        "found_variant_count": len(found_variants),
        "found_variants": found_variants,
        "known_variants": [
            _build_known_variant_summary(record, assembly=assembly)
            for record in curated_records
        ],
    }


def _build_population_frequency_rows(
    entries: list[dict[str, Any]],
    *,
    focus_alleles: list[str],
    effect_allele: str | None,
) -> list[dict[str, Any]]:
    """Normalize allele-frequency rows so the template can render them directly."""
    normalized_rows: list[dict[str, Any]] = []
    for entry in entries:
        allele_frequencies = entry.get("allele_frequencies", {})
        normalized_rows.append(
            {
                "population_code": entry.get("population_code"),
                "location_group": entry.get("location_group", "Unspecified"),
                "label": entry.get("label", entry.get("population_code", "Population")),
                "granularity": entry.get("granularity", "reference"),
                "effect_allele": effect_allele,
                "effect_allele_frequency": allele_frequencies.get(effect_allele) if effect_allele else None,
                "effect_allele_display": (
                    _format_frequency(allele_frequencies.get(effect_allele))
                    if effect_allele
                    else "n/a"
                ),
                "allele_frequencies": [
                    {
                        "allele": allele,
                        "frequency": allele_frequencies.get(allele),
                        "display": _format_frequency(allele_frequencies.get(allele)),
                    }
                    for allele in focus_alleles
                    if allele in allele_frequencies
                ],
            }
        )
    return normalized_rows


def _group_population_rows_by_location(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group population rows by geography for collapsible UI sections."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["location_group"], []).append(row)

    return [
        {
            "location_group": location_group,
            "entries": sorted(entries, key=lambda item: item["label"]),
        }
        for location_group, entries in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _summarize_population_extremes(
    rows: list[dict[str, Any]], effect_allele: str | None
) -> dict[str, Any] | None:
    """Find the highest and lowest superpopulation frequencies for a focal allele."""
    if not effect_allele:
        return None

    comparable_rows = [
        row
        for row in rows
        if row.get("granularity") == "superpopulation" and row.get("effect_allele_frequency") is not None
    ]
    if not comparable_rows:
        return None

    highest = max(comparable_rows, key=lambda item: item["effect_allele_frequency"])
    lowest = min(comparable_rows, key=lambda item: item["effect_allele_frequency"])
    return {
        "effect_allele": effect_allele,
        "highest": highest,
        "lowest": lowest,
        "summary": (
            f"The {effect_allele} allele is most frequent in {highest['location_group']} "
            f"({highest['effect_allele_display']}) and least frequent in {lowest['location_group']} "
            f"({lowest['effect_allele_display']}) across the 1000 Genomes superpopulation layer."
        ),
    }


def _build_sample_variant_highlights(
    *,
    matched_records: list[dict[str, Any]],
    promoter_analysis: dict[str, Any],
    gene_analysis: dict[str, Any],
    gene_name: str,
) -> dict[str, Any]:
    """Summarize the most visible sample-specific variant findings first."""
    if matched_records:
        summary = (
            f"This sample matched {len(matched_records)} curated {gene_name} marker site(s). "
            "Those direct sample hits are shown first below with decoded GT so homozygous-reference, heterozygous, and homozygous-alternate states stay separate."
        )
        items = [
            {
                "title": record["variant"],
                "observed_variant": record["observed_variant"],
                "change": record.get("change", "Unavailable"),
                "site_definition": record.get("site_definition", record.get("change", "Unavailable")),
                "genotype": record.get("genotype", "Unavailable"),
                "gt_raw": record.get("gt_raw", "./."),
                "zygosity": record.get("zygosity", "missing"),
                "allele_dosage": record.get("allele_dosage", "Unavailable"),
                "confidence_score": record.get("confidence_score", 0),
                "confidence_explanation": record.get("confidence_explanation", ""),
                "qc_flags": record.get("qc_flags", []),
                "category": record.get("interpretation_scope", "Research context"),
                "description": (
                    f"{record.get('clinical_interpretation', '')} "
                    f"Genotype: {record.get('genotype', 'Unavailable')} "
                    f"({record.get('zygosity', 'missing')}); confidence "
                    f"{record.get('confidence_score', 0)}."
                ).strip(),
                "conditions": record.get("associated_conditions", []),
                "literature_findings": record.get("literature_findings", []),
            }
            for record in matched_records
        ]
        return {
            "summary": summary,
            "highlight_items": items,
            "result_table_rows": [
                {
                    "variant_label": record.get("variant_label", "None"),
                    "change": record.get("change", "Unavailable"),
                    "genotype": record.get("genotype", "Unavailable"),
                    "zygosity": record.get("zygosity", "missing"),
                    "allele_dosage": record.get("allele_dosage", "Unavailable"),
                    "confidence_score": record.get("confidence_score", 0),
                    "qc_flags": record.get("qc_flags", []),
                    "linked_to": record.get("linked_to", ""),
                }
                for record in matched_records
            ],
        }

    found_items: list[dict[str, Any]] = []
    result_table_rows: list[dict[str, str]] = []
    for region_label, region_analysis in (
        ("Promoter review", promoter_analysis),
        ("Gene body review", gene_analysis),
    ):
        for record in region_analysis.get("found_variants", []):
            found_items.append(
                {
                    "title": record["display"],
                    "observed_variant": record["position"],
                    "change": record.get("change", "Unavailable"),
                    "genotype": record.get("genotype", "Unavailable"),
                    "gt_raw": record.get("gt_raw", "./."),
                    "zygosity": record.get("zygosity", "missing"),
                    "allele_dosage": record.get("allele_dosage", "Unavailable"),
                    "confidence_score": record.get("confidence_score", 0),
                    "confidence_explanation": record.get("confidence_explanation", ""),
                    "qc_flags": record.get("qc_flags", []),
                    "category": region_label,
                    "description": record.get("summary", ""),
                    "conditions": record.get("associated_conditions", []),
                    "literature_findings": record.get("literature_findings", []),
                }
            )
            result_table_rows.append(
                {
                    "variant_label": record.get("variant_label", "None"),
                    "change": record.get("change", "Unavailable"),
                    "genotype": record.get("genotype", "Unavailable"),
                    "zygosity": record.get("zygosity", "missing"),
                    "allele_dosage": record.get("allele_dosage", "Unavailable"),
                    "confidence_score": record.get("confidence_score", 0),
                    "qc_flags": record.get("qc_flags", []),
                    "linked_to": record.get("linked_to", ""),
                }
            )

    if found_items:
        summary = (
            f"This sample did not hit one of the curated named {gene_name} markers, but it did contain "
            f"{len(found_items)} observed VCF call(s) inside the reviewed promoter or gene intervals."
        )
    else:
        summary = (
            f"This sample did not yield a visible {gene_name} promoter or gene-body VCF call in the current preview slice."
        )

    return {
        "summary": summary,
        "highlight_items": found_items,
        "result_table_rows": result_table_rows,
    }


def _build_region_recommendations(
    promoter_region_record: dict[str, Any],
    gene_region_record: dict[str, Any],
    combined_region: str,
    *,
    gene_name: str,
) -> list[dict[str, str]]:
    """Return practical region-span recommendations for common DRD4 review goals."""
    _ = combined_region
    return [
        {
            "title": "Promoter only",
            "region": promoter_region_record["display"],
            "purpose": (
                "Use this when you want to focus on the upstream promoter-review window "
                f"and the classically studied {gene_name} promoter hotspot without loading the transcribed gene body."
            ),
        },
        {
            "title": "Gene body only",
            "region": gene_region_record["display"],
            "purpose": (
                f"Use this when you want the canonical {gene_name} transcribed interval but do not need the upstream promoter review window."
            ),
        },
    ]


def _categorize_beta(mean_beta: float | None) -> str:
    """Map average beta values to a coarse descriptive band for UI summaries."""
    if mean_beta is None or pd.isna(mean_beta):
        return "unavailable"
    if mean_beta < 0.20:
        return "low"
    if mean_beta < 0.60:
        return "intermediate"
    if mean_beta < 0.80:
        return "moderately high"
    return "high"


def _coerce_numeric_beta_values(df: pd.DataFrame) -> pd.Series:
    """Return non-null numeric beta values from a methylation table."""
    if "beta" not in df.columns or df.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df["beta"], errors="coerce").dropna()


def _mean_beta_or_none(beta_values: pd.Series) -> float | None:
    """Return the mean beta value for a numeric series or ``None`` when empty."""
    if beta_values.empty:
        return None
    return float(beta_values.mean())


def _round_beta(mean_beta: float | None) -> float | None:
    """Round a mean beta value for UI display while preserving ``None``."""
    return round(mean_beta, 3) if mean_beta is not None else None


def _build_methylation_metric_rows(
    *,
    gene_name: str,
    whitelist_mean_beta: float | None,
    whitelist_count: int,
    gene_name_mean_beta: float | None,
    gene_name_count: int,
    raw_mean_beta: float | None,
    raw_count: int,
) -> list[dict[str, Any]]:
    """Build table-ready methylation summary rows for the UI and generated reports."""
    return [
        {
            "metric": "Whitelist mean beta",
            "mean_beta": _round_beta(whitelist_mean_beta),
            "mean_beta_display": str(_round_beta(whitelist_mean_beta)) if whitelist_mean_beta is not None else "Unavailable",
            "numeric_values": int(whitelist_count),
            "summary": (
                f"Observed whitelist probe mean across {whitelist_count} numeric value(s)."
                if whitelist_mean_beta is not None
                else "No numeric observed whitelist probe was available in this run."
            ),
        },
        {
            "metric": f"{gene_name}-named row mean beta",
            "mean_beta": _round_beta(gene_name_mean_beta),
            "mean_beta_display": str(_round_beta(gene_name_mean_beta)) if gene_name_mean_beta is not None else "Unavailable",
            "numeric_values": int(gene_name_count),
            "summary": (
                f"Rows where {gene_name} was explicitly annotated in the gene-name columns."
                if gene_name_mean_beta is not None
                else f"No numeric row explicitly annotated {gene_name} in the available gene-name columns."
            ),
        },
        {
            "metric": "All numeric-row mean beta",
            "mean_beta": _round_beta(raw_mean_beta),
            "mean_beta_display": str(_round_beta(raw_mean_beta)) if raw_mean_beta is not None else "Unavailable",
            "numeric_values": int(raw_count),
            "summary": (
                "All numeric beta values across the full raw methylation table."
                if raw_mean_beta is not None
                else "No numeric beta values were available in the raw methylation table."
            ),
        },
    ]


def _select_gene_named_methylation_rows(
    methylation: pd.DataFrame, gene_name: str
) -> tuple[pd.DataFrame, list[str]]:
    """Return rows whose gene-annotation columns explicitly mention ``gene_name``."""
    if methylation.empty:
        return methylation.iloc[0:0].copy(), []

    normalized_gene_name = gene_name.strip().upper()
    if not normalized_gene_name:
        return methylation.iloc[0:0].copy(), []

    annotation_columns = [
        column
        for column in ("GencodeBasicV12_NAME", "gene", "UCSC_RefGene_Name")
        if column in methylation.columns
    ]
    if not annotation_columns:
        return methylation.iloc[0:0].copy(), []

    match_pattern = rf"(?:^|;)\s*{re.escape(normalized_gene_name)}\s*(?:;|$)"
    combined_mask = pd.Series(False, index=methylation.index, dtype="bool")
    matched_columns: list[str] = []

    for column in annotation_columns:
        mask = (
            methylation[column]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.contains(match_pattern, regex=True, na=False)
        )
        if bool(mask.any()):
            matched_columns.append(column)
            combined_mask = combined_mask | mask

    return methylation.loc[combined_mask].copy(), matched_columns


def _build_whitelist_explanation(gene_name: str, relevant_probe_ids: list[str]) -> str:
    """Explain how the curated methylation whitelist is assembled."""
    if not relevant_probe_ids:
        return (
            f"No curated {gene_name} methylation whitelist is bundled yet, so whitelist-only "
            "statistics are unavailable for this gene."
        )
    return (
        f"The curated {gene_name} methylation whitelist is bundled manually in the local "
        "interpretation database under `relevant_methylation_probe_ids`. It is a literature-guided "
        "hotspot subset used for interpretation, not an automatic list of every probe row in the "
        "current manifest slice."
    )


def _split_semicolon_tokens(value: Any) -> list[str]:
    """Split a semicolon-delimited annotation field into unique non-empty tokens."""
    if value is None or pd.isna(value):
        return []
    tokens = [token.strip() for token in str(value).split(";")]
    unique_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in tokens:
        if not token or token in seen_tokens:
            continue
        seen_tokens.add(token)
        unique_tokens.append(token)
    return unique_tokens


def _load_gene_manifest_probe_lookup(
    gene_name: str, probe_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Load bundled manifest annotations for the requested whitelist probes."""
    if not probe_ids:
        return {}

    try:
        manifest = load_gene_epigenetics_manifest(gene_name)
    except Exception:
        logger.exception("Failed to read bundled manifest subset for %s", gene_name)
        return {}
    if manifest is None:
        manifest_path = GENE_DATA_DIR / _gene_manifest_filename(gene_name)
        logger.warning("Bundled manifest subset is missing for %s: %s", gene_name, manifest_path)
        return {}

    manifest = manifest.rename(columns={"IlmnID": "probe_id", "CHR": "chrom", "MAPINFO": "pos"})
    if "probe_id" not in manifest.columns:
        return {}

    subset = manifest[manifest["probe_id"].isin(set(probe_ids))].copy()
    if subset.empty:
        return {}

    subset = subset.drop_duplicates(subset=["probe_id"], keep="first")
    return {str(row["probe_id"]): row.to_dict() for _, row in subset.iterrows()}


def _format_probe_locus(annotation: dict[str, Any]) -> str | None:
    """Format a probe genomic locus from manifest or observed-row annotations."""
    chrom = annotation.get("chrom")
    pos = annotation.get("pos")
    if chrom is None or pos is None or pd.isna(chrom) or pd.isna(pos):
        return None
    chrom_text = str(chrom).removeprefix("chr")
    try:
        pos_value = int(float(pos))
    except (TypeError, ValueError):
        return None
    return f"chr{chrom_text}:{pos_value:,}"


def _build_nearby_manifest_variant_rows(
    annotation: dict[str, Any],
    *,
    allowed_variant_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return nearby manifest SNP annotations paired with their reported distances."""
    variant_ids = _split_semicolon_tokens(annotation.get("SNP_ID"))
    distance_tokens = _split_semicolon_tokens(annotation.get("SNP_DISTANCE"))
    allowed_lookup = (
        {_normalize_lookup_key(variant_id) for variant_id in allowed_variant_ids if variant_id}
        if allowed_variant_ids is not None
        else None
    )
    nearby_rows: list[dict[str, Any]] = []
    for index, variant_id in enumerate(variant_ids):
        if allowed_lookup is not None and _normalize_lookup_key(variant_id) not in allowed_lookup:
            continue
        distance = distance_tokens[index] if index < len(distance_tokens) else ""
        nearby_rows.append(
            {
                "variant": variant_id,
                "distance": distance,
            }
        )
    return nearby_rows


def _collect_variant_record_papers(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect deduplicated papers and evidence links for one curated variant record."""
    papers: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for finding in _build_literature_findings(record):
        label = str(finding.get("paper", "")).strip()
        url = str(finding.get("url", "")).strip()
        if not label and not url:
            continue
        dedupe_key = (label, url)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        papers.append(
            {
                "label": label or url,
                "url": url,
                "source_variant": record.get("display_name", record.get("variant", "")),
            }
        )

    for source in record.get("evidence", []):
        label = str(source.get("label", "")).strip()
        url = str(source.get("url", "")).strip()
        if not label and not url:
            continue
        dedupe_key = (label, url)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        papers.append(
            {
                "label": label or url,
                "url": url,
                "source_variant": record.get("display_name", record.get("variant", "")),
            }
        )

    return papers


def _build_whitelist_probe_reference_rows(
    methylation: pd.DataFrame,
    knowledge_base: dict[str, Any],
    *,
    matched_variant_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Map each whitelist probe to curated variants, nearby loci, and supporting papers."""
    gene_context = knowledge_base.get("gene_context", {})
    gene_name = str(gene_context.get("gene_name", DEFAULT_GENE_NAME))
    relevant_probe_ids = list(gene_context.get("relevant_methylation_probe_ids", []))
    if not relevant_probe_ids:
        return []

    observed_rows = (
        methylation.drop_duplicates(subset=["probe_id"], keep="first")
        if "probe_id" in methylation.columns
        else methylation.iloc[0:0].copy()
    )
    observed_lookup = (
        {str(row["probe_id"]): row.to_dict() for _, row in observed_rows.iterrows()}
        if "probe_id" in observed_rows.columns
        else {}
    )
    manifest_lookup = _load_gene_manifest_probe_lookup(gene_name, relevant_probe_ids)
    variant_records = knowledge_base.get("variant_records", [])
    matched_variant_lookup = (
        {_normalize_lookup_key(variant_id) for variant_id in matched_variant_ids if variant_id}
        if matched_variant_ids is not None
        else None
    )

    reference_rows: list[dict[str, Any]] = []
    for probe_id in relevant_probe_ids:
        observed_annotation = observed_lookup.get(probe_id, {})
        manifest_annotation = manifest_lookup.get(probe_id, {})
        merged_annotation = {**manifest_annotation, **observed_annotation}
        observed_beta = pd.to_numeric(observed_annotation.get("beta"), errors="coerce")

        linked_variant_records = [
            record
            for record in variant_records
            if probe_id in record.get("relevant_methylation_probe_ids", [])
            and (
                matched_variant_lookup is None
                or _normalize_lookup_key(str(record.get("variant", ""))) in matched_variant_lookup
            )
        ]
        linked_variants = []
        for record in linked_variant_records:
            label = str(record.get("display_name", record.get("variant", probe_id))).strip()
            common_name = str(record.get("common_name", "")).strip()
            locus = None
            position = record.get("position")
            chromosome = record.get("chromosome")
            if chromosome is not None and position is not None and not pd.isna(position):
                try:
                    locus = f"chr{str(chromosome).removeprefix('chr')}:{int(position):,}"
                except (TypeError, ValueError):
                    locus = None
            linked_variants.append(
                {
                    "label": label,
                    "common_name": common_name,
                    "locus": locus,
                }
            )

        papers: list[dict[str, Any]] = []
        seen_papers: set[tuple[str, str, str]] = set()
        for record in linked_variant_records:
            for paper in _collect_variant_record_papers(record):
                dedupe_key = (
                    paper.get("label", ""),
                    paper.get("url", ""),
                    paper.get("source_variant", ""),
                )
                if dedupe_key in seen_papers:
                    continue
                seen_papers.add(dedupe_key)
                papers.append(paper)

        nearby_manifest_variants = _build_nearby_manifest_variant_rows(
            merged_annotation,
            allowed_variant_ids=matched_variant_ids,
        )
        if matched_variant_lookup is not None and not (
            linked_variants or nearby_manifest_variants or papers
        ):
            continue

        reference_rows.append(
            {
                "probe_id": probe_id,
                "observed_in_run": probe_id in observed_lookup,
                "beta": _round_beta(float(observed_beta)) if pd.notna(observed_beta) else None,
                "probe_locus": _format_probe_locus(merged_annotation),
                "linked_variants": linked_variants,
                "nearby_manifest_variants": nearby_manifest_variants,
                "papers": papers,
            }
        )

    return reference_rows


def build_variant_interpretations(
    variants: pd.DataFrame, knowledge_base: dict[str, Any], *, region: str
) -> dict[str, Any]:
    """Build a structured locus audit for observed VCF calls and decoded sample GT."""
    variants = _ensure_variant_genotype_annotations(variants)
    variant_records = knowledge_base.get("variant_records", [])
    gene_context = knowledge_base.get("gene_context", {})
    gene_name = str(gene_context.get("gene_name", DEFAULT_GENE_NAME))
    chrom = str(gene_context.get("chromosome", "11")).removeprefix("chr")
    assembly = str(gene_context.get("assembly", "GRCh37 / hg19"))
    gene_region_source = gene_context.get("gene_region", {})
    promoter_region_source = gene_context.get("promoter_review_region", {})
    promoter_hotspot_source = gene_context.get("promoter_hotspot_region", {})

    search_region = _parse_region_string(region)
    search_region_record = _build_interval_record(
        "Current search interval",
        str(search_region["chrom"]).removeprefix("chr"),
        int(search_region["start"]),
        int(search_region["end"]),
        "Exact interval used to pull VCF calls from the selected file.",
    )
    gene_region_record = _build_interval_record(
        gene_region_source.get("label", f"{gene_name} transcribed interval"),
        chrom,
        int(gene_region_source["start"]),
        int(gene_region_source["end"]),
        gene_region_source.get("definition", ""),
    )
    promoter_region_record = _build_interval_record(
        promoter_region_source.get("label", "Operational promoter review window"),
        chrom,
        int(promoter_region_source["start"]),
        int(promoter_region_source["end"]),
        promoter_region_source.get("definition", ""),
    )
    promoter_hotspot_record = _build_interval_record(
        promoter_hotspot_source.get("label", "Promoter polymorphism hotspot"),
        chrom,
        int(promoter_hotspot_source["start"]),
        int(promoter_hotspot_source["end"]),
        promoter_hotspot_source.get("definition", ""),
    )

    promoter_records = [
        record
        for record in variant_records
        if record.get("region_class") in {"promoter", "promoter_structural", "upstream_regulatory"}
    ]
    gene_records = [
        record
        for record in variant_records
        if record.get("region_class") in {"coding_repeat", "gene_body"}
    ]
    region_records_for_union = [promoter_region_record, gene_region_record]
    recommended_region = str(gene_context.get("recommended_promoter_plus_gene_region") or "")
    if recommended_region and _region_text_covers_records(
        recommended_region,
        region_records_for_union,
        chrom=chrom,
    ):
        combined_region = recommended_region
    else:
        combined_region = _format_plain_interval_union(chrom, region_records_for_union)

    matched_records: list[dict[str, Any]] = []
    seen_matches: set[tuple[str, str]] = set()
    for _, row in variants.iterrows():
        matched_record = _match_variant_record(row, variant_records)
        if matched_record is None:
            continue

        observed_label = _format_variant_display(row)
        dedupe_key = (matched_record["variant"], observed_label)
        if dedupe_key in seen_matches:
            continue
        genotype = build_canonical_genotype(row)
        genotype_interpretation = (
            f"VCF genotype decoding: GT {genotype['gt_raw']} maps to "
            f"{genotype['genotype']} ({genotype['zygosity']}) with ALT dosage "
            f"{_format_allele_dosage(genotype['allele_dosage_per_alt'])}. "
            "REF and ALT define the site alleles; they are not treated as the person's genotype."
        )

        matched_records.append(
            {
                "rsid": _format_variant_label(row.get("id")),
                "chrom": str(row.get("chrom", "")).strip(),
                "pos": int(row.get("pos")) if pd.notna(row.get("pos")) else None,
                "ref": "" if pd.isna(row.get("ref")) else str(row.get("ref")).strip(),
                "alt": "" if pd.isna(row.get("alt")) else str(row.get("alt")).strip(),
                "observed_variant": observed_label,
                "variant_label": _format_variant_label(row.get("id")),
                "change": _format_variant_change(row.get("ref"), row.get("alt")),
                "site_definition": _format_variant_change(row.get("ref"), row.get("alt")),
                "reference_allele": "" if pd.isna(row.get("ref")) else str(row.get("ref")).strip(),
                "alternate_allele": "" if pd.isna(row.get("alt")) else str(row.get("alt")).strip(),
                "gt_raw": genotype["gt_raw"],
                "phased": genotype["phased"],
                "genotype_alleles": genotype["genotype_alleles"],
                "genotype": genotype["genotype"],
                "zygosity": genotype["zygosity"],
                "allele_dosage_per_alt": genotype["allele_dosage_per_alt"],
                "allele_dosage": _format_allele_dosage(genotype["allele_dosage_per_alt"]),
                "filter_status": genotype["filter_status"],
                "qual": genotype["qual"],
                "dp": genotype["dp"],
                "ad": genotype["ad"],
                "sample_af": genotype["sample_af"],
                "gq": genotype["gq"],
                "pl_or_gp_summary": genotype["pl_or_gp_summary"],
                "qc_flags": genotype["qc_flags"],
                "interpretation": genotype_interpretation,
                "confidence_score": genotype["confidence_score"],
                "confidence_explanation": genotype["confidence_explanation"],
                "linked_to": _build_specific_variant_link_summary(
                    matched_record,
                    fallback=f"No curated local {gene_name} link is bundled for this VCF call yet.",
                ),
                "variant": matched_record.get("display_name", matched_record["variant"]),
                "interpretation_scope": matched_record.get("interpretation_scope", "Research context"),
                "clinical_interpretation": matched_record.get("clinical_interpretation", ""),
                "clinical_significance": matched_record.get(
                    "clinical_significance", "Clinical significance not specified."
                ),
                "methylation_interpretation": matched_record.get("methylation_interpretation", ""),
                "functional_effects": matched_record.get("functional_effects", []),
                "associated_conditions": matched_record.get("associated_conditions", []),
                "research_context": matched_record.get("research_context", []),
                "literature_findings": _build_literature_findings(matched_record),
                "relevant_probe_ids": matched_record.get("relevant_methylation_probe_ids", []),
                "evidence": matched_record.get("evidence", []),
            }
        )
        seen_matches.add(dedupe_key)

    promoter_analysis = _build_region_variant_analysis(
        region_record=promoter_region_record,
        search_region=search_region,
        variants=variants,
        curated_records=promoter_records,
        gene_name=gene_name,
        inclusion_hint=(
            "Use the promoter-plus-gene interval "
            f"{combined_region} "
            f"if you want the upstream {gene_name} promoter reviewed alongside the gene."
        ),
        assembly=assembly,
    )
    gene_analysis = _build_region_variant_analysis(
        region_record=gene_region_record,
        search_region=search_region,
        variants=variants,
        curated_records=gene_records,
        gene_name=gene_name,
        inclusion_hint=f"Choose a region that overlaps the canonical {gene_name} gene interval if you want gene-body variants interpreted.",
        assembly=assembly,
    )

    promoter_phrase = "overlaps" if promoter_analysis["included"] else "does not overlap"
    gene_phrase = "overlaps" if gene_analysis["included"] else "does not overlap"
    summary = (
        f"{gene_name} is located at {gene_region_record['display']} on {gene_context.get('cytoband', 'the reported cytoband')} "
        f"and spans {gene_region_record['length_bp']:,} bp on the {gene_context.get('assembly', 'GRCh37 / hg19')} assembly. "
        f"The current search interval {search_region_record['display']} {promoter_phrase} the operational promoter review window "
        f"{promoter_region_record['display']} and {gene_phrase} the {gene_name} transcribed interval {gene_region_record['display']}. "
        f"In the current preview, {promoter_analysis['found_variant_count']} promoter-window variant(s) and "
        f"{gene_analysis['found_variant_count']} gene-interval VCF call(s) were surfaced. "
        "Sample genotype state is decoded from FORMAT/GT and QC fields; REF -> ALT is shown only as the site definition."
    )
    curated_named_markers = _build_curated_named_marker_catalog(
        variants,
        variant_records,
        assembly=assembly,
    )
    observed_named_marker_count = sum(1 for item in curated_named_markers if item["observed_in_run"])
    if curated_named_markers:
        curated_named_markers_summary = (
            f"The local {gene_name} bundle seeds {len(curated_named_markers)} curated named marker(s). "
            f"The current run directly matched {observed_named_marker_count} of them."
        )
    else:
        curated_named_markers_summary = (
            f"No curated named markers are bundled for {gene_name} yet."
        )

    return {
        "summary": summary,
        "matched_records": matched_records,
        "unclassified_variant_count": max(len(variants) - len(matched_records), 0),
        "gene_summary": gene_context.get("gene_summary", ""),
        "database_name": knowledge_base.get("database_name", f"Local {gene_name} interpretation database"),
        "gene_name": gene_name,
        "clinical_context": gene_context.get("clinical_context", ""),
        "variant_effect_overview": gene_context.get("variant_effect_overview", []),
        "condition_research_overview": gene_context.get("condition_research_overview", []),
        "biorender_visuals": gene_context.get("biorender_visuals"),
        "sample_highlights": _build_sample_variant_highlights(
            matched_records=matched_records,
            promoter_analysis=promoter_analysis,
            gene_analysis=gene_analysis,
            gene_name=gene_name,
        ),
        "region_recommendations": _build_region_recommendations(
            promoter_region_record,
            gene_region_record,
            combined_region,
            gene_name=gene_name,
        ),
        "gene_region": gene_region_record,
        "promoter_region": promoter_region_record,
        "promoter_hotspot_region": promoter_hotspot_record,
        "search_region": search_region_record,
        "curated_named_markers": curated_named_markers,
        "curated_named_markers_summary": curated_named_markers_summary,
        "promoter_analysis": promoter_analysis,
        "gene_analysis": gene_analysis,
        "recommended_promoter_plus_gene_region": combined_region,
    }


def build_population_insights(
    variants: pd.DataFrame,
    knowledge_base: dict[str, Any],
    population_database: dict[str, Any],
) -> dict[str, Any]:
    """Build geography-aware population summaries for common curated gene variants."""
    variant_records = knowledge_base.get("variant_records", [])
    gene_name = str(knowledge_base.get("gene_context", {}).get("gene_name", DEFAULT_GENE_NAME))
    curated_record_map = {record["variant"]: record for record in variant_records}

    observed_variant_map: dict[str, list[str]] = {}
    for _, row in variants.iterrows():
        matched_record = _match_variant_record(row, variant_records)
        if matched_record is None:
            continue
        observed_variant_map.setdefault(matched_record["variant"], []).append(_format_variant_display(row))

    population_variant_records: list[dict[str, Any]] = []
    location_groups: set[str] = set()
    for record in population_database.get("variant_population_records", []):
        knowledge_record = curated_record_map.get(record["variant"], {})
        focus_alleles = list(record.get("focus_alleles", []))
        effect_allele = record.get("effect_allele")
        top_level_rows = _build_population_frequency_rows(
            record.get("top_level_location_frequencies", []),
            focus_alleles=focus_alleles,
            effect_allele=effect_allele,
        )
        detailed_rows = _build_population_frequency_rows(
            record.get("detailed_population_frequencies", []),
            focus_alleles=focus_alleles,
            effect_allele=effect_allele,
        )

        for row in top_level_rows:
            location_groups.add(row["location_group"])

        population_variant_records.append(
            {
                "variant": record["variant"],
                "display_name": record.get("display_name", record["variant"]),
                "common_name": record.get("common_name"),
                "effect_allele": effect_allele,
                "effect_summary": record.get("effect_summary", ""),
                "functional_effects": knowledge_record.get("functional_effects", []),
                "associated_conditions": knowledge_record.get("associated_conditions", []),
                "research_context": knowledge_record.get("research_context", []),
                "observed_in_run": record["variant"] in observed_variant_map,
                "observed_variants": observed_variant_map.get(record["variant"], []),
                "top_level_location_frequencies": top_level_rows,
                "detailed_population_groups": _group_population_rows_by_location(detailed_rows),
                "population_extremes": _summarize_population_extremes(top_level_rows, effect_allele),
                "source_url": record.get("source_url"),
            }
        )

    matched_population_variants = [
        record["display_name"] for record in population_variant_records if record["observed_in_run"]
    ]
    if matched_population_variants:
        overlap_note = (
            "The current run directly overlaps curated population-backed variants: "
            + ", ".join(matched_population_variants)
            + "."
        )
    else:
        overlap_note = (
            "The current run did not directly hit one of the curated SNPs with built-in population frequencies, "
            f"so this section provides reference context for commonly studied {gene_name} variants."
        )

    if population_variant_records:
        location_summary = ", ".join(sorted(location_groups)) if location_groups else "available reference panels"
        summary = (
            f"Population reference data are available for {len(population_variant_records)} curated {gene_name} SNPs "
            f"across {location_summary} panels. "
            f"{overlap_note}"
        )
    elif population_database.get("gene_population_patterns"):
        summary = (
            f"No embedded allele-frequency panel is bundled for {gene_name} yet, but literature-backed gene-level "
            f"population notes are available. {overlap_note}"
        )
    else:
        summary = (
            f"No curated population reference panel is bundled for {gene_name} yet. "
            "The app is therefore showing raw variant and methylation results without population-frequency overlays."
        )

    return {
        "database_name": population_database.get("database_name", f"Local {gene_name} population database"),
        "summary": summary,
        "location_groups": sorted(location_groups),
        "variant_population_records": population_variant_records,
        "gene_population_patterns": population_database.get("gene_population_patterns", []),
        "gene_population_patterns_intro": population_database.get(
            "gene_population_patterns_intro",
            f"Broader population patterns curated from the {gene_name} literature.",
        ),
        "sources": population_database.get("sources", []),
    }


def build_methylation_insights(
    methylation: pd.DataFrame,
    knowledge_base: dict[str, Any],
    *,
    matched_variant_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build a gene-level methylation interpretation from the current probe table."""
    gene_context = knowledge_base.get("gene_context", {})
    gene_name = str(gene_context.get("gene_name", DEFAULT_GENE_NAME))
    relevant_probe_ids = list(gene_context.get("relevant_methylation_probe_ids", []))
    relevant_probe_lookup = set(relevant_probe_ids)

    observed_relevant = (
        methylation[methylation["probe_id"].isin(relevant_probe_lookup)].copy()
        if relevant_probe_lookup
        else methylation.iloc[0:0].copy()
    )
    gene_named_rows, gene_named_match_columns = _select_gene_named_methylation_rows(
        methylation, gene_name
    )

    whitelist_beta_values = _coerce_numeric_beta_values(observed_relevant)
    gene_named_beta_values = _coerce_numeric_beta_values(gene_named_rows)
    full_table_beta_values = _coerce_numeric_beta_values(methylation)

    whitelist_mean_beta = _mean_beta_or_none(whitelist_beta_values)
    gene_name_mean_beta = _mean_beta_or_none(gene_named_beta_values)
    raw_mean_beta = _mean_beta_or_none(full_table_beta_values)

    primary_mean_beta = (
        whitelist_mean_beta
        if whitelist_mean_beta is not None
        else gene_name_mean_beta
        if gene_name_mean_beta is not None
        else raw_mean_beta
    )
    beta_band = _categorize_beta(primary_mean_beta)
    beta_band_source_label = (
        "Whitelist mean beta"
        if whitelist_mean_beta is not None
        else f"{gene_name}-named row mean beta"
        if gene_name_mean_beta is not None
        else "All numeric-row mean beta"
    )
    beta_band_probe_count = (
        len(whitelist_beta_values)
        if whitelist_mean_beta is not None
        else len(gene_named_beta_values)
        if gene_name_mean_beta is not None
        else len(full_table_beta_values)
    )
    band_interpretations = gene_context.get("methylation_band_interpretations", {})
    band_interpretation = (
        str(band_interpretations.get(beta_band, "")).strip()
        if isinstance(band_interpretations, dict)
        else ""
    )
    sample_interpretation = ""
    if primary_mean_beta is not None and band_interpretation:
        beta_display = f"{primary_mean_beta:.3f}".rstrip("0").rstrip(".")
        sample_interpretation = (
            f"In this sample, the {beta_band_source_label.lower()} is {beta_display} across "
            f"{beta_band_probe_count} numeric probe(s), which falls in the {beta_band} band. "
            f"{band_interpretation}"
        )

    if "UCSC_RefGene_Group" in observed_relevant.columns:
        group_breakdown = (
            observed_relevant["UCSC_RefGene_Group"]
            .fillna("Unannotated")
            .value_counts()
            .to_dict()
        )
    else:
        group_breakdown = {}

    group_summary = ", ".join(f"{count} {group}" for group, count in group_breakdown.items())
    if not group_summary:
        group_summary = "annotation breakdown unavailable"

    whitelist_summary = (
        f"the whitelist mean beta is {whitelist_mean_beta:.2f} from {len(whitelist_beta_values)} numeric whitelist probe(s)"
        if whitelist_mean_beta is not None
        else "the whitelist mean beta is unavailable because no observed whitelist probe carried a numeric beta value"
    )
    gene_named_summary = (
        f"the {gene_name}-named row mean beta is {gene_name_mean_beta:.2f} from {len(gene_named_beta_values)} numeric row(s)"
        if gene_name_mean_beta is not None
        else f"the {gene_name}-named row mean beta is unavailable because no numeric row explicitly mentioned {gene_name}"
    )
    raw_summary = (
        f"the all-numeric-row mean beta is {raw_mean_beta:.2f} from {len(full_table_beta_values)} numeric row(s)"
        if raw_mean_beta is not None
        else "the all-numeric-row mean beta is unavailable because the table did not contain any numeric beta values"
    )

    summary = (
        f"The current run observed {len(observed_relevant)} of {len(relevant_probe_ids)} curated whitelist probe(s), "
        f"{len(gene_named_rows)} row(s) whose gene annotation explicitly mentions {gene_name}, "
        f"and {len(methylation)} row(s) in the full methylation table. "
        f"Across those views, {whitelist_summary}; {gene_named_summary}; and {raw_summary}. "
        f"The observed whitelist subset is dominated by {group_summary}. "
        f"{sample_interpretation} "
        f"{gene_context.get('methylation_interpretation', '')}"
    ).strip()

    preview_columns = [
        "probe_id",
        "beta",
        "UCSC_RefGene_Group",
        "Relation_to_UCSC_CpG_Island",
        "UCSC_CpG_Islands_Name",
    ]
    available_preview_columns = [
        column for column in preview_columns if column in observed_relevant.columns
    ]
    observed_relevant_lookup = set(observed_relevant["probe_id"].tolist()) if "probe_id" in observed_relevant.columns else set()
    gene_name_match_rule = (
        f"Rows count toward the {gene_name}-named mean when {gene_name} appears as a semicolon-delimited token in "
        + ", ".join(gene_named_match_columns)
        + "."
        if gene_named_match_columns
        else f"No gene-annotation column in the current methylation table explicitly mentioned {gene_name}."
    )
    whitelist_probe_reference_rows = _build_whitelist_probe_reference_rows(
        methylation,
        knowledge_base,
        matched_variant_ids=matched_variant_ids,
    )
    summary_metric_rows = _build_methylation_metric_rows(
        gene_name=gene_name,
        whitelist_mean_beta=whitelist_mean_beta,
        whitelist_count=len(whitelist_beta_values),
        gene_name_mean_beta=gene_name_mean_beta,
        gene_name_count=len(gene_named_beta_values),
        raw_mean_beta=raw_mean_beta,
        raw_count=len(full_table_beta_values),
    )
    if matched_variant_ids is None:
        whitelist_probe_reference_summary = (
            "Each whitelist probe is cross-referenced to any curated variant records that explicitly cite it, "
            "plus nearby manifest SNP annotations and the bundled papers behind those variant interpretations."
        )
    elif matched_variant_ids:
        whitelist_probe_reference_summary = (
            "Each whitelist probe is cross-referenced only to curated variant records observed in the current run, "
            "plus any matching manifest SNP annotations and bundled papers for those same observed variants."
        )
    else:
        whitelist_probe_reference_summary = (
            "No curated variant observed in the current run has a probe-specific reference map, so the probe-to-variant table is hidden."
        )

    return {
        "gene_name": gene_name,
        "clinical_context": gene_context.get("clinical_context", ""),
        "summary": summary,
        "sample_interpretation": sample_interpretation,
        "mean_beta": _round_beta(whitelist_mean_beta),
        "mean_beta_label": "Whitelist mean beta",
        "mean_beta_probe_count": int(len(whitelist_beta_values)),
        "whitelist_mean_beta": _round_beta(whitelist_mean_beta),
        "whitelist_mean_beta_label": "Whitelist mean beta",
        "whitelist_mean_beta_probe_count": int(len(whitelist_beta_values)),
        "whitelist_probe_count": len(relevant_probe_ids),
        "whitelist_observed_probe_count": int(len(observed_relevant)),
        "whitelist_explanation": gene_context.get(
            "methylation_whitelist_explanation"
        )
        or _build_whitelist_explanation(gene_name, relevant_probe_ids),
        "whitelist_literature_context": gene_context.get("methylation_interpretation", ""),
        "whitelist_probe_statuses": [
            {
                "probe_id": probe_id,
                "observed_in_run": probe_id in observed_relevant_lookup,
            }
            for probe_id in relevant_probe_ids
        ],
        "gene_name_mean_beta": _round_beta(gene_name_mean_beta),
        "gene_name_mean_beta_label": f"{gene_name}-named row mean beta",
        "gene_name_mean_beta_probe_count": int(len(gene_named_beta_values)),
        "gene_name_row_count": int(len(gene_named_rows)),
        "gene_name_match_columns": gene_named_match_columns,
        "gene_name_match_rule": gene_name_match_rule,
        "raw_mean_beta": _round_beta(raw_mean_beta),
        "raw_mean_beta_label": "All numeric-row mean beta",
        "raw_probe_count": int(len(methylation)),
        "raw_mean_beta_probe_count": int(len(full_table_beta_values)),
        "all_numeric_mean_beta": _round_beta(raw_mean_beta),
        "all_numeric_mean_beta_label": "All numeric-row mean beta",
        "all_numeric_mean_beta_probe_count": int(len(full_table_beta_values)),
        "summary_metric_rows": summary_metric_rows,
        "beta_band": beta_band,
        "beta_band_source_label": beta_band_source_label,
        "observed_probe_count": int(len(observed_relevant)),
        "curated_probe_count": len(relevant_probe_ids),
        "probe_ids": observed_relevant["probe_id"].tolist(),
        "group_breakdown": group_breakdown,
        "methylation_effects": gene_context.get("methylation_effects", []),
        "methylation_condition_research": gene_context.get("methylation_condition_research", []),
        "evidence": gene_context.get("evidence", []),
        "probe_preview": observed_relevant[available_preview_columns].copy(),
        "whitelist_probe_reference_rows": whitelist_probe_reference_rows,
        "whitelist_probe_reference_summary": whitelist_probe_reference_summary,
    }


def build_generic_variant_interpretations(
    variants: pd.DataFrame,
    *,
    region: str,
    gene_name: str,
) -> dict[str, Any]:
    """Build a generic region-based interpretation payload for genes without a curated database."""
    variants = _ensure_variant_genotype_annotations(variants)
    search_region = _parse_region_string(region)
    chrom = str(search_region["chrom"]).removeprefix("chr")
    search_region_record = _build_interval_record(
        f"{gene_name} selected interval",
        chrom,
        int(search_region["start"]),
        int(search_region["end"]),
        "Interval selected during preprocessing or manual entry for the current gene.",
    )

    observed_summaries = [
        _build_observed_variant_summary(row, None, gene_name=gene_name)
        for _, row in variants.sort_values("pos").head(12).iterrows()
    ]
    highlight_items = [
        {
            "title": item["display"],
            "observed_variant": item["position"],
            "change": item.get("change", "Unavailable"),
            "genotype": item.get("genotype", "Unavailable"),
            "gt_raw": item.get("gt_raw", "./."),
            "zygosity": item.get("zygosity", "missing"),
            "allele_dosage": item.get("allele_dosage", "Unavailable"),
            "confidence_score": item.get("confidence_score", 0),
            "confidence_explanation": item.get("confidence_explanation", ""),
            "qc_flags": item.get("qc_flags", []),
            "category": "Observed VCF call",
            "description": item["summary"],
            "conditions": [],
            "literature_findings": [],
        }
        for item in observed_summaries
    ]

    no_model_region = {
        "label": "Promoter model unavailable",
        "window": "Manual promoter interval required",
        "length_bp": 0,
        "definition": (
            f"No curated promoter interval is bundled for {gene_name} yet. "
            "Use a gene-specific promoter window if you need promoter-only review."
        ),
        "included": False,
        "analysis_note": (
            f"Promoter-specific interpretation is not currently modeled for {gene_name}. "
            "The app is using the selected interval as a generic gene-level review window."
        ),
        "found_variant_count": 0,
        "found_variants": [],
        "known_variants": [],
    }

    return {
        "summary": (
            f"{gene_name} is being analyzed with a generic region-based workflow. "
            f"No curated interpretation database is bundled for this gene yet, so the app is prioritizing "
            f"observed VCF calls and raw previews inside {search_region_record['display']}."
        ),
        "matched_records": [],
        "unclassified_variant_count": int(len(variants)),
        "gene_summary": (
            f"The current run treats {search_region_record['display']} as the active {gene_name} review interval."
        ),
        "database_name": f"No curated {gene_name} interpretation database loaded",
        "gene_name": gene_name,
        "clinical_context": (
            f"Variant interpretation for {gene_name} is currently generic. "
            "Observed calls are shown directly, but no bundled disease- or gene-specific assertions are being applied."
        ),
        "variant_effect_overview": [],
        "condition_research_overview": [],
        "sample_highlights": {
            "summary": (
                f"This sample yielded {len(observed_summaries)} visible VCF call(s) in the current {gene_name} interval preview."
            ),
            "highlight_items": highlight_items,
            "result_table_rows": [
                {
                    "variant_label": item.get("variant_label", "None"),
                    "change": item.get("change", "Unavailable"),
                    "genotype": item.get("genotype", "Unavailable"),
                    "zygosity": item.get("zygosity", "missing"),
                    "allele_dosage": item.get("allele_dosage", "Unavailable"),
                    "confidence_score": item.get("confidence_score", 0),
                    "qc_flags": item.get("qc_flags", []),
                    "linked_to": item.get("linked_to", ""),
                }
                for item in observed_summaries
            ],
        },
        "region_recommendations": [
            {
                "title": "Promoter only",
                "region": "Set manually for this gene",
                "purpose": (
                    f"No curated promoter span is bundled for {gene_name} yet. "
                    "Enter a promoter-focused coordinate window manually if you need upstream review."
                ),
            },
            {
                "title": "Gene body only",
                "region": search_region_record["display"],
                "purpose": (
                    f"Use the selected {gene_name} interval when you want a direct gene-body review with the current generic workflow."
                ),
            },
            {
                "title": "Promoter plus gene body",
                "region": search_region_record["display"],
                "purpose": (
                    f"Start from the selected {gene_name} interval and extend it upstream manually if you want promoter-plus-gene coverage."
                ),
            },
        ],
        "gene_region": search_region_record,
        "curated_named_markers": [],
        "curated_named_markers_summary": (
            f"No curated named markers are bundled for {gene_name} yet."
        ),
        "promoter_region": {
            "label": "Promoter model unavailable",
            "chrom": chrom,
            "start": int(search_region["start"]),
            "end": int(search_region["start"]),
            "length_bp": 0,
            "display": "Manual promoter interval required",
            "definition": no_model_region["definition"],
        },
        "promoter_hotspot_region": {
            "label": "Promoter hotspot unavailable",
            "chrom": chrom,
            "start": int(search_region["start"]),
            "end": int(search_region["start"]),
            "length_bp": 0,
            "display": "No curated hotspot available",
            "definition": f"No curated promoter hotspot interval is bundled for {gene_name}.",
        },
        "search_region": search_region_record,
        "promoter_analysis": no_model_region,
        "gene_analysis": {
            "label": f"{gene_name} selected interval",
            "window": search_region_record["display"],
            "length_bp": search_region_record["length_bp"],
            "definition": search_region_record["definition"],
            "included": True,
            "analysis_note": (
                f"The current run found {len(observed_summaries)} VCF call(s) in the first preview slice for the selected {gene_name} interval."
            ),
            "found_variant_count": len(observed_summaries),
            "found_variants": observed_summaries,
            "known_variants": [],
        },
        "recommended_promoter_plus_gene_region": search_region_record["display"],
    }


def build_generic_methylation_insights(methylation: pd.DataFrame, *, gene_name: str) -> dict[str, Any]:
    """Build a generic methylation payload for genes without a curated interpretation database."""
    gene_named_rows, gene_named_match_columns = _select_gene_named_methylation_rows(
        methylation, gene_name
    )
    gene_named_beta_values = _coerce_numeric_beta_values(gene_named_rows)
    beta_values = _coerce_numeric_beta_values(methylation)
    gene_name_mean_beta = _mean_beta_or_none(gene_named_beta_values)
    mean_beta = _mean_beta_or_none(beta_values)
    primary_mean_beta = gene_name_mean_beta if gene_name_mean_beta is not None else mean_beta
    beta_band = _categorize_beta(primary_mean_beta)
    beta_band_source_label = (
        f"{gene_name}-named row mean beta"
        if gene_name_mean_beta is not None
        else "All numeric-row mean beta"
    )

    if "UCSC_RefGene_Group" in methylation.columns:
        group_breakdown = (
            methylation["UCSC_RefGene_Group"].fillna("Unannotated").value_counts().to_dict()
        )
    else:
        group_breakdown = {}

    preview_columns = [
        "probe_id",
        "beta",
        "chrom",
        "pos",
        "UCSC_RefGene_Group",
        "Relation_to_UCSC_CpG_Island",
        "UCSC_CpG_Islands_Name",
    ]
    available_preview_columns = [column for column in preview_columns if column in methylation.columns]

    summary_prefix = (
        f"The current run captured {len(methylation)} probe(s) from the selected {gene_name} manifest subset, "
        f"with a {gene_name}-named row mean beta of {gene_name_mean_beta:.2f} from {len(gene_named_beta_values)} numeric row(s) "
        f"and an all-numeric-row mean beta of {mean_beta:.2f} from {len(beta_values)} numeric row(s). "
        if mean_beta is not None and gene_name_mean_beta is not None
        else f"The current run captured {len(methylation)} probe(s) from the selected {gene_name} manifest subset "
        f"with an all-numeric-row mean beta of {mean_beta:.2f}. "
        if mean_beta is not None
        else f"The current run captured {len(methylation)} probe(s) from the selected {gene_name} manifest subset. "
    )
    gene_name_match_rule = (
        f"Rows count toward the {gene_name}-named mean when {gene_name} appears as a semicolon-delimited token in "
        + ", ".join(gene_named_match_columns)
        + "."
        if gene_named_match_columns
        else f"No gene-annotation column in the current methylation table explicitly mentioned {gene_name}."
    )
    summary_metric_rows = _build_methylation_metric_rows(
        gene_name=gene_name,
        whitelist_mean_beta=None,
        whitelist_count=0,
        gene_name_mean_beta=gene_name_mean_beta,
        gene_name_count=len(gene_named_beta_values),
        raw_mean_beta=mean_beta,
        raw_count=len(beta_values),
    )

    return {
        "gene_name": gene_name,
        "clinical_context": (
            f"No curated methylation interpretation database is bundled for {gene_name} yet."
        ),
        "summary": (
            summary_prefix
            + "These values are shown as gene-focused methylation context rather than as a curated clinical interpretation."
        ),
        "mean_beta": _round_beta(gene_name_mean_beta if gene_name_mean_beta is not None else mean_beta),
        "mean_beta_label": (
            f"{gene_name}-named row mean beta"
            if gene_name_mean_beta is not None
            else "All numeric-row mean beta"
        ),
        "mean_beta_probe_count": int(
            len(gene_named_beta_values) if gene_name_mean_beta is not None else len(beta_values)
        ),
        "whitelist_mean_beta": None,
        "whitelist_mean_beta_label": "Whitelist mean beta",
        "whitelist_mean_beta_probe_count": 0,
        "whitelist_probe_count": 0,
        "whitelist_observed_probe_count": 0,
        "whitelist_explanation": _build_whitelist_explanation(gene_name, []),
        "whitelist_literature_context": "",
        "whitelist_probe_statuses": [],
        "gene_name_mean_beta": _round_beta(gene_name_mean_beta),
        "gene_name_mean_beta_label": f"{gene_name}-named row mean beta",
        "gene_name_mean_beta_probe_count": int(len(gene_named_beta_values)),
        "gene_name_row_count": int(len(gene_named_rows)),
        "gene_name_match_columns": gene_named_match_columns,
        "gene_name_match_rule": gene_name_match_rule,
        "raw_mean_beta": _round_beta(mean_beta),
        "raw_mean_beta_label": "All numeric-row mean beta",
        "raw_probe_count": int(len(methylation)),
        "raw_mean_beta_probe_count": int(len(beta_values)),
        "all_numeric_mean_beta": _round_beta(mean_beta),
        "all_numeric_mean_beta_label": "All numeric-row mean beta",
        "all_numeric_mean_beta_probe_count": int(len(beta_values)),
        "summary_metric_rows": summary_metric_rows,
        "beta_band": beta_band,
        "beta_band_source_label": beta_band_source_label,
        "observed_probe_count": int(len(methylation)),
        "curated_probe_count": 0,
        "probe_ids": methylation["probe_id"].tolist() if "probe_id" in methylation.columns else [],
        "group_breakdown": group_breakdown,
        "methylation_effects": [
            f"The current methylation summary for {gene_name} is based on the filtered manifest subset selected during preprocessing.",
            "Beta values are being shown as region-level context until a curated gene-specific methylation knowledge base is added.",
        ],
        "methylation_condition_research": [],
        "evidence": [],
        "probe_preview": methylation[available_preview_columns].copy() if available_preview_columns else methylation.head(12).copy(),
        "whitelist_probe_reference_rows": [],
        "whitelist_probe_reference_summary": (
            f"No curated whitelist-probe reference map is bundled for {gene_name} yet."
        ),
    }


def build_empty_population_insights(*, gene_name: str) -> dict[str, Any]:
    """Return a placeholder population-insight payload for genes without curated reference data."""
    return {
        "summary": (
            f"No curated population reference database is bundled for {gene_name} yet. "
            "The app is therefore showing raw variant and methylation results without population-frequency overlays."
        ),
        "location_groups": [],
        "sources": [],
        "variant_population_records": [],
        "gene_population_patterns": [],
        "gene_population_patterns_intro": "",
        "database_name": f"No curated {gene_name} population database loaded",
        "database_version": "generic",
    }


def _categorize_predictive_beta_band(mean_beta: float | None) -> str:
    """Collapse UI methylation bands into the three predictive case buckets."""
    descriptive_band = _categorize_beta(mean_beta)
    if descriptive_band == "unavailable":
        return "unavailable"
    if descriptive_band == "low":
        return "low"
    if descriptive_band == "high":
        return "high"
    return "medium"


def _format_predictive_beta_display(mean_beta: float | None) -> str:
    """Render predictive beta values consistently for the UI."""
    rounded = _round_beta(mean_beta)
    return str(rounded) if rounded is not None else "Unavailable"


def _summarize_predictive_observed_variants(
    variant_interpretations: dict[str, Any],
) -> list[str]:
    """Return deduplicated human-readable labels for the current sample's observed variants."""
    observed_items: list[str] = []
    for item in variant_interpretations.get("sample_highlights", {}).get("highlight_items", []):
        title = str(item.get("title", "")).strip()
        observed_variant = str(item.get("observed_variant", "")).strip()
        change = str(item.get("change", "")).strip()
        genotype = str(item.get("genotype", "")).strip()
        has_change = bool(change and change.casefold() != "unavailable")
        genotype_suffix = f" GT={genotype}" if genotype and genotype.casefold() != "unavailable" else ""
        if title and has_change:
            observed_items.append(f"{title} {change}{genotype_suffix}")
        elif title and observed_variant and observed_variant != title:
            observed_items.append(f"{title} ({observed_variant}){genotype_suffix}")
        elif title:
            observed_items.append(f"{title}{genotype_suffix}")
        elif observed_variant:
            observed_items.append(f"{observed_variant}{genotype_suffix}")
    return _dedupe_text_items(observed_items)


def _build_synthesis_variant_prediction_rule_lookup(
    synthesis_database: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index concrete variant prediction rules by the labels seen in analysis output."""
    rule_lookup: dict[str, dict[str, Any]] = {}
    for rule in synthesis_database.get("variant_prediction_rules", []):
        lookup_candidates = [
            rule.get("variant"),
            rule.get("display_name"),
            rule.get("common_name"),
            *rule.get("lookup_keys", []),
        ]
        for candidate in lookup_candidates:
            candidate_text = str(candidate or "").strip()
            if not candidate_text:
                continue
            rule_lookup[_normalize_lookup_key(candidate_text)] = rule
    return rule_lookup


def _find_synthesis_variant_prediction_rule(
    record: dict[str, Any],
    rule_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the concrete prediction rule for a matched variant record when available."""
    lookup_candidates = [
        record.get("variant"),
        record.get("variant_label"),
        record.get("observed_variant"),
    ]
    for candidate in lookup_candidates:
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            continue
        matched_rule = rule_lookup.get(_normalize_lookup_key(candidate_text))
        if matched_rule is not None:
            return matched_rule
    return None


def _format_predictive_observed_signal(record: dict[str, Any]) -> str:
    """Render observed variant, site definition, and decoded sample genotype."""
    observed_signal = str(record.get("observed_variant", record.get("variant", ""))).strip()
    change = str(record.get("change", "")).strip()
    genotype = str(record.get("genotype", "")).strip()
    zygosity = str(record.get("zygosity", "")).strip()
    genotype_text = ""
    if genotype and genotype.casefold() != "unavailable":
        genotype_text = f"GT {record.get('gt_raw', './.')} = {genotype}"
        if zygosity:
            genotype_text = f"{genotype_text} ({zygosity})"
    if change and change.casefold() != "unavailable" and change not in observed_signal:
        observed_signal = f"{observed_signal} ({change})" if observed_signal else change
    if genotype_text:
        return f"{observed_signal}; {genotype_text}" if observed_signal else genotype_text
    return observed_signal


def _record_has_non_reference_genotype(record: dict[str, Any]) -> bool:
    """Return whether a matched record has GT-confirmed non-reference dosage."""
    zygosity = str(record.get("zygosity", "")).strip()
    if zygosity in {"heterozygous", "homozygous_alternate", "compound_heterozygous", "hemizygous_alternate"}:
        return True
    return _genotype_has_alt_dosage(record)


def _record_genotype_is_missing(record: dict[str, Any]) -> bool:
    """Return whether the record lacks a usable sample genotype."""
    return str(record.get("zygosity", "")).strip() in {"", "missing"}


def _format_prediction_confidence(score: Any) -> str:
    """Map numeric call confidence to a short display bucket."""
    parsed = _safe_float(score)
    if parsed is None:
        return "unknown"
    if parsed >= 0.80:
        return "high"
    if parsed >= 0.55:
        return "moderate"
    if parsed >= 0.30:
        return "low"
    return "very low"


def _format_genotype_prediction_context(record: dict[str, Any]) -> str:
    """Explain how genotype decoding constrains interpretation."""
    genotype = str(record.get("genotype", "Unavailable")).strip() or "Unavailable"
    gt_raw = str(record.get("gt_raw", "./.")).strip() or "./."
    zygosity = str(record.get("zygosity", "missing")).strip() or "missing"
    dosage = record.get("allele_dosage")
    if not dosage:
        dosage = _format_allele_dosage(record.get("allele_dosage_per_alt", {}))
    return f"GT {gt_raw} decodes as {genotype} ({zygosity}); ALT dosage is {dosage}."


def _prediction_row_from_record(
    *,
    record: dict[str, Any],
    selected_prediction: dict[str, str],
) -> dict[str, Any]:
    """Build a genotype-aware predictive-thesis table row."""
    confidence_score = record.get("confidence_score", 0)
    qc_flags = record.get("qc_flags", [])
    qc_note = "; ".join(str(flag) for flag in qc_flags) if qc_flags else "No major genotype-call QC flags"
    return {
        "observed_signal": _format_predictive_observed_signal(record),
        "genotype": str(record.get("genotype", "Unavailable")),
        "zygosity": str(record.get("zygosity", "missing")),
        "allele_dosage": record.get("allele_dosage") or _format_allele_dosage(record.get("allele_dosage_per_alt", {})),
        "confidence": _format_prediction_confidence(confidence_score),
        "confidence_score": confidence_score,
        "qc_flags": qc_flags,
        "source": selected_prediction["source"],
        "prediction": selected_prediction["prediction"],
        "research_focus": selected_prediction["research_focus"],
        "confidence_explanation": (
            f"{_format_genotype_prediction_context(record)} "
            f"Call confidence is {_format_prediction_confidence(confidence_score)} "
            f"({confidence_score}); {qc_note}. "
            f"{record.get('confidence_explanation', '')}"
        ).strip(),
    }


def _render_sample_change_template(
    template: str,
    *,
    record: dict[str, Any],
    concrete_rule: dict[str, Any],
) -> str:
    """Fill a controlled sample-change template from the synthesis database."""
    replacements = {
        "{change}": str(record.get("change", "Unavailable")).strip() or "Unavailable",
        "{variant}": str(record.get("variant", concrete_rule.get("variant", ""))).strip(),
        "{display_name}": str(concrete_rule.get("display_name", record.get("variant", ""))).strip(),
        "{observed_variant}": str(record.get("observed_variant", "")).strip(),
        "{alt_allele}": _extract_alt_allele_from_change(record.get("change")),
        "{gt_raw}": str(record.get("gt_raw", "./.")).strip() or "./.",
        "{genotype}": str(record.get("genotype", "Unavailable")).strip() or "Unavailable",
        "{zygosity}": str(record.get("zygosity", "missing")).strip() or "missing",
        "{allele_dosage}": str(
            record.get("allele_dosage")
            or _format_allele_dosage(record.get("allele_dosage_per_alt", {}))
        ),
    }
    rendered = str(template or "")
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered.strip()


def _select_synthesis_prediction_for_record(
    record: dict[str, Any],
    concrete_rule: dict[str, Any] | None,
) -> dict[str, str]:
    """Choose the most sample-specific prediction text for one matched variant."""
    genotype_context = _format_genotype_prediction_context(record)
    if _record_genotype_is_missing(record):
        genotype_note = (
            f"{genotype_context} Because GT is missing, this row is treated as a site-level marker match only; "
            "phenotype inference is not upgraded from REF/ALT alone."
        )
        if concrete_rule is None:
            prediction = str(record.get("clinical_interpretation", "")).strip()
        else:
            prediction = str(concrete_rule.get("prediction", "")).strip()
        return {
            "prediction": f"{genotype_note} {prediction}".strip(),
            "research_focus": (
                "Genotype unavailable; use as directional locus context only, not an individual genotype call."
            ),
            "source": "GT-missing site-level thesis",
        }

    if not _record_has_non_reference_genotype(record):
        return {
            "prediction": (
                f"{genotype_context} This sample is homozygous reference at the site, so the ALT-defined "
                "variant effect is not applied as a carried alternate-allele phenotype signal."
            ),
            "research_focus": "Reference-genotype context; no alternate-allele dosage for this marker.",
            "source": "Reference genotype thesis",
        }

    if concrete_rule is None:
        return {
            "prediction": (
                f"{genotype_context} "
                f"{str(record.get('clinical_interpretation', '')).strip()}"
            ).strip(),
            "research_focus": "; ".join(
                _dedupe_text_items(record.get("associated_conditions", []))[:3]
            ),
            "source": str(record.get("interpretation_scope", "Curated marker")).strip(),
        }

    change = str(record.get("change", "")).strip()
    normalized_change = _normalize_allele_change(change)
    observed_alt_allele = _extract_alt_allele_from_change(change)

    for allele_rule in concrete_rule.get("allele_change_rules", []):
        rule_change = _normalize_allele_change(allele_rule.get("change"))
        rule_alt_allele = str(allele_rule.get("alt_allele", "")).strip().upper()
        change_matches = bool(rule_change and normalized_change and rule_change == normalized_change)
        alt_dosage = int(record.get("allele_dosage_per_alt", {}).get(rule_alt_allele, 0) or 0)
        alt_matches = bool(
            rule_alt_allele
            and observed_alt_allele
            and rule_alt_allele == observed_alt_allele
            and alt_dosage > 0
        )
        if not change_matches and not alt_matches:
            continue

        prediction = str(allele_rule.get("prediction", "")).strip()
        if not prediction:
            prediction = str(concrete_rule.get("prediction", "")).strip()
        prediction = f"{genotype_context} {prediction}".strip()
        research_focus = str(allele_rule.get("basis", "")).strip()
        if not research_focus:
            research_focus = str(concrete_rule.get("basis", "")).strip()
        return {
            "prediction": prediction,
            "research_focus": research_focus,
            "source": "GT-confirmed allele-dosage thesis",
        }

    prediction = str(concrete_rule.get("prediction", "")).strip()
    research_focus = str(concrete_rule.get("basis", "")).strip()
    sample_change_template = str(concrete_rule.get("sample_change_template", "")).strip()
    if change and change.casefold() != "unavailable" and sample_change_template:
        change_anchor = _render_sample_change_template(
            sample_change_template,
            record=record,
            concrete_rule=concrete_rule,
        )
        if change_anchor and prediction:
            prediction = f"{genotype_context} {change_anchor} {prediction}"
        elif change_anchor:
            prediction = f"{genotype_context} {change_anchor}"
        return {
            "prediction": prediction,
            "research_focus": research_focus,
            "source": "Sample change-anchored thesis",
        }

    return {
        "prediction": f"{genotype_context} {prediction}".strip(),
        "research_focus": research_focus,
        "source": "Concrete variant thesis",
    }


def _record_matches_marker(record: dict[str, Any], marker: str) -> bool:
    """Return whether a predictive record refers to a marker label."""
    normalized_marker = _normalize_lookup_key(marker)
    candidates = [
        record.get("variant"),
        record.get("variant_label"),
        record.get("observed_variant"),
        record.get("rsid"),
    ]
    return any(normalized_marker in _normalize_lookup_key(str(candidate or "")) for candidate in candidates)


def _allele_count_in_genotype(record: dict[str, Any], allele: str) -> int:
    """Count a concrete nucleotide allele in the decoded sample genotype."""
    target = allele.strip().upper()
    return sum(1 for item in record.get("genotype_alleles", []) if str(item).strip().upper() == target)


def _build_generic_phenotype_prediction(
    *,
    gene_name: str,
    genotype_positive_records: list[dict[str, Any]],
    genotype_uncertain_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a conservative gene-level phenotype payload for non-specialized genes."""
    qc_warnings = _dedupe_text_items(
        [
            str(flag)
            for record in genotype_positive_records + genotype_uncertain_records
            for flag in record.get("qc_flags", [])
        ]
    )
    if genotype_positive_records:
        scores = [
            float(record.get("confidence_score", 0) or 0)
            for record in genotype_positive_records
        ]
        confidence_score = min(scores) if scores else 0.0
        prediction = (
            f"{gene_name} genotype evidence is compatible with a directional gene-level research signal, "
            "not a deterministic phenotype call."
        )
        evidence = (
            f"{len(genotype_positive_records)} curated marker(s) have GT-confirmed non-reference dosage. "
            "Variant effects should be interpreted by genotype dosage and call QC."
        )
    elif genotype_uncertain_records:
        confidence_score = 0.2
        prediction = (
            f"{gene_name} site-level marker evidence was seen, but GT is missing or unusable; "
            "no individual genotype-based phenotype should be inferred."
        )
        evidence = (
            f"{len(genotype_uncertain_records)} marker(s) matched the local database without usable sample GT."
        )
    else:
        confidence_score = 0.0
        prediction = f"No GT-confirmed {gene_name} phenotype signal was available from the current marker set."
        evidence = "No curated marker had non-reference sample genotype dosage."

    return {
        "phenotype_prediction": prediction,
        "confidence": _format_prediction_confidence(confidence_score),
        "confidence_score": round(confidence_score, 3),
        "evidence_summary": evidence,
        "uncertainty_summary": (
            "Most traits are polygenic and context dependent; this app reports directional compatibility from the "
            "available marker subset rather than biological certainty."
        ),
        "conflicting_evidence": [],
        "qc_warnings": qc_warnings,
    }


def _build_herc2_eye_color_prediction(matched_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a conservative HERC2/OCA2 eye-colour prediction from GT dosage."""
    marker_records = {
        marker: next((record for record in matched_records if _record_matches_marker(record, marker)), None)
        for marker in ("rs12913832", "rs1129038", "rs7170852")
    }
    major = marker_records["rs12913832"]
    secondary_records = [record for marker, record in marker_records.items() if marker != "rs12913832" and record]
    qc_warnings = _dedupe_text_items(
        [
            str(flag)
            for record in marker_records.values()
            if record
            for flag in record.get("qc_flags", [])
        ]
    )

    if major is None:
        return {
            "phenotype_prediction": (
                "HERC2/OCA2 eye-colour inference is incomplete because rs12913832, the strongest marker in this "
                "small panel, was not available as a decoded sample genotype."
            ),
            "confidence": "low",
            "confidence_score": 0.25,
            "evidence_summary": "Secondary markers alone are not enough for a clear eye-colour call.",
            "uncertainty_summary": (
                "Eye colour is polygenic; this app uses only a small HERC2/OCA2 subset and cannot exclude brown, hazel, "
                "green, or blue outcomes without broader genotype context."
            ),
            "conflicting_evidence": [],
            "qc_warnings": qc_warnings,
        }

    rs129_g = _allele_count_in_genotype(major, "G")
    rs129_a = _allele_count_in_genotype(major, "A")
    rs112_t = _allele_count_in_genotype(marker_records["rs1129038"], "T") if marker_records["rs1129038"] else 0
    rs112_c = _allele_count_in_genotype(marker_records["rs1129038"], "C") if marker_records["rs1129038"] else 0
    rs717_a = _allele_count_in_genotype(marker_records["rs7170852"], "A") if marker_records["rs7170852"] else 0
    rs717_t = _allele_count_in_genotype(marker_records["rs7170852"], "T") if marker_records["rs7170852"] else 0

    light_score = (rs129_g * 2.0) + (rs112_t * 0.6) + (rs717_a * 0.4)
    darker_score = (rs129_a * 1.5) + (rs112_c * 0.25) + (rs717_t * 0.2)
    strongest_is_heterozygous = major.get("zygosity") == "heterozygous"
    confidence_score = 0.35
    confidence_score += 0.20 if secondary_records else 0.0
    confidence_score += 0.15 if not strongest_is_heterozygous else 0.0
    confidence_score = min(confidence_score, float(major.get("confidence_score", 0) or 0))
    if strongest_is_heterozygous:
        confidence_score = min(confidence_score, 0.62)
    if len(secondary_records) < 2:
        confidence_score = min(confidence_score, 0.58)

    conflicting_evidence: list[str] = []
    if rs129_g and rs129_a:
        conflicting_evidence.append(
            "rs12913832 is heterozygous, so both lighter-eye-associated G and darker-eye-compatible A are present."
        )
    if light_score > 0 and darker_score > 0:
        conflicting_evidence.append(
            "The small panel contains both lighter-leaning and darker-compatible allele evidence."
        )

    if rs129_g == 2 and light_score > darker_score:
        phenotype = (
            "Blue or lighter-eye-compatible signal is present, but the result remains probabilistic rather than deterministic."
        )
        uncertainty = (
            "rs12913832 G/G is a strong HERC2/OCA2 contributor, yet eye colour remains polygenic and ancestry-dependent."
        )
    elif rs129_g == 1:
        phenotype = (
            "Lighter/intermediate-eye signal is present, but the strongest major marker is heterozygous rather than "
            "homozygous alternate; brown or hazel remains plausible."
        )
        uncertainty = (
            "Intermediate or hazel outcomes should carry only moderate confidence from this small SNP subset. "
            "The available markers are compatible with lighter pigmentation directionally, not a deterministic blue-eye call."
        )
    elif rs129_g == 0 and rs129_a >= 1:
        phenotype = (
            "Brown or darker-eye-compatible signal is stronger at rs12913832, while secondary markers may still leave "
            "room for intermediate pigmentation depending on the broader polygenic background."
        )
        uncertainty = (
            "Absence of the rs12913832 G signal in this panel does not fully determine eye colour; additional OCA2/HERC2 "
            "and genome-wide pigmentation markers can modify the visible result."
        )
    else:
        phenotype = (
            "Eye-colour prediction is directionally inconclusive because rs12913832 genotype dosage could not be mapped cleanly."
        )
        uncertainty = (
            "The marker is present, but the decoded genotype does not match the expected A/G representation."
        )
        confidence_score = min(confidence_score, 0.25)

    evidence_parts = [
        f"rs12913832 decoded as {major.get('genotype')} ({major.get('zygosity')}).",
        f"Lighter-score components: rs12913832 G dosage {rs129_g}, rs1129038 T dosage {rs112_t}, rs7170852 A dosage {rs717_a}.",
        f"Darker-compatible components: rs12913832 A dosage {rs129_a}, rs1129038 C dosage {rs112_c}, rs7170852 T dosage {rs717_t}.",
    ]

    return {
        "phenotype_prediction": phenotype,
        "confidence": _format_prediction_confidence(confidence_score),
        "confidence_score": round(confidence_score, 3),
        "evidence_summary": " ".join(evidence_parts),
        "uncertainty_summary": uncertainty,
        "conflicting_evidence": conflicting_evidence,
        "qc_warnings": qc_warnings,
    }


def _build_phenotype_prediction(
    *,
    gene_name: str,
    matched_records: list[dict[str, Any]],
    genotype_positive_records: list[dict[str, Any]],
    genotype_uncertain_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build phenotype-level output with explicit uncertainty."""
    if gene_name.upper() == "HERC2":
        return _build_herc2_eye_color_prediction(matched_records)
    return _build_generic_phenotype_prediction(
        gene_name=gene_name,
        genotype_positive_records=genotype_positive_records,
        genotype_uncertain_records=genotype_uncertain_records,
    )


def build_predictive_theses(
    *,
    variant_interpretations: dict[str, Any],
    methylation_insights: dict[str, Any],
    knowledge_base: dict[str, Any] | None = None,
    synthesis_database: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the predictive-thesis payload shown after a completed analysis run."""
    knowledge_base = knowledge_base or {}
    synthesis_database = synthesis_database or {}
    gene_context = knowledge_base.get("gene_context", {})
    gene_name = str(
        synthesis_database.get("gene_name")
        or variant_interpretations.get("gene_name")
        or methylation_insights.get("gene_name")
        or gene_context.get("gene_name")
        or DEFAULT_GENE_NAME
    ).strip() or DEFAULT_GENE_NAME

    case_catalog = [
        case
        for case in synthesis_database.get("cases", [])
        if str(case.get("case_id", "")).strip()
    ]
    case_lookup = {
        str(case["case_id"]).strip(): case
        for case in case_catalog
    }
    variant_prediction_rule_lookup = _build_synthesis_variant_prediction_rule_lookup(
        synthesis_database
    )

    matched_records = variant_interpretations.get("matched_records", [])
    genotype_positive_records = [
        record for record in matched_records if _record_has_non_reference_genotype(record)
    ]
    genotype_uncertain_records = [
        record for record in matched_records if _record_genotype_is_missing(record)
    ]
    highlight_items = variant_interpretations.get("sample_highlights", {}).get("highlight_items", [])
    observed_variant_labels = _summarize_predictive_observed_variants(variant_interpretations)
    variant_found = bool(genotype_positive_records)
    phenotype_prediction = _build_phenotype_prediction(
        gene_name=gene_name,
        matched_records=matched_records,
        genotype_positive_records=genotype_positive_records,
        genotype_uncertain_records=genotype_uncertain_records,
    )

    variant_case = case_lookup.get("gene_variant_found")
    variant_summary = (
        str(variant_case.get("prediction", "")).strip()
        if variant_found and variant_case is not None
        else ""
    )
    if not variant_summary and variant_found:
        variant_summary = (
            f"GT-confirmed non-reference {gene_name} variation is present in this sample, so the gene-level predictive thesis "
            "should be read as locus-specific research context rather than as a stand-alone diagnosis."
        )
    if not variant_found and genotype_uncertain_records:
        variant_summary = (
            f"{gene_name} marker rows matched the local database, but sample GT was missing or unusable. "
            "The app therefore reports site-level context and does not infer an individual genotype from REF/ALT alone."
        )
    if not variant_found:
        variant_summary = (
            variant_summary
            or (
                f"No GT-confirmed non-reference promoter or gene-body {gene_name} genotype was visible in the current preview, so the "
                "variant-gated predictive thesis cases did not match this sample."
            )
        )
    if observed_variant_labels:
        variant_summary = (
            f"{variant_summary} Observed sample signal: {', '.join(observed_variant_labels[:4])}."
        ).strip()

    variant_prediction_rows: list[dict[str, str]] = []
    if variant_found and variant_case is not None:
        variant_prediction_rows.append(
            {
                "observed_signal": (
                    ", ".join(observed_variant_labels[:3])
                    if observed_variant_labels
                    else f"{gene_name} interval non-reference genotype observed"
                ),
                "genotype": "Multiple",
                "zygosity": "See marker rows",
                "allele_dosage": "See marker rows",
                "confidence": phenotype_prediction.get("confidence", "unknown"),
                "confidence_score": phenotype_prediction.get("confidence_score", 0),
                "qc_flags": phenotype_prediction.get("qc_warnings", []),
                "source": "Gene-level thesis",
                "prediction": str(variant_case.get("prediction", "")).strip(),
                "research_focus": "; ".join(
                    _dedupe_text_items(variant_case.get("research_focus", []))[:3]
                ),
                "confidence_explanation": phenotype_prediction.get("uncertainty_summary", ""),
            }
        )

    for record in matched_records:
        concrete_rule = _find_synthesis_variant_prediction_rule(
            record,
            variant_prediction_rule_lookup,
        )
        selected_prediction = _select_synthesis_prediction_for_record(record, concrete_rule)

        variant_prediction_rows.append(
            _prediction_row_from_record(
                record=record,
                selected_prediction=selected_prediction,
            )
        )

    if not matched_records:
        for item in highlight_items:
            variant_prediction_rows.append(
                {
                    "observed_signal": str(item.get("title", item.get("observed_variant", ""))).strip(),
                    "genotype": str(item.get("genotype", "Unavailable")).strip(),
                    "zygosity": str(item.get("zygosity", "missing")).strip(),
                    "allele_dosage": str(item.get("allele_dosage", "Unavailable")).strip(),
                    "confidence": _format_prediction_confidence(item.get("confidence_score", 0)),
                    "confidence_score": item.get("confidence_score", 0),
                    "qc_flags": item.get("qc_flags", []),
                    "source": str(item.get("category", "Interval variant")).strip(),
                    "prediction": str(item.get("description", "")).strip(),
                    "research_focus": "; ".join(
                        _dedupe_text_items(item.get("conditions", []))[:3]
                    ),
                    "confidence_explanation": str(item.get("confidence_explanation", "")).strip(),
                }
            )

    methylation_source_rows = [
        {
            "metric_key": "whitelist",
            "label": str(methylation_insights.get("whitelist_mean_beta_label", "Whitelist mean beta")).strip(),
            "mean_beta": methylation_insights.get("whitelist_mean_beta"),
            "probe_count": int(methylation_insights.get("whitelist_mean_beta_probe_count", 0) or 0),
        },
        {
            "metric_key": "gene_name_related",
            "label": str(
                methylation_insights.get("gene_name_mean_beta_label", f"{gene_name}-named row mean beta")
            ).strip(),
            "mean_beta": methylation_insights.get("gene_name_mean_beta"),
            "probe_count": int(methylation_insights.get("gene_name_mean_beta_probe_count", 0) or 0),
        },
        {
            "metric_key": "all_numeric",
            "label": str(
                methylation_insights.get("all_numeric_mean_beta_label", "All numeric-row mean beta")
            ).strip(),
            "mean_beta": methylation_insights.get("all_numeric_mean_beta"),
            "probe_count": int(methylation_insights.get("all_numeric_mean_beta_probe_count", 0) or 0),
        },
    ]

    methylation_prediction_rows: list[dict[str, Any]] = []
    matched_cases: list[dict[str, str]] = []

    if variant_found and variant_case is not None:
        matched_cases.append(
            {
                "case_label": str(variant_case.get("label", "Gene variant found")).strip(),
                "trigger": "Observed promoter or gene-body variant",
                "source": "Variant-only synthesis",
                "mean_beta_display": "n/a",
                "band": "n/a",
                "prediction": str(variant_case.get("prediction", "")).strip(),
                "research_focus": "; ".join(
                    _dedupe_text_items(variant_case.get("research_focus", []))[:3]
                ),
            }
        )

    for source_row in methylation_source_rows:
        mean_beta = source_row["mean_beta"]
        band = _categorize_predictive_beta_band(mean_beta)
        case_id = (
            f"gene_variant_found__{source_row['metric_key']}__{band}"
            if band != "unavailable"
            else ""
        )
        case = case_lookup.get(case_id) if case_id else None
        matched = variant_found and case is not None

        if mean_beta is None:
            prediction = f"No numeric beta value was available for {source_row['label'].lower()}."
        elif not variant_found:
            prediction = (
                f"{source_row['label']} was computed, but the predictive thesis matrix only matches after "
                f"a GT-confirmed non-reference {gene_name} genotype is observed."
            )
        elif case is not None:
            prediction = str(case.get("prediction", "")).strip()
        else:
            prediction = (
                f"No bundled predictive thesis case is available for {source_row['label']} with a {band} methylation band."
            )

        research_focus_items = (
            _dedupe_text_items(case.get("research_focus", []))[:3]
            if case is not None
            else []
        )

        methylation_prediction_rows.append(
            {
                "metric_key": source_row["metric_key"],
                "metric_label": source_row["label"],
                "mean_beta": _round_beta(mean_beta),
                "mean_beta_display": _format_predictive_beta_display(mean_beta),
                "probe_count": source_row["probe_count"],
                "band": band,
                "band_display": band.title() if band != "unavailable" else "Unavailable",
                "prediction": prediction,
                "matched": matched,
                "matched_case_label": str(case.get("label", "")).strip() if case is not None else "",
                "research_focus": "; ".join(research_focus_items),
            }
        )

        if matched and case is not None:
            matched_cases.append(
                {
                    "case_label": str(case.get("label", "")).strip(),
                    "trigger": f"Variant found + {source_row['label']}",
                    "source": source_row["label"],
                    "mean_beta_display": _format_predictive_beta_display(mean_beta),
                    "band": band.title(),
                    "prediction": str(case.get("prediction", "")).strip(),
                    "research_focus": "; ".join(research_focus_items),
                }
            )

    if matched_cases:
        summary = (
            f"{gene_name} matched {len(matched_cases)} predictive thesis case(s) in this run: "
            f"the base variant case plus {max(len(matched_cases) - 1, 0)} methylation-linked case(s)."
        )
    elif variant_found:
        summary = (
            f"GT-confirmed {gene_name} non-reference variation was observed, but none of the bundled predictive thesis cases could be matched "
            "to the available methylation values."
        )
    else:
        summary = (
            f"No predictive thesis case matched because the current {gene_name} run did not surface a GT-confirmed promoter or gene-body non-reference genotype."
        )

    return {
        "gene_name": gene_name,
        "database_name": synthesis_database.get(
            "database_name",
            f"No curated {gene_name} predictive synthesis database loaded",
        ),
        "database_version": synthesis_database.get("version", "generic"),
        "matching_rule": synthesis_database.get(
            "matching_rule",
            "Cases match only when a gene variant is present, plus the requested methylation source resolves to a low, medium, or high beta band.",
        ),
        "disclaimer": synthesis_database.get(
            "disclaimer",
            "Predictive theses in this app are literature-guided research summaries, not diagnostic claims.",
        ),
        "seeded_markers": synthesis_database.get("seeded_markers", []),
        "concrete_variant_prediction": synthesis_database.get("concrete_variant_prediction", ""),
        "variant_found": variant_found,
        "variant_found_label": "Yes" if variant_found else "No",
        "variant_summary": variant_summary,
        "phenotype_prediction": phenotype_prediction,
        "phenotype_prediction_text": phenotype_prediction.get("phenotype_prediction", ""),
        "phenotype_confidence": phenotype_prediction.get("confidence", "unknown"),
        "phenotype_evidence_summary": phenotype_prediction.get("evidence_summary", ""),
        "phenotype_uncertainty_summary": phenotype_prediction.get("uncertainty_summary", ""),
        "phenotype_conflicting_evidence": phenotype_prediction.get("conflicting_evidence", []),
        "phenotype_qc_warnings": phenotype_prediction.get("qc_warnings", []),
        "variant_prediction_rows": variant_prediction_rows,
        "methylation_prediction_rows": methylation_prediction_rows,
        "matched_cases": matched_cases,
        "matched_case_count": len(matched_cases),
        "case_catalog_size": len(case_catalog),
        "summary": summary,
    }


def _join_unique_database_values(values: list[Any]) -> str:
    """Join compact central-database values while preserving order."""
    cleaned_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = "" if value is None else str(value).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        cleaned_values.append(text)
    return "; ".join(cleaned_values)


def _format_database_beta(value: Any) -> float | str:
    """Return beta values as stable numeric CSV fields when available."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value).strip()


def _format_database_quality(value: Any) -> str:
    """Render VCF quality values for the central variant-level database."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value).strip()


def _format_database_position(value: Any) -> str:
    """Render a stable genomic coordinate for a central database row."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def _build_observed_variant_key(row: pd.Series) -> str:
    """Return a biologically stable key for one observed variant allele."""
    chrom = str(row.get("chrom", "")).strip().removeprefix("chr")
    pos = _format_database_position(row.get("pos"))
    ref = "" if _is_missing_value(row.get("ref")) else str(row.get("ref")).strip()
    alt = "" if _is_missing_value(row.get("alt")) else str(row.get("alt")).strip()
    sample = "" if _is_missing_value(row.get("sample")) else str(row.get("sample")).strip()
    gt_raw = "" if _is_missing_value(row.get("gt_raw")) else str(row.get("gt_raw")).strip()
    suffix = f":{sample}" if sample else ""
    gt_suffix = f":GT={gt_raw}" if gt_raw else ""
    if chrom and pos and (ref or alt):
        return f"chr{chrom}:{pos}:{ref}>{alt}{suffix}{gt_suffix}"
    return _format_variant_display(row)


def _format_variant_location_for_database(row: pd.Series) -> str:
    """Render the point location used for variant-level central database rows."""
    chrom = str(row.get("chrom", "")).strip().removeprefix("chr")
    pos = _format_database_position(row.get("pos"))
    if chrom and pos:
        return f"chr{chrom}:{int(pos):,}" if pos.isdigit() else f"chr{chrom}:{pos}"
    return ""


def _build_general_analysis_database_rows(
    *,
    gene_name: str,
    variants: pd.DataFrame,
    variant_interpretations: dict[str, Any],
    methylation_insights: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build central database rows with one entry per observed variant."""
    normalized_gene_name = gene_name.strip().upper() or DEFAULT_GENE_NAME
    variants = _ensure_variant_genotype_annotations(variants)

    gene_region = variant_interpretations.get("gene_region", {})
    search_region = variant_interpretations.get("search_region", {})
    gene_location = str(
        gene_region.get("display")
        or search_region.get("display")
        or variant_interpretations.get("recommended_promoter_plus_gene_region")
        or ""
    ).strip()

    matched_records_by_variant = {
        str(record.get("observed_variant", "")).strip(): record
        for record in variant_interpretations.get("matched_records", [])
        if str(record.get("observed_variant", "")).strip()
    }

    output_rows: list[dict[str, Any]] = []
    for _, row in variants.iterrows():
        observed_variant = _format_variant_display(row)
        matched_record = matched_records_by_variant.get(observed_variant, {})
        variant_label = _format_variant_label(row.get("id"))
        chrom = str(row.get("chrom", "")).strip().removeprefix("chr")
        position = _format_database_position(row.get("pos"))
        output_rows.append(
            {
                "gene": normalized_gene_name,
                "sample": str(row.get("sample", "") or "").strip(),
                "variant key": _build_observed_variant_key(row),
                "observed gene variant": observed_variant,
                "gene variant label": variant_label,
                "change": _format_variant_change(row.get("ref"), row.get("alt")),
                "genotype": str(row.get("genotype", "Unavailable")).strip(),
                "zygosity": str(row.get("zygosity", "missing")).strip(),
                "allele dosage": _format_allele_dosage(row.get("allele_dosage_per_alt", {})),
                "chromosome": f"chr{chrom}" if chrom else "",
                "position": position,
                "variant location": _format_variant_location_for_database(row),
                "gene location": gene_location,
                "source": "VCF",
                "(VCF) quality (qual)": _format_database_quality(row.get("qual")),
                "VCF filter": str(row.get("filter_status", "") or "").strip(),
                "VCF depth (DP)": "" if _is_missing_value(row.get("dp")) else row.get("dp"),
                "VCF allele depths (AD)": _join_unique_database_values(list(row.get("ad") or [])),
                "VCF genotype quality (GQ)": "" if _is_missing_value(row.get("gq")) else row.get("gq"),
                "genotype confidence": "" if _is_missing_value(row.get("confidence_score")) else row.get("confidence_score"),
                "genotype QC flags": _join_unique_database_values(list(row.get("qc_flags") or [])),
                "matched curated marker": str(matched_record.get("variant", "")).strip(),
                "variant interpretation scope": str(
                    matched_record.get("interpretation_scope", "Unclassified observed variant")
                ).strip(),
                "curated biological significance": str(
                    matched_record.get("clinical_significance")
                    or matched_record.get("clinical_interpretation")
                    or f"No curated local {normalized_gene_name} significance is bundled for this observed variant."
                ).strip(),
                "functional effects": _join_unique_database_values(
                    list(matched_record.get("functional_effects") or [])
                ),
                "associated conditions": _join_unique_database_values(
                    list(matched_record.get("associated_conditions") or [])
                ),
                "methylation-linked probes": _join_unique_database_values(
                    list(matched_record.get("relevant_probe_ids") or [])
                ),
                "mean beta whitelist": _format_database_beta(methylation_insights.get("whitelist_mean_beta")),
                "mean beta related to gene": _format_database_beta(methylation_insights.get("gene_name_mean_beta")),
                "mean beta on found probes in the area (numerical rows)": _format_database_beta(
                    methylation_insights.get("all_numeric_mean_beta")
                ),
            }
        )
    return output_rows


def update_general_analysis_database(
    *,
    gene_name: str,
    variants: pd.DataFrame,
    variant_interpretations: dict[str, Any],
    methylation_insights: dict[str, Any],
    overwrite: bool = False,
    database_path: str | Path = GENERAL_ANALYSIS_DATABASE_PATH,
) -> dict[str, Any]:
    """Add or optionally replace observed variant rows in the central database."""
    output_path = Path(database_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_gene_name = gene_name.strip().upper() or DEFAULT_GENE_NAME
    new_rows = _build_general_analysis_database_rows(
        gene_name=normalized_gene_name,
        variants=variants,
        variant_interpretations=variant_interpretations,
        methylation_insights=methylation_insights,
    )
    if not new_rows:
        return {
            "action": "skipped_empty",
            "path": output_path,
            "message": f"Central database did not add {normalized_gene_name}; no observed variant rows were available.",
        }

    if output_path.exists():
        try:
            database = pd.read_csv(output_path, dtype=object)
        except pd.errors.EmptyDataError:
            database = pd.DataFrame(columns=GENERAL_ANALYSIS_DATABASE_COLUMNS)
    else:
        database = pd.DataFrame(columns=GENERAL_ANALYSIS_DATABASE_COLUMNS)

    for column in GENERAL_ANALYSIS_DATABASE_COLUMNS:
        if column not in database.columns:
            database[column] = ""

    primary_columns = list(GENERAL_ANALYSIS_DATABASE_COLUMNS)
    extra_columns = [column for column in database.columns if column not in primary_columns]
    database = database[primary_columns + extra_columns].fillna("")

    existing_mask = database["gene"].astype(str).str.strip().str.upper() == normalized_gene_name
    if overwrite and existing_mask.any():
        database = database.loc[~existing_mask].copy()
        database = pd.concat([database, pd.DataFrame(new_rows)], ignore_index=True)
        database = database[primary_columns + extra_columns].fillna("")
        database.to_csv(output_path, index=False)
        return {
            "action": "overwritten",
            "path": output_path,
            "message": (
                f"Overwrote {len(new_rows)} observed {normalized_gene_name} variant row(s) "
                "in the central analysis database."
            ),
        }

    legacy_gene_mask = existing_mask & (
        database["variant key"].astype(str).str.strip() == ""
    )
    if legacy_gene_mask.any():
        database = database.loc[~legacy_gene_mask].copy()
        existing_mask = database["gene"].astype(str).str.strip().str.upper() == normalized_gene_name

    existing_variant_keys = set(
        database.loc[existing_mask, "variant key"].astype(str).str.strip().str.casefold()
    )
    rows_to_add = [
        row
        for row in new_rows
        if str(row.get("variant key", "")).strip().casefold() not in existing_variant_keys
    ]

    if not rows_to_add:
        return {
            "action": "skipped_existing",
            "path": output_path,
            "message": (
                f"Central database already contains all {len(new_rows)} observed {normalized_gene_name} "
                "variant row(s); existing entries were kept. Check overwrite in Run Analysis to replace them."
            ),
        }

    database = pd.concat([database, pd.DataFrame(rows_to_add)], ignore_index=True)
    database = database[primary_columns + extra_columns].fillna("")
    database.to_csv(output_path, index=False)

    skipped_count = len(new_rows) - len(rows_to_add)
    skipped_note = f" Kept {skipped_count} existing row(s)." if skipped_count else ""
    return {
        "action": "added",
        "path": output_path,
        "message": (
            f"Added {len(rows_to_add)} observed {normalized_gene_name} variant row(s) "
            f"to the central analysis database.{skipped_note}"
        ),
    }


def load_variants(vcf_path: str, region: str) -> pd.DataFrame:
    """Load VCF calls for the requested region with sample-level genotype state.

    Parameters
    ----------
    vcf_path : str
        Filesystem path to a bgzip-compressed and tabix-indexed VCF.
    region : str
        Region string understood by ``scikit-allel``.

    Returns
    -------
    pd.DataFrame
        Variant table with one row per site/sample call. REF and ALT describe
        site alleles; FORMAT/GT and quality fields describe the sample call.

    Raises
    ------
    AnalysisError
        Raised when the VCF cannot be found or parsed. A valid region with no
        calls returns an empty table so the rest of the report can still explain
        the no-variant result.
    """
    if not os.path.exists(vcf_path):
        raise AnalysisError(f"VCF not found: {vcf_path}")

    try:
        callset = allel.read_vcf(
            vcf_path,
            region=region,
            alt_number=10,
            fields=[
                "variants/CHROM",
                "variants/ID",
                "variants/POS",
                "variants/REF",
                "variants/ALT",
                "variants/QUAL",
                "variants/FILTER_PASS",
                "calldata/GT",
                "calldata/AD",
                "calldata/DP",
                "calldata/GQ",
                "calldata/PL",
                "calldata/GP",
                "calldata/AF",
                "calldata/SB",
                "calldata/F1R2",
                "calldata/F2R1",
            ],
        )
    except Exception as exc:
        raise AnalysisError(f"Failed to read VCF '{vcf_path}' for region '{region}': {exc}") from exc

    if not callset or "variants/CHROM" not in callset:
        logger.info("No VCF calls were present in %s for region %s", vcf_path, region)
        return _empty_variant_dataframe()

    sample_names = [_decode_scalar(sample) for sample in callset.get("samples", [])]
    if not sample_names:
        sample_names = [""]
    raw_sample_fields = _load_raw_vcf_sample_fields(vcf_path, region)

    def _calldata_value(field: str, variant_index: int, sample_index: int) -> Any:
        payload = callset.get(f"calldata/{field}")
        if payload is None:
            return None
        try:
            if payload.ndim >= 2:
                return payload[variant_index, sample_index]
            return payload[variant_index]
        except Exception:
            return None

    rows: list[dict[str, Any]] = []
    for variant_index, chrom_value in enumerate(callset["variants/CHROM"]):
        alt_alleles = _normalize_alt_alleles(callset["variants/ALT"][variant_index])
        filter_pass_values = callset.get("variants/FILTER_PASS")
        filter_pass = bool(filter_pass_values[variant_index]) if filter_pass_values is not None else False
        for sample_index, sample_name in enumerate(sample_names):
            gt_codes = _as_clean_list(_calldata_value("GT", variant_index, sample_index))
            gt_codes = [
                None if _clean_vcf_text(code) in {"", "."} else _safe_int(code)
                for code in gt_codes
            ]
            row = {
                "sample": sample_name,
                "chrom": _decode_scalar(chrom_value),
                "id": (
                    None
                    if _decode_scalar(callset["variants/ID"][variant_index]) in {None, "."}
                    else _decode_scalar(callset["variants/ID"][variant_index])
                ),
                "pos": callset["variants/POS"][variant_index],
                "ref": _decode_scalar(callset["variants/REF"][variant_index]),
                "alt": _format_alt_alleles(alt_alleles),
                "alt_alleles": alt_alleles,
                "qual": callset["variants/QUAL"][variant_index],
                "filter_pass": filter_pass,
                "filter_status": "PASS" if filter_pass else "Non-PASS",
                "gt_codes": gt_codes,
                "gt_raw": _format_gt_from_codes(gt_codes, phased=False) if gt_codes else "./.",
                "ad": _as_clean_list(_calldata_value("AD", variant_index, sample_index)),
                "dp": _calldata_value("DP", variant_index, sample_index),
                "gq": _calldata_value("GQ", variant_index, sample_index),
                "pl": _as_clean_list(_calldata_value("PL", variant_index, sample_index)),
                "gp": _as_clean_list(_calldata_value("GP", variant_index, sample_index)),
                "sample_af": _as_clean_list(_calldata_value("AF", variant_index, sample_index)),
                "sb": _as_clean_list(_calldata_value("SB", variant_index, sample_index)),
                "f1r2": _as_clean_list(_calldata_value("F1R2", variant_index, sample_index)),
                "f2r1": _as_clean_list(_calldata_value("F2R1", variant_index, sample_index)),
            }
            raw_overlay = raw_sample_fields.get(
                _sample_field_lookup_key(
                    chrom=row["chrom"],
                    pos=row["pos"],
                    ref=row["ref"],
                    alt=row["alt"],
                    sample=row["sample"],
                ),
                {},
            )
            if raw_overlay:
                row.update({key: value for key, value in raw_overlay.items() if value is not None})
            rows.append(row)

    df = _ensure_variant_genotype_annotations(pd.DataFrame(rows))
    if df.empty:
        logger.info("No VCF sample calls were present in %s for region %s", vcf_path, region)
        return _empty_variant_dataframe()

    named_id_count = int(df["id"].notna().sum())
    pass_count = int(df["filter_pass"].fillna(False).sum()) if "filter_pass" in df else 0
    non_reference_count = int(
        sum(_genotype_has_alt_dosage(row) for _, row in df.iterrows())
    )
    logger.info(
        "Loaded %d VCF sample call(s) from %s in region %s (%d PASS, %d GT-confirmed non-reference, %d named IDs, %d unlabeled in source VCF)",
        len(df),
        vcf_path,
        region,
        pass_count,
        non_reference_count,
        named_id_count,
        len(df) - named_id_count,
    )
    return df


def _gene_manifest_filename(gene_name: str, genome_build: str = "hg19") -> str:
    """Build the conventional gene-specific manifest subset filename."""
    return f"{sanitize_gene_name_for_filename(gene_name)}_epigenetics_{genome_build}.csv"


def _allows_empty_methylation_subset(gene_name: str) -> bool:
    """Return whether a curated gene can use a zero-row methylation subset."""
    knowledge_base = load_gene_interpretation_database(gene_name.strip().upper() or DEFAULT_GENE_NAME)
    if knowledge_base is None:
        return False

    gene_context = knowledge_base.get("gene_context", {})
    chromosome = str(gene_context.get("chromosome", "")).strip().upper()
    relevant_probe_ids = gene_context.get("relevant_methylation_probe_ids")
    return chromosome in {"M", "MT"} or relevant_probe_ids == []


def _prepare_gene_manifest_subset(
    *,
    manifest_filepath: str,
    gene_name: str,
    region: str,
    genome_build: str = "hg19",
) -> Path:
    """Save the selected gene-region manifest subset into `src/gene_data`.

    The full manifest path is still passed through to methylprep, but the
    analysis also keeps a smaller gene-specific CSV next to the bundled
    knowledge-base files so the post-pipeline annotation join can reuse it.
    """
    try:
        selection = save_filtered_manifest(
            gene_name=gene_name,
            manifest_path=manifest_filepath,
            region=region,
            genome_build=genome_build,
            output_dir=GENE_DATA_DIR,
            allow_empty=_allows_empty_methylation_subset(gene_name),
        )
    except Exception as exc:
        raise AnalysisError(
            f"Failed to prepare the {gene_name} methylation subset from '{manifest_filepath}': {exc}"
        ) from exc

    output_path = Path(selection["output_path"])
    logger.info("Saved %s gene manifest subset to %s", gene_name, output_path)
    return output_path


def _run_methylprep_pipeline(data_dir: str, *, manifest_filepath: str | None) -> pd.DataFrame:
    """Run methylprep with the requested manifest override, if any."""
    return run_pipeline(
        data_dir,
        export=True,
        betas=True,
        manifest_filepath=manifest_filepath,
    )


def load_methylation_beta_values(
    idat_base: str,
    manifest_filepath: str | None = None,
) -> pd.DataFrame:
    """Process one IDAT pair and return a reusable probe/beta table."""
    logger.info("Starting methylation loading for sample")
    data_dir = os.path.dirname(idat_base) or "."
    sample_name = os.path.basename(idat_base)

    for suffix in ("_Grn.idat", "_Red.idat"):
        path = os.path.join(data_dir, sample_name + suffix)
        logger.debug("Checking IDAT: %s", path)
        if not os.path.isfile(path):
            raise AnalysisError(f"Missing IDAT file: {path}")

    pipeline_manifest_path = manifest_filepath
    try:
        logger.info("Running methylprep pipeline with betas=True")
        beta_values = _run_methylprep_pipeline(
            data_dir,
            manifest_filepath=pipeline_manifest_path,
        )
    except Exception as exc:
        if not pipeline_manifest_path:
            logger.exception("run_pipeline failed")
            raise AnalysisError(f"methylprep failed for sample '{sample_name}': {exc}") from exc

        logger.warning(
            "methylprep rejected custom manifest '%s'; retrying with methylprep's default manifest.",
            pipeline_manifest_path,
            exc_info=True,
        )
        try:
            beta_values = _run_methylprep_pipeline(data_dir, manifest_filepath=None)
        except Exception as retry_exc:
            logger.exception("run_pipeline failed even after retrying without a custom manifest")
            raise AnalysisError(
                f"methylprep failed for sample '{sample_name}' with custom manifest "
                f"'{pipeline_manifest_path}', and the retry without that manifest also failed: {retry_exc}"
            ) from retry_exc

    if sample_name not in beta_values.columns:
        raise AnalysisError(f"No beta column named '{sample_name}' in methylprep output")

    beta_df = beta_values[sample_name].rename("beta").reset_index()
    beta_df = beta_df.rename(columns={beta_df.columns[0]: "probe_id"})
    logger.info("Loaded %d methylation beta value(s) for %s", len(beta_df), sample_name)
    return beta_df


def build_gene_methylation_table(
    beta_values: pd.DataFrame,
    manifest_region: pd.DataFrame,
    *,
    region: str,
    genome_build: str = "hg19",
) -> pd.DataFrame:
    """Join reusable beta values to one region-specific manifest table."""
    manifest_region = manifest_region.copy()
    normalized_build = str(genome_build or "hg19").strip().lower()
    chrom_column = "CHR_hg38" if normalized_build == "hg38" else "CHR"
    position_column = "Start_hg38" if normalized_build == "hg38" else "MAPINFO"
    if {chrom_column, position_column}.issubset(manifest_region.columns):
        region_chrom, region_start, region_end = parse_region_string(region)
        region_chrom = str(region_chrom)
        if normalized_build == "hg38" and not region_chrom.startswith("chr"):
            region_chrom = f"chr{region_chrom}"
        elif normalized_build != "hg38":
            region_chrom = region_chrom.removeprefix("chr")
        region_positions = pd.to_numeric(manifest_region[position_column], errors="coerce")
        manifest_region = manifest_region[
            (manifest_region[chrom_column].astype(str) == region_chrom)
            & (region_positions >= region_start)
            & (region_positions <= region_end)
        ].copy()

    manifest_region = manifest_region.rename(
        columns={
            "IlmnID": "probe_id",
            chrom_column: "chrom",
            position_column: "pos",
            "UCSC_RefGene_Name": "gene",
        }
    )
    required = {"probe_id", "chrom", "pos", "gene"}
    missing = required - set(manifest_region.columns)
    if missing:
        raise AnalysisError(f"Region manifest is missing required columns: {sorted(missing)}")

    merged = pd.merge(beta_values, manifest_region, on="probe_id", how="inner")
    logger.info("After merging, %d probes remain", len(merged))
    keep_columns = [
        "probe_id",
        "beta",
        "chrom",
        "pos",
        "GencodeBasicV12_NAME",
        "UCSC_RefGene_Name",
        "UCSC_RefGene_Accession",
        "UCSC_RefGene_Group",
        "UCSC_CpG_Islands_Name",
        "Relation_to_UCSC_CpG_Island",
        "Phantom4_Enhancers",
        "Phantom5_Enhancers,DMR,450k_Enhancer,HMM_Island",
        "DNase_Hypersensitivity_NAME",
        "DNase_Hypersensitivity_Evidence_Count",
    ]
    available = set(merged.columns)
    missing_columns = [column for column in keep_columns if column not in available]
    if missing_columns:
        logger.warning("Some expected columns are missing: %s", missing_columns)
    return merged[[column for column in keep_columns if column in available]]


def load_full_methylation_manifest(manifest_filepath: str) -> pd.DataFrame:
    """Load a full Illumina manifest for reuse across a multi-gene job."""
    try:
        return load_manifest(manifest_filepath)
    except Exception as exc:
        raise AnalysisError(f"Failed to read methylation manifest '{manifest_filepath}': {exc}") from exc


def build_gene_manifest_subset(
    manifest: pd.DataFrame,
    *,
    region: str,
    genome_build: str,
) -> pd.DataFrame:
    """Return one gene-region subset from an already loaded full manifest."""
    try:
        chrom, start, end = parse_region_string(region)
        return filter_probes_by_region(manifest, chrom, start, end, genome_build)
    except Exception as exc:
        raise AnalysisError(
            f"Failed to filter methylation manifest for {region} ({genome_build}): {exc}"
        ) from exc


def load_methylation(
    idat_base: str,
    manifest_filepath: str | None = None,
    *,
    gene_name: str = DEFAULT_GENE_NAME,
    region: str = DEFAULT_REGION,
) -> pd.DataFrame:
    """Load methylation beta values and annotate them with a selected manifest subset.

    Parameters
    ----------
    idat_base : str
        Path prefix shared by the two Illumina IDAT files. For a sample stored as
        ``data/R01C01_Grn.idat`` and ``data/R01C01_Red.idat``, pass
        ``data/R01C01``.
    manifest_filepath : str | None, optional
        Optional path to the full Illumina manifest. When provided, the
        function saves a gene-specific subset CSV to ``src/gene_data`` before
        running methylprep, then reuses that subset for the final probe
        annotation join.
    gene_name : str, optional
        Gene symbol used to derive the gene-specific manifest subset filename.
    region : str, optional
        Genomic interval used to select probe rows for the gene-specific
        manifest subset.

    Returns
    -------
    pd.DataFrame
        Probe-level methylation table limited to probes present in the selected
        region-specific manifest subset.

    Raises
    ------
    AnalysisError
        Raised when the IDAT pair is incomplete, methylprep fails, or the local
        manifest subset cannot be loaded.
    """
    normalized_gene_name = gene_name.strip().upper() or DEFAULT_GENE_NAME

    region_manifest_file = (
        Path(__file__).resolve().parent
        / "gene_data"
        / _gene_manifest_filename(normalized_gene_name)
    )
    if manifest_filepath:
        region_manifest_file = _prepare_gene_manifest_subset(
            manifest_filepath=manifest_filepath,
            gene_name=normalized_gene_name,
            region=region,
        )
    beta_df = load_methylation_beta_values(
        idat_base,
        manifest_filepath=manifest_filepath,
    )
    try:
        if region_manifest_file.exists():
            manifest_region = pd.read_csv(region_manifest_file)
        else:
            manifest_region = load_gene_epigenetics_manifest(normalized_gene_name)
    except Exception as exc:
        raise AnalysisError(f"Failed to read region manifest file '{region_manifest_file}': {exc}") from exc
    if manifest_region is None:
        raise AnalysisError(f"Failed to read region manifest file '{region_manifest_file}': file not found")
    return build_gene_methylation_table(beta_df, manifest_region, region=region)


def fetch_population_stats(popstats_source: str, variants: pd.DataFrame) -> Any:
    """Load optional population statistics from a CSV or JSON sidecar file.

    Parameters
    ----------
    popstats_source : str
        Path to a CSV or JSON file containing population-level annotations.
    variants : pd.DataFrame
        Variant table returned by :func:`load_variants`. The current
        implementation does not merge by variant yet, but the argument is kept so
        the function signature already matches the future enrichment step.

    Returns
    -------
    Any
        Parsed CSV as a DataFrame or JSON as a Python object.

    Raises
    ------
    AnalysisError
        Raised when the file does not exist or uses an unsupported extension.
    """
    _ = variants
    popstats_path = Path(popstats_source)
    if not popstats_path.exists():
        raise AnalysisError(f"Population statistics file not found: {popstats_source}")

    suffix = popstats_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(popstats_path)
    if suffix == ".json":
        return json.loads(popstats_path.read_text(encoding="utf-8"))

    raise AnalysisError(
        f"Unsupported population statistics format '{popstats_path.suffix}'. Use CSV or JSON."
    )


def _render_section_table(df: pd.DataFrame, title: str, rows: int | None = 20) -> str:
    """Render a DataFrame preview section for the HTML report."""
    preview_df = df if rows is None else df.head(rows)
    preview = preview_df.to_html(index=False, classes="data-table", border=0)
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f'<div class="report-table-shell">{preview}</div>'
        "</section>"
    )


def _with_preferred_column_order(df: pd.DataFrame, preferred_columns: list[str]) -> pd.DataFrame:
    """Move preferred columns to the front while preserving the remaining order."""
    ordered_columns = [column for column in preferred_columns if column in df.columns]
    ordered_columns.extend(column for column in df.columns if column not in ordered_columns)
    return df.loc[:, ordered_columns]


def _prepare_variant_table_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing IDs so exported reports stay readable."""
    preview_df = _ensure_variant_genotype_annotations(df)
    if "id" in preview_df.columns:
        preview_df["id"] = preview_df["id"].where(preview_df["id"].notna(), "Unlabeled in source VCF")
        preview_df["id"] = preview_df["id"].replace({"": "Unlabeled in source VCF", ".": "Unlabeled in source VCF"})
    return _with_preferred_column_order(
        preview_df,
        [
            "sample",
            "chrom",
            "id",
            "pos",
            "ref",
            "alt",
            "gt_raw",
            "genotype",
            "zygosity",
            "allele_dosage_per_alt",
            "filter_status",
            "qual",
            "dp",
            "ad",
            "sample_af",
            "gq",
            "confidence_score",
            "qc_flags",
            "filter_pass",
            "id_source",
        ],
    )


def _prepare_methylation_table_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stable column ordering to methylation tables for the UI and reports."""
    return _with_preferred_column_order(
        df.copy(),
        [
            "probe_id",
            "beta",
            "chrom",
            "pos",
            "ref",
            "alt",
            "GencodeBasicV12_NAME",
            "UCSC_RefGene_Name",
            "UCSC_RefGene_Group",
            "UCSC_CpG_Islands_Name",
            "Relation_to_UCSC_CpG_Island",
        ],
    )


def _flatten_report_value(value: Any) -> str:
    """Convert structured payload values into compact report-table text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        flattened_items = [_flatten_report_value(item) for item in value]
        flattened_items = [item for item in flattened_items if item]
        return "; ".join(flattened_items)
    if isinstance(value, dict):
        if value.get("label") and value.get("url"):
            return f"{value['label']} ({value['url']})"
        if value.get("paper") and value.get("finding"):
            return f"{value['paper']}: {value['finding']}"
        if value.get("variant") and value.get("summary"):
            return f"{value['variant']}: {value['summary']}"
        if value.get("location_group") and value.get("label"):
            return f"{value['location_group']} - {value['label']}"
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _report_df_from_rows(
    rows: list[dict[str, Any]],
    column_map: list[tuple[str, str]],
) -> pd.DataFrame:
    """Build a report-ready DataFrame from structured rows and user-facing labels."""
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_rows.append(
            {
                label: _flatten_report_value(row.get(source_key))
                for source_key, label in column_map
            }
        )
    if not normalized_rows:
        return pd.DataFrame(columns=[label for _, label in column_map])
    return pd.DataFrame(normalized_rows)


def _render_report_paragraphs(
    title: str,
    paragraphs: list[str],
    *,
    extra_markup: str = "",
) -> str:
    """Render one report section made of paragraphs and optional nested markup."""
    rendered_paragraphs = "".join(
        f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs if str(paragraph).strip()
    )
    if not rendered_paragraphs and not extra_markup:
        return ""
    return f"<section><h2>{html.escape(title)}</h2>{rendered_paragraphs}{extra_markup}</section>"


def _render_variant_interpretation_report(
    variant_interpretations: dict[str, Any],
    population_insights: dict[str, Any],
) -> str:
    """Render the richer variant interpretation block for the exported HTML report."""
    if not variant_interpretations:
        return ""

    nested_sections: list[str] = []
    sample_rows = variant_interpretations.get("sample_highlights", {}).get("result_table_rows", [])
    if sample_rows:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    sample_rows,
                    [
                        ("variant_label", "Variant label"),
                        ("change", "Site definition"),
                        ("genotype", "Decoded genotype"),
                        ("zygosity", "Zygosity"),
                        ("allele_dosage", "ALT dosage"),
                        ("confidence_score", "Call confidence"),
                        ("qc_flags", "QC flags"),
                        ("linked_to", "Linked to"),
                    ],
                ),
                "Sample Results",
                rows=None,
            )
        )

    biorender_visuals = variant_interpretations.get("biorender_visuals") or {}
    if biorender_visuals:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [
                        {
                            "provider": biorender_visuals.get("provider", "BioRender"),
                            "focus": biorender_visuals.get("focus", ""),
                            "template_title": biorender_visuals.get("template_title", ""),
                            "template_url": biorender_visuals.get("template_url", ""),
                            "recommended_icons": biorender_visuals.get("recommended_icons", []),
                            "icon_search_terms": biorender_visuals.get("icon_search_terms", []),
                            "usage_note": biorender_visuals.get("usage_note", ""),
                        }
                    ],
                    [
                        ("provider", "Provider"),
                        ("focus", "Figure focus"),
                        ("template_title", "BioRender template"),
                        ("template_url", "Template URL"),
                        ("recommended_icons", "Recommended icons"),
                        ("icon_search_terms", "Icon search terms"),
                        ("usage_note", "Usage note"),
                    ],
                ),
                "BioRender Figure Starter",
                rows=None,
            )
        )

    matched_records = variant_interpretations.get("matched_records", [])
    if matched_records:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    matched_records,
                    [
                        ("variant", "Curated variant"),
                        ("variant_label", "Variant label"),
                        ("observed_variant", "Observed in run"),
                        ("change", "Site definition"),
                        ("gt_raw", "GT"),
                        ("genotype", "Decoded genotype"),
                        ("zygosity", "Zygosity"),
                        ("allele_dosage", "ALT dosage"),
                        ("filter_status", "FILTER"),
                        ("dp", "DP"),
                        ("ad", "AD"),
                        ("sample_af", "Sample AF"),
                        ("gq", "GQ"),
                        ("pl_or_gp_summary", "PL/GP support"),
                        ("qc_flags", "QC flags"),
                        ("confidence_score", "Confidence score"),
                        ("confidence_explanation", "Confidence explanation"),
                        ("linked_to", "Linked to"),
                        ("interpretation_scope", "Scope"),
                        ("clinical_significance", "Clinical significance"),
                        ("clinical_interpretation", "Clinical interpretation"),
                        ("methylation_interpretation", "Methylation interpretation"),
                        ("associated_conditions", "Associated conditions"),
                        ("functional_effects", "Functional effects"),
                        ("research_context", "Research context"),
                        ("relevant_probe_ids", "Relevant probes"),
                        ("evidence", "Evidence"),
                    ],
                ),
                "Matched Variant Interpretations",
                rows=None,
            )
        )

    curated_markers = variant_interpretations.get("curated_named_markers", [])
    if curated_markers:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    curated_markers,
                    [
                        ("variant", "Variant"),
                        ("common_name", "Common name"),
                        ("observed_in_run", "Observed in run"),
                        ("observed_variants", "Observed labels"),
                        ("genome_location", "Genome location"),
                        ("nucleotide_change", "Nucleotide change"),
                        ("nucleotide_change_basis", "Change basis"),
                        ("reference_allele", "Reference allele"),
                        ("alternate_allele", "Alternate allele"),
                        ("coding_change", "Coding / mtDNA change"),
                        ("protein_change", "Protein change"),
                        ("rsids", "Variant IDs"),
                        ("marker_type", "Marker type"),
                        ("assayability", "Assayability"),
                        ("region_class", "Curated class"),
                        ("clinical_significance", "Clinical significance"),
                        ("clinical_parameter_summary", "Clinical parameters"),
                        ("summary", "Interpretation"),
                        ("associated_conditions", "Associated conditions"),
                        ("functional_effects", "Functional effects"),
                        ("research_context", "Research context"),
                        ("research_links", "Research links"),
                    ],
                ),
                "Curated Marker Catalog",
                rows=None,
            )
        )

    variant_effect_overview = variant_interpretations.get("variant_effect_overview", [])
    if variant_effect_overview:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [{"item": item} for item in variant_effect_overview],
                    [("item", "Variant interpretation overview")],
                ),
                "How This Gene's Variants Are Usually Interpreted",
                rows=None,
            )
        )

    condition_research_overview = variant_interpretations.get("condition_research_overview", [])
    if condition_research_overview:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [{"item": item} for item in condition_research_overview],
                    [("item", "Condition or research theme")],
                ),
                "Conditions and Research Themes",
                rows=None,
            )
        )

    region_recommendations = variant_interpretations.get("region_recommendations", [])
    if region_recommendations:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    region_recommendations,
                    [
                        ("title", "View"),
                        ("region", "Region"),
                        ("purpose", "Purpose"),
                    ],
                ),
                "Region Recommendations",
                rows=None,
            )
        )

    promoter_analysis = variant_interpretations.get("promoter_analysis", {})
    if promoter_analysis.get("found_variants"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    promoter_analysis["found_variants"],
                    [
                        ("display", "Observed variant"),
                        ("variant_label", "Variant label"),
                        ("change", "Site definition"),
                        ("genotype", "Decoded genotype"),
                        ("zygosity", "Zygosity"),
                        ("allele_dosage", "ALT dosage"),
                        ("confidence_score", "Call confidence"),
                        ("position", "Position"),
                        ("linked_to", "Linked to"),
                        ("clinical_significance", "Clinical significance"),
                        ("summary", "Interpretation"),
                    ],
                ),
                promoter_analysis.get("label", "Promoter Analysis") + " - Observed Variants",
                rows=None,
            )
        )
    if promoter_analysis.get("known_variants"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    promoter_analysis["known_variants"],
                    [
                        ("variant", "Variant"),
                        ("common_name", "Common name"),
                        ("position", "Position"),
                        ("clinical_significance", "Clinical significance"),
                        ("summary", "Interpretation"),
                        ("associated_conditions", "Associated conditions"),
                    ],
                ),
                promoter_analysis.get("label", "Promoter Analysis") + " - Curated Variants",
                rows=None,
            )
        )

    gene_analysis = variant_interpretations.get("gene_analysis", {})
    if gene_analysis.get("found_variants"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    gene_analysis["found_variants"],
                    [
                        ("display", "Observed variant"),
                        ("variant_label", "Variant label"),
                        ("change", "Site definition"),
                        ("genotype", "Decoded genotype"),
                        ("zygosity", "Zygosity"),
                        ("allele_dosage", "ALT dosage"),
                        ("confidence_score", "Call confidence"),
                        ("position", "Position"),
                        ("linked_to", "Linked to"),
                        ("clinical_significance", "Clinical significance"),
                        ("summary", "Interpretation"),
                    ],
                ),
                gene_analysis.get("label", "Gene Analysis") + " - Observed Variants",
                rows=None,
            )
        )
    if gene_analysis.get("known_variants"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    gene_analysis["known_variants"],
                    [
                        ("variant", "Variant"),
                        ("common_name", "Common name"),
                        ("position", "Position"),
                        ("clinical_significance", "Clinical significance"),
                        ("summary", "Interpretation"),
                        ("associated_conditions", "Associated conditions"),
                    ],
                ),
                gene_analysis.get("label", "Gene Analysis") + " - Curated Variants",
                rows=None,
            )
        )

    if population_insights and population_insights.get("variant_population_records"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    population_insights["variant_population_records"],
                    [
                        ("display_name", "Variant"),
                        ("observed_in_run", "Observed in run"),
                        ("observed_variants", "Observed labels"),
                        ("effect_allele", "Effect allele"),
                        ("effect_summary", "Population summary"),
                        ("associated_conditions", "Associated conditions"),
                        ("functional_effects", "Functional effects"),
                        ("population_extremes", "Population extremes"),
                    ],
                ),
                "Population Reference Data",
                rows=None,
            )
        )

    if population_insights and population_insights.get("sources"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    population_insights["sources"],
                    [
                        ("label", "Source"),
                        ("url", "URL"),
                    ],
                ),
                "Population Sources",
                rows=None,
            )
        )

    if population_insights and population_insights.get("gene_population_patterns"):
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    population_insights["gene_population_patterns"],
                    [
                        ("variant", "Variant or locus"),
                        ("location_group", "Location group"),
                        ("summary", "Summary"),
                    ],
                ),
                "Gene-wide Population Patterns",
                rows=None,
            )
        )

    return _render_report_paragraphs(
        "Variant Interpretation",
        [
            variant_interpretations.get("sample_highlights", {}).get("summary", ""),
            variant_interpretations.get("summary", ""),
            variant_interpretations.get("gene_summary", ""),
            variant_interpretations.get("clinical_context", ""),
            variant_interpretations.get("curated_named_markers_summary", ""),
            population_insights.get("summary", "") if population_insights else "",
            population_insights.get("gene_population_patterns_intro", "") if population_insights else "",
        ],
        extra_markup="".join(nested_sections),
    )


def _render_methylation_interpretation_report(methylation_insights: dict[str, Any]) -> str:
    """Render the richer methylation interpretation block for the exported HTML report."""
    if not methylation_insights:
        return ""

    nested_sections: list[str] = []
    summary_metric_rows = methylation_insights.get("summary_metric_rows", [])
    if summary_metric_rows:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    summary_metric_rows,
                    [
                        ("metric", "Metric"),
                        ("mean_beta_display", "Mean beta"),
                        ("numeric_values", "Numeric values"),
                        ("summary", "Summary"),
                    ],
                ),
                "Methylation Summary Metrics",
                rows=None,
            )
        )

    whitelist_probe_statuses = methylation_insights.get("whitelist_probe_statuses", [])
    if whitelist_probe_statuses:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    whitelist_probe_statuses,
                    [
                        ("probe_id", "Whitelist probe"),
                        ("observed_in_run", "Observed in run"),
                    ],
                ),
                "Whitelist Probe Status",
                rows=None,
            )
        )

    methylation_effects = methylation_insights.get("methylation_effects", [])
    if methylation_effects:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [{"item": item} for item in methylation_effects],
                    [("item", "Likely biological effect")],
                ),
                "Likely Biological Effects",
                rows=None,
            )
        )

    methylation_condition_research = methylation_insights.get("methylation_condition_research", [])
    if methylation_condition_research:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [{"item": item} for item in methylation_condition_research],
                    [("item", "Condition or research setting")],
                ),
                "Methylation Research Context",
                rows=None,
            )
        )

    evidence = methylation_insights.get("evidence", [])
    if evidence:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    evidence,
                    [
                        ("label", "Source"),
                        ("url", "URL"),
                    ],
                ),
                "Methylation Evidence",
                rows=None,
            )
        )

    whitelist_probe_reference_rows = methylation_insights.get("whitelist_probe_reference_rows", [])
    if whitelist_probe_reference_rows:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    whitelist_probe_reference_rows,
                    [
                        ("probe_id", "Whitelist probe"),
                        ("observed_in_run", "Observed in run"),
                        ("beta", "Observed beta"),
                        ("probe_locus", "Probe locus"),
                        ("linked_variants", "Linked variants"),
                        ("nearby_manifest_variants", "Nearby manifest variants"),
                        ("papers", "Bundled papers"),
                    ],
                ),
                "Whitelist Probe Reference Map",
                rows=None,
            )
        )

    probe_preview = methylation_insights.get("probe_preview")
    if isinstance(probe_preview, pd.DataFrame) and not probe_preview.empty:
        nested_sections.append(
            _render_section_table(
                _prepare_methylation_table_for_output(probe_preview),
                "Observed Whitelist Probe Rows",
                rows=None,
            )
        )

    return _render_report_paragraphs(
        "Methylation Interpretation",
        [
            methylation_insights.get("summary", ""),
            methylation_insights.get("clinical_context", ""),
            methylation_insights.get("whitelist_explanation", ""),
            methylation_insights.get("whitelist_literature_context", ""),
            methylation_insights.get("gene_name_match_rule", ""),
            methylation_insights.get("whitelist_probe_reference_summary", ""),
        ],
        extra_markup="".join(nested_sections),
    )


def _render_predictive_theses_report(predictive_theses: dict[str, Any]) -> str:
    """Render the predictive-thesis panels in the exported HTML report."""
    if not predictive_theses:
        return ""

    nested_sections: list[str] = []
    phenotype_prediction = predictive_theses.get("phenotype_prediction")
    if isinstance(phenotype_prediction, dict) and phenotype_prediction:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [phenotype_prediction],
                    [
                        ("phenotype_prediction", "Phenotype prediction"),
                        ("confidence", "Confidence"),
                        ("confidence_score", "Confidence score"),
                        ("evidence_summary", "Evidence summary"),
                        ("uncertainty_summary", "Uncertainty summary"),
                        ("conflicting_evidence", "Conflicting evidence"),
                        ("qc_warnings", "QC warnings"),
                    ],
                ),
                "Phenotype-Level Prediction",
                rows=None,
            )
        )

    variant_prediction_rows = predictive_theses.get("variant_prediction_rows", [])
    if variant_prediction_rows:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    variant_prediction_rows,
                    [
                        ("observed_signal", "Observed signal"),
                        ("genotype", "Decoded genotype"),
                        ("zygosity", "Zygosity"),
                        ("allele_dosage", "ALT dosage"),
                        ("confidence", "Confidence"),
                        ("source", "Source"),
                        ("prediction", "Prediction"),
                        ("research_focus", "Research focus"),
                        ("confidence_explanation", "Confidence explanation"),
                    ],
                ),
                "Variant Prediction",
                rows=None,
            )
        )

    methylation_prediction_rows = predictive_theses.get("methylation_prediction_rows", [])
    if methylation_prediction_rows:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    methylation_prediction_rows,
                    [
                        ("metric_label", "Metric"),
                        ("mean_beta_display", "Mean beta"),
                        ("probe_count", "Numeric values"),
                        ("band_display", "Band"),
                        ("matched_case_label", "Matched case"),
                        ("prediction", "Prediction"),
                        ("research_focus", "Research focus"),
                    ],
                ),
                "Methylation Prediction",
                rows=None,
            )
        )

    matched_cases = predictive_theses.get("matched_cases", [])
    if matched_cases:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    matched_cases,
                    [
                        ("case_label", "Case"),
                        ("trigger", "Trigger"),
                        ("source", "Source"),
                        ("mean_beta_display", "Observed value"),
                        ("band", "Band"),
                        ("prediction", "Prediction"),
                        ("research_focus", "Research focus"),
                    ],
                ),
                "Synthesis",
                rows=None,
            )
        )

    seeded_markers = predictive_theses.get("seeded_markers", [])
    if seeded_markers:
        nested_sections.append(
            _render_section_table(
                _report_df_from_rows(
                    [{"marker": marker} for marker in seeded_markers],
                    [("marker", "Seeded marker")],
                ),
                "Seeded Predictive Markers",
                rows=None,
            )
        )

    return _render_report_paragraphs(
        "Predictive Theses",
        [
            predictive_theses.get("summary", ""),
            predictive_theses.get("variant_summary", ""),
            predictive_theses.get("phenotype_prediction_text", ""),
            predictive_theses.get("phenotype_evidence_summary", ""),
            predictive_theses.get("phenotype_uncertainty_summary", ""),
            predictive_theses.get("matching_rule", ""),
            predictive_theses.get("disclaimer", ""),
        ],
        extra_markup="".join(nested_sections),
    )


def _dynamic_payload_for_report(dynamic_knowledge_base_path: str | Path | None) -> dict[str, Any]:
    """Load workflow metadata from the dynamic KB artifact when available."""
    if not dynamic_knowledge_base_path:
        return {}
    payload = load_dynamic_knowledge_base(dynamic_knowledge_base_path)
    return payload if isinstance(payload, dict) else {}


def _render_dynamic_workflow_report(
    workflow_runs: list[dict[str, Any]],
    *,
    artifact_path: str | Path | None,
) -> str:
    """Render compact per-workflow dynamic KB evidence/status details."""
    if not workflow_runs:
        return ""
    cards: list[str] = []
    for workflow in workflow_runs:
        label = html.escape(str(workflow.get("label") or workflow.get("workflow_key") or "Workflow"))
        status = html.escape(str(workflow.get("status") or "unknown"))
        summary = html.escape(str(workflow.get("summary") or "No workflow summary was generated."))
        source_keys = [str(key) for key in workflow.get("selected_source_keys", [])]
        record_counts = workflow.get("record_counts") if isinstance(workflow.get("record_counts"), dict) else {}
        provider_statuses = workflow.get("provider_statuses") if isinstance(workflow.get("provider_statuses"), list) else []
        provider_rows = []
        for provider in provider_statuses:
            if not isinstance(provider, dict):
                continue
            provider_rows.append(
                "<li>"
                f"<strong>{html.escape(str(provider.get('name') or provider.get('source_key') or 'Provider'))}</strong>: "
                f"{html.escape(str(provider.get('status') or 'unknown'))}"
                f" ({html.escape(str(provider.get('record_count', 0)))} record(s))"
                "</li>"
            )
        warning_items = [
            f"<li>{html.escape(str(warning))}</li>"
            for warning in workflow.get("warnings", [])
            if str(warning).strip()
        ]
        error_items = [
            f"<li>{html.escape(str(error))}</li>"
            for error in workflow.get("errors", [])
            if str(error).strip()
        ]
        cards.append(
            "<details class=\"workflow-card\" open>"
            f"<summary><strong>{label}</strong><span>{status}</span></summary>"
            f"<p>{summary}</p>"
            "<div class=\"workflow-counts\">"
            f"<span>Sources: {len(source_keys)}</span>"
            f"<span>Source records: {html.escape(str(record_counts.get('source_records', 0)))}</span>"
            f"<span>Literature: {html.escape(str(record_counts.get('literature_records', 0)))}</span>"
            f"<span>Population: {html.escape(str(record_counts.get('population_records', 0)))}</span>"
            "</div>"
            f"<p><strong>Sources:</strong> {html.escape(', '.join(source_keys) or 'None selected')}</p>"
            + (f"<ul>{''.join(provider_rows)}</ul>" if provider_rows else "")
            + (f"<p><strong>Warnings</strong></p><ul>{''.join(warning_items)}</ul>" if warning_items else "")
            + (f"<p><strong>Errors</strong></p><ul>{''.join(error_items)}</ul>" if error_items else "")
            + "</details>"
        )
    path_markup = (
        f"<p><strong>Dynamic KB artifact:</strong> {html.escape(str(artifact_path))}</p>"
        if artifact_path
        else ""
    )
    return (
        "<section><h2>Dynamic Workflow Summary</h2>"
        f"{path_markup}"
        "<div class=\"workflow-summary-grid\">"
        f"{''.join(cards)}"
        "</div></section>"
    )


def generate_report(
    variants: pd.DataFrame,
    methylation: pd.DataFrame,
    popstats: Any,
    output_path: str,
    *,
    gene_name: str = DEFAULT_GENE_NAME,
    region: str,
    methylation_output_path: Path | None = None,
    variant_interpretations: dict[str, Any] | None = None,
    methylation_insights: dict[str, Any] | None = None,
    population_insights: dict[str, Any] | None = None,
    predictive_theses: dict[str, Any] | None = None,
    analysis_scope: str = DEFAULT_ANALYSIS_SCOPE,
    dynamic_knowledge_base_status: str = "",
    dynamic_knowledge_base_path: str | Path | None = None,
) -> Path:
    """Generate a report artifact from the assembled analysis tables.

    Parameters
    ----------
    variants : pd.DataFrame
        VCF call table for the requested gene interval, including decoded
        sample-level genotype fields when available.
    methylation : pd.DataFrame
        Annotated probe-level beta-value table returned by
        :func:`load_methylation`.
    popstats : Any
        Optional population statistics payload loaded from CSV or JSON.
    output_path : str
        Destination path for the final report artifact.
    region : str
        Genomic interval used during the run. It is surfaced in the report
        summary so the output remains self-describing.
    gene_name : str, optional
        Gene name displayed in the report heading and summary copy.
    methylation_output_path : Path | None, optional
        Path to the exported methylation CSV, shown in the report when provided.
    predictive_theses : dict[str, Any] | None, optional
        Predictive thesis payload rendered into the report when available.

    Returns
    -------
    Path
        Final report path written to disk.

    Raises
    ------
    AnalysisError
        Raised when the requested report extension is unsupported.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = report_path.suffix.lower() or ".html"
    normalized_analysis_scope = normalize_analysis_scope(analysis_scope)
    analysis_scope_label = get_analysis_scope_label(normalized_analysis_scope)
    prepared_variants = _prepare_variant_table_for_output(variants)
    dynamic_payload = _dynamic_payload_for_report(dynamic_knowledge_base_path)
    dynamic_workflow_runs = [
        workflow
        for workflow in dynamic_payload.get("workflow_runs", [])
        if isinstance(workflow, dict)
    ]
    dynamic_workflow_source_matrix = (
        dynamic_payload.get("workflow_source_matrix", {})
        if isinstance(dynamic_payload.get("workflow_source_matrix"), dict)
        else {}
    )

    popstats_section = ""
    if isinstance(popstats, pd.DataFrame):
        popstats_section = _render_section_table(popstats, "Population Statistics Preview")
    elif popstats is not None:
        payload = html.escape(json.dumps(popstats, indent=2))
        popstats_section = f"<section><h2>Population Statistics Preview</h2><pre>{payload}</pre></section>"

    if suffix == ".html":
        methylation_path_markup = ""
        if methylation_output_path is not None:
            methylation_path_markup = (
                "<p><strong>Methylation CSV:</strong> "
                f"{html.escape(str(methylation_output_path))}</p>"
            )
        dynamic_kb_markup = ""
        if dynamic_knowledge_base_status or dynamic_knowledge_base_path:
            dynamic_kb_markup = (
                "<p><strong>Dynamic knowledge base:</strong> "
                f"{html.escape(dynamic_knowledge_base_status or 'Available')}"
                + (
                    f" ({html.escape(str(dynamic_knowledge_base_path))})"
                    if dynamic_knowledge_base_path
                    else ""
                )
                + "</p>"
            )

        variant_interpretation_section = _render_variant_interpretation_report(
            variant_interpretations or {},
            population_insights or {},
        )
        methylation_interpretation_section = _render_methylation_interpretation_report(
            methylation_insights or {},
        )
        predictive_theses_section = _render_predictive_theses_report(
            predictive_theses or {},
        )

        report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(gene_name)} {html.escape(analysis_scope_label)} Analysis Report</title>
  <style>
    :root {{
      --bg: #f6efe3;
      --panel: rgba(255, 252, 245, 0.92);
      --ink: #1f2a2e;
      --muted: #51666a;
      --accent: #0f766e;
      --accent-2: #c26a3d;
      --line: rgba(31, 42, 46, 0.14);
      --shadow: 0 24px 70px rgba(31, 42, 46, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194, 106, 61, 0.20), transparent 30rem),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.20), transparent 28rem),
        linear-gradient(180deg, #fbf6ee 0%, var(--bg) 100%);
    }}
    main {{
      width: min(98vw, 1800px);
      max-width: none;
      margin: 0 auto;
      padding: 38px 12px 64px;
    }}
    .hero {{
      padding: 28px 30px;
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 1;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 8px 0;
      color: var(--muted);
      max-width: 78rem;
      overflow-wrap: anywhere;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-top: 24px;
    }}
    .metric {{
      padding: 18px 20px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid var(--line);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.8rem;
    }}
    .workflow-summary-grid {{
      display: grid;
      gap: 14px;
    }}
    .workflow-card {{
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.62);
    }}
    .workflow-card summary {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      cursor: pointer;
    }}
    .workflow-card summary span {{
      color: var(--accent);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
    }}
    .workflow-counts {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }}
    .workflow-counts span {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.10);
      color: var(--ink);
      font-size: 0.88rem;
    }}
    section {{
      margin-top: 24px;
      padding: 24px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      width: 100%;
      min-width: 0;
      overflow: hidden;
    }}
    h2 {{
      margin-top: 0;
      font-size: 1.3rem;
      letter-spacing: -0.02em;
    }}
    p, li, td, th, strong {{
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .report-table-shell {{
      width: 100%;
      max-width: 100%;
      overflow-x: auto;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.54);
    }}
    .data-table {{
      width: 100%;
      min-width: 760px;
      border-collapse: collapse;
      font-size: 0.92rem;
      table-layout: fixed;
    }}
    .data-table th,
    .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }}
    .data-table th {{
      background: rgba(15, 118, 110, 0.08);
    }}
    pre {{
      padding: 16px;
      overflow-x: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border-radius: 16px;
      background: #172023;
      color: #f4f0e8;
    }}
    @media (max-width: 760px) {{
      main {{
        width: min(100vw, 100%);
        padding: 18px 8px 42px;
      }}
      .hero,
      section {{
        padding: 18px;
      }}
      .data-table {{
        min-width: 680px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(gene_name)} Analysis Report</h1>
      <p>This report summarizes the current {html.escape(gene_name)} variant and methylation analysis run.</p>
      <p><strong>Report focus:</strong> {html.escape(analysis_scope_label)}</p>
      <p><strong>Region:</strong> {html.escape(region)}</p>
      <p><strong>Report path:</strong> {html.escape(str(report_path))}</p>
      {methylation_path_markup}
      {dynamic_kb_markup}
      <div class="metrics">
        <article class="metric">
          <span>VCF calls</span>
          <strong>{len(variants)}</strong>
        </article>
        <article class="metric">
          <span>Methylation probes</span>
          <strong>{len(methylation)}</strong>
        </article>
        <article class="metric">
          <span>Population stats</span>
          <strong>{"Yes" if popstats is not None else "No"}</strong>
        </article>
      </div>
    </section>
    {_render_section_table(prepared_variants, "Genetic Variant Results", rows=None)}
    {_render_dynamic_workflow_report(dynamic_workflow_runs, artifact_path=dynamic_knowledge_base_path)}
    {variant_interpretation_section}
    {predictive_theses_section}
    {methylation_interpretation_section}
    {_render_section_table(_prepare_methylation_table_for_output(methylation), "Methylation Raw Results", rows=None)}
    {popstats_section}
  </main>
</body>
</html>
"""
        report_path.write_text(report_html, encoding="utf-8")
        return report_path

    if suffix == ".json":
        payload = {
            "region": region,
            "analysis_scope": normalized_analysis_scope,
            "analysis_scope_label": analysis_scope_label,
            "variants": prepared_variants.to_dict(orient="records"),
            "variant_interpretations": variant_interpretations or {},
            "methylation": methylation.to_dict(orient="records"),
            "population_statistics": _serialize_popstats(popstats),
            "methylation_output_path": str(methylation_output_path) if methylation_output_path else None,
            "predictive_theses": predictive_theses or {},
            "dynamic_knowledge_base": {
                "status": dynamic_knowledge_base_status,
                "path": str(dynamic_knowledge_base_path) if dynamic_knowledge_base_path else "",
                "workflow_runs": dynamic_workflow_runs,
                "workflow_source_matrix": dynamic_workflow_source_matrix,
            },
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report_path

    if suffix == ".csv":
        summary = pd.DataFrame(
            [
                {"metric": "region", "value": region},
                {"metric": "analysis_scope", "value": normalized_analysis_scope},
                {"metric": "analysis_scope_label", "value": analysis_scope_label},
                {"metric": "variant_count", "value": len(variants)},
                {"metric": "methylation_probe_count", "value": len(methylation)},
                {"metric": "has_population_stats", "value": popstats is not None},
                {
                    "metric": "methylation_output_path",
                    "value": str(methylation_output_path) if methylation_output_path else "",
                },
                {
                    "metric": "predictive_thesis_matched_cases",
                    "value": (predictive_theses or {}).get("matched_case_count", 0),
                },
                {
                    "metric": "dynamic_knowledge_base_status",
                    "value": dynamic_knowledge_base_status,
                },
                {
                    "metric": "dynamic_workflow_count",
                    "value": len(dynamic_workflow_runs),
                },
            ]
        )
        summary.to_csv(report_path, index=False)
        return report_path

    raise AnalysisError(
        f"Unsupported output format '{report_path.suffix}'. Use .html, .json, or .csv."
    )


def run_analysis(
    *,
    vcf_path: str,
    idat_base: str,
    output_path: str,
    gene_name: str = DEFAULT_GENE_NAME,
    region: str = DEFAULT_REGION,
    popstats_source: str | None = None,
    manifest_filepath: str | None = None,
    analysis_scope: str = DEFAULT_ANALYSIS_SCOPE,
    overwrite_general_database: bool = False,
    general_database_path: str | Path = GENERAL_ANALYSIS_DATABASE_PATH,
    dynamic_knowledge_base_path: str | Path | None = None,
) -> AnalysisResult:
    """Run the end-to-end gene analysis workflow.

    Parameters
    ----------
    vcf_path : str
        Input VCF path containing DRD4-region variants.
    idat_base : str
        Input IDAT prefix without the color suffix.
    output_path : str
        Output report path.
    gene_name : str, optional
        Gene symbol associated with the current run.
    region : str, optional
        Genomic region to inspect.
    analysis_scope : str, optional
        Report focus for the run. The central database is only updated for the
        standard promoter_plus_gene scope.
    popstats_source : str | None, optional
        Optional population statistics sidecar file path.
    manifest_filepath : str | None, optional
        Optional full manifest file path passed through to methylprep and used
        to refresh the gene-specific subset stored in ``src/gene_data``.
    overwrite_general_database : bool, optional
        When true, replace an existing central-database row for the analyzed
        gene. When false, an existing row is left unchanged.
    general_database_path : str | Path, optional
        Destination CSV for the central one-row-per-observed-variant database.

    Returns
    -------
    AnalysisResult
        Structured result containing the in-memory tables plus the generated
        output paths.
    """
    normalized_gene_name = gene_name.strip().upper() or DEFAULT_GENE_NAME
    normalized_analysis_scope = normalize_analysis_scope(analysis_scope)
    analysis_scope_label = get_analysis_scope_label(normalized_analysis_scope)

    variants = load_variants(vcf_path, region)
    methylation = load_methylation(
        idat_base,
        manifest_filepath=manifest_filepath,
        gene_name=normalized_gene_name,
        region=region,
    )
    popstats = fetch_population_stats(popstats_source, variants) if popstats_source else None
    prepared = analyze_prepared_data(
        variants=variants,
        methylation=methylation,
        gene_name=normalized_gene_name,
        region=region,
        analysis_scope=normalized_analysis_scope,
        popstats=popstats,
        update_general_database_enabled=True,
        overwrite_general_database=overwrite_general_database,
        general_database_path=general_database_path,
        dynamic_knowledge_base_path=dynamic_knowledge_base_path,
    )
    variants = prepared.variants
    methylation = prepared.methylation

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    methylation_output_path = _derive_methylation_output_path(report_path)
    methylation_output_path.parent.mkdir(parents=True, exist_ok=True)
    methylation.to_csv(methylation_output_path, index=False)
    logger.info("Saved methylation data to %s", methylation_output_path)

    final_report_path = generate_report(
        variants,
        methylation,
        popstats,
        str(report_path),
        gene_name=normalized_gene_name,
        region=region,
        methylation_output_path=methylation_output_path,
        variant_interpretations=prepared.variant_interpretations,
        methylation_insights=prepared.methylation_insights,
        population_insights=prepared.population_insights,
        predictive_theses=prepared.predictive_theses,
        analysis_scope=normalized_analysis_scope,
        dynamic_knowledge_base_status=prepared.dynamic_knowledge_base_status,
        dynamic_knowledge_base_path=prepared.dynamic_knowledge_base_path,
    )

    return AnalysisResult(
        variants=variants,
        methylation=methylation,
        popstats=popstats,
        report_path=final_report_path,
        methylation_output_path=methylation_output_path,
        region=region,
        analysis_scope=normalized_analysis_scope,
        analysis_scope_label=analysis_scope_label,
        vcf_path=Path(vcf_path),
        idat_base=Path(idat_base),
        variant_interpretations=prepared.variant_interpretations,
        methylation_insights=prepared.methylation_insights,
        knowledge_base=prepared.knowledge_base,
        population_insights=prepared.population_insights,
        population_database=prepared.population_database,
        predictive_theses=prepared.predictive_theses,
        general_database_path=prepared.general_database_path,
        general_database_status=prepared.general_database_status,
        dynamic_knowledge_base_path=prepared.dynamic_knowledge_base_path,
        dynamic_knowledge_base_status=prepared.dynamic_knowledge_base_status,
    )


def analyze_prepared_data(
    *,
    variants: pd.DataFrame,
    methylation: pd.DataFrame,
    gene_name: str,
    region: str,
    analysis_scope: str = DEFAULT_ANALYSIS_SCOPE,
    popstats: Any | None = None,
    update_general_database_enabled: bool = False,
    overwrite_general_database: bool = False,
    general_database_path: str | Path = GENERAL_ANALYSIS_DATABASE_PATH,
    dynamic_knowledge_base_path: str | Path | None = None,
) -> PreparedAnalysisResult:
    """Interpret already loaded variant and methylation tables."""
    normalized_gene_name = gene_name.strip().upper() or DEFAULT_GENE_NAME
    normalized_analysis_scope = normalize_analysis_scope(analysis_scope)
    analysis_scope_label = get_analysis_scope_label(normalized_analysis_scope)

    resolved_dynamic_knowledge_base_path = (
        Path(dynamic_knowledge_base_path) if dynamic_knowledge_base_path else None
    )
    dynamic_payload = load_dynamic_knowledge_base(resolved_dynamic_knowledge_base_path)
    if resolved_dynamic_knowledge_base_path and dynamic_payload is None:
        dynamic_knowledge_base_status = (
            f"Dynamic knowledge base was requested but could not be loaded from "
            f"{resolved_dynamic_knowledge_base_path}."
        )
    elif dynamic_payload:
        provider_count = len(dynamic_payload.get("provider_statuses", []))
        dynamic_knowledge_base_status = (
            f"Merged dynamic knowledge base with {provider_count} provider status record(s)."
        )
    else:
        dynamic_knowledge_base_status = "Dynamic knowledge base was not provided."

    knowledge_base = load_gene_interpretation_database(normalized_gene_name)
    knowledge_base = merge_dynamic_knowledge_base(
        knowledge_base,
        dynamic_payload,
        gene_name=normalized_gene_name,
        region=region,
    )
    if knowledge_base is not None:
        variants = annotate_known_variant_ids(variants, knowledge_base)
        variant_interpretations = build_variant_interpretations(variants, knowledge_base, region=region)
        matched_variant_ids = {
            str(record.get("variant", "")).strip()
            for record in variant_interpretations.get("matched_records", [])
            if str(record.get("variant", "")).strip()
        }
        methylation_insights = build_methylation_insights(
            methylation,
            knowledge_base,
            matched_variant_ids=matched_variant_ids,
        )
    else:
        knowledge_base = {
            "database_name": f"No curated {normalized_gene_name} interpretation database loaded",
            "version": "generic",
            "gene_context": {"gene_name": normalized_gene_name},
        }
        population_database = {
            "database_name": f"No curated {normalized_gene_name} population database loaded",
            "version": "generic",
        }
        variant_interpretations = build_generic_variant_interpretations(
            variants,
            region=region,
            gene_name=normalized_gene_name,
        )
        methylation_insights = build_generic_methylation_insights(
            methylation,
            gene_name=normalized_gene_name,
        )

    synthesis_database = load_gene_synthesis_database(normalized_gene_name)
    predictive_theses = build_predictive_theses(
        variant_interpretations=variant_interpretations,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        synthesis_database=synthesis_database,
    )

    population_database = load_gene_population_database(normalized_gene_name)
    if population_database is not None and knowledge_base.get("version") != "generic":
        population_insights = build_population_insights(variants, knowledge_base, population_database)
    else:
        population_database = {
            "database_name": f"No curated {normalized_gene_name} population database loaded",
            "version": "generic",
        }
        population_insights = build_empty_population_insights(gene_name=normalized_gene_name)

    if update_general_database_enabled and normalized_analysis_scope == DEFAULT_ANALYSIS_SCOPE:
        general_database_result = update_general_analysis_database(
            gene_name=normalized_gene_name,
            variants=variants,
            variant_interpretations=variant_interpretations,
            methylation_insights=methylation_insights,
            overwrite=overwrite_general_database,
            database_path=general_database_path,
        )
    elif not update_general_database_enabled:
        general_database_result = {
            "path": Path(general_database_path),
            "message": "Central database update was not requested for this analysis.",
        }
    else:
        general_database_result = {
            "path": Path(general_database_path),
            "message": (
                f"Central database was not updated for the {analysis_scope_label} focused report; "
                "the general database remains tied to the standard Promoter + gene run."
            ),
        }

    return PreparedAnalysisResult(
        variants=variants,
        methylation=methylation,
        popstats=popstats,
        region=region,
        analysis_scope=normalized_analysis_scope,
        analysis_scope_label=analysis_scope_label,
        variant_interpretations=variant_interpretations,
        methylation_insights=methylation_insights,
        knowledge_base=knowledge_base,
        population_insights=population_insights,
        population_database=population_database,
        predictive_theses=predictive_theses,
        general_database_path=Path(general_database_result["path"]),
        general_database_status=str(general_database_result["message"]),
        dynamic_knowledge_base_path=resolved_dynamic_knowledge_base_path if dynamic_payload else None,
        dynamic_knowledge_base_status=dynamic_knowledge_base_status,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the DRD4 workflow from the command line and return an exit code."""
    try:
        args = parse_args(argv)
        result = run_analysis(
            vcf_path=args.vcf,
            idat_base=args.idat,
            output_path=args.out,
            gene_name=DEFAULT_GENE_NAME,
            region=args.region,
            popstats_source=args.popstats,
            manifest_filepath=args.manifest_file,
            analysis_scope=args.analysis_scope,
        )
    except AnalysisError as exc:
        logger.error("%s", exc)
        return 1

    print(f"Saved report to {result.report_path}")
    print(f"Saved methylation CSV to {result.methylation_output_path}")
    print(f"{result.general_database_status} ({result.general_database_path})")
    return 0


if __name__ == "__main__":
    print(__doc__)
    raise SystemExit(main())
