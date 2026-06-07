"""Reusable gene workflow helpers shared by the UI and REST API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from .analysis import (
        DEFAULT_ANALYSIS_SCOPE,
        load_gene_interpretation_database,
        normalize_analysis_scope,
    )
    from .gene_region_extraction import find_gene_region
except ImportError:
    from analysis import DEFAULT_ANALYSIS_SCOPE, load_gene_interpretation_database, normalize_analysis_scope
    from gene_region_extraction import find_gene_region

GENE_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def normalize_gene_symbol(value: Any) -> str:
    """Normalize and validate one HGNC-style gene symbol."""
    gene = str(value or "").strip().upper()
    if not gene:
        raise ValueError("Gene symbols must not be empty.")
    if not GENE_SYMBOL_PATTERN.fullmatch(gene):
        raise ValueError(
            f"Invalid gene symbol '{value}'. Start with a letter or digit and use only "
            "letters, digits, dots, underscores, or hyphens."
        )
    return gene


def normalize_genome_build(value: Any, *, allow_auto: bool = False) -> str:
    """Normalize common human genome-build labels."""
    cleaned = str(value or "").strip().lower().replace(" ", "")
    if allow_auto and cleaned in {"", "auto"}:
        return "auto"
    if cleaned in {"hg19", "grch37", "grch37/hg19", "hg19/grch37"}:
        return "hg19"
    if cleaned in {"hg38", "grch38", "grch38/hg38", "hg38/grch38"}:
        return "hg38"
    raise ValueError("Genome build must be auto, hg19/GRCh37, or hg38/GRCh38.")


def genome_build_from_knowledge_base(knowledge_base: dict[str, Any] | None) -> str | None:
    """Return the assembly declared by a local knowledge base."""
    assembly = str((knowledge_base or {}).get("gene_context", {}).get("assembly", "")).lower()
    if "hg38" in assembly or "grch38" in assembly:
        return "hg38"
    if "hg19" in assembly or "grch37" in assembly:
        return "hg19"
    return None


def knowledge_base_matches_build(
    knowledge_base: dict[str, Any] | None,
    genome_build: str,
) -> bool:
    """Return whether bundled coordinates are usable for the requested build."""
    declared = genome_build_from_knowledge_base(knowledge_base)
    return declared == normalize_genome_build(genome_build)


def _format_interval(record: dict[str, Any] | None, *, default_chrom: str = "") -> str:
    if not isinstance(record, dict):
        return ""
    chrom = str(record.get("chromosome") or record.get("chrom") or default_chrom).strip()
    try:
        start = int(record["start"])
        end = int(record["end"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not chrom:
        return ""
    return f"{chrom}:{min(start, end)}-{max(start, end)}"


def _parse_region(region: str) -> tuple[str, int, int]:
    match = re.fullmatch(
        r"(?:chr)?(?P<chrom>[^:]+):(?P<start>[\d,]+)-(?P<end>[\d,]+)",
        str(region or "").strip(),
    )
    if match is None:
        raise ValueError(f"Invalid region '{region}'. Use chrom:start-end.")
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    if start < 1 or start > end:
        raise ValueError(f"Invalid region '{region}': start must be positive and not exceed end.")
    return match.group("chrom"), start, end


def format_region_with_padding(region: str, upstream_bp: int = 1000) -> str:
    """Return a generic promoter-plus-gene interval."""
    chrom, start, end = _parse_region(region)
    return f"{chrom}:{max(1, start - upstream_bp)}-{end}"


def _region_union(regions: list[str]) -> str:
    parsed = [_parse_region(region) for region in regions if region]
    if not parsed:
        return ""
    chroms = {chrom.removeprefix("chr") for chrom, _start, _end in parsed}
    if len(chroms) != 1:
        return ""
    chrom = parsed[0][0]
    return f"{chrom}:{min(start for _chrom, start, _end in parsed)}-{max(end for _chrom, _start, end in parsed)}"


def _region_covers(candidate: str, required: list[str]) -> bool:
    try:
        candidate_chrom, candidate_start, candidate_end = _parse_region(candidate)
        for region in required:
            chrom, start, end = _parse_region(region)
            if chrom.removeprefix("chr") != candidate_chrom.removeprefix("chr"):
                return False
            if candidate_start > start or candidate_end < end:
                return False
    except ValueError:
        return False
    return True


def build_scope_regions(
    gene_name: str,
    selected_gene_region: str,
    *,
    genome_build: str,
    knowledge_base: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build promoter, gene-body, and combined regions for one assembly."""
    normalized_gene = normalize_gene_symbol(gene_name)
    normalized_build = normalize_genome_build(genome_build)
    knowledge_base = knowledge_base or load_gene_interpretation_database(normalized_gene)
    if knowledge_base_matches_build(knowledge_base, normalized_build):
        context = knowledge_base.get("gene_context", {})
        chrom = str(context.get("chromosome", "")).strip()
        promoter = _format_interval(context.get("promoter_review_region"), default_chrom=chrom)
        gene = _format_interval(context.get("gene_region"), default_chrom=chrom) or selected_gene_region
        recommended = str(context.get("recommended_promoter_plus_gene_region") or "").strip()
        required = [region for region in (promoter, gene) if region]
        combined = (
            recommended
            if recommended and _region_covers(recommended, required)
            else _region_union(required) or format_region_with_padding(gene)
        )
        return {
            "promoter_plus_gene": combined,
            "promoter_only": promoter,
            "gene_only": gene,
        }

    return {
        "promoter_plus_gene": format_region_with_padding(selected_gene_region),
        "promoter_only": "",
        "gene_only": selected_gene_region,
    }


def resolve_gene_region(
    gene_name: str,
    *,
    genome_build: str = "auto",
    default_genome_build: str = "hg19",
    analysis_scope: str = DEFAULT_ANALYSIS_SCOPE,
    region_override: str | None = None,
) -> dict[str, Any]:
    """Resolve one gene to an assembly-aware active analysis region."""
    gene = normalize_gene_symbol(gene_name)
    requested_build = normalize_genome_build(genome_build, allow_auto=True)
    fallback_build = normalize_genome_build(default_genome_build)
    scope = normalize_analysis_scope(analysis_scope)
    knowledge_base = load_gene_interpretation_database(gene)
    declared_build = genome_build_from_knowledge_base(knowledge_base)
    selected_build = declared_build if requested_build == "auto" and declared_build else requested_build
    if selected_build == "auto":
        selected_build = fallback_build

    if region_override:
        _parse_region(region_override)
        regions = {
            "promoter_plus_gene": "",
            "promoter_only": "",
            "gene_only": "",
        }
        regions[scope] = region_override
        return {
            "gene": gene,
            "genome_build": selected_build,
            "region": region_override,
            "scope": scope,
            "scope_regions": regions,
            "selected_gene_region": region_override,
            "selected_sources": ["Request region override"],
            "candidate_regions": [],
            "curated_coordinates": False,
        }

    if knowledge_base_matches_build(knowledge_base, selected_build):
        context = knowledge_base.get("gene_context", {})
        selected_gene_region = _format_interval(
            context.get("gene_region"),
            default_chrom=str(context.get("chromosome", "")),
        )
        if selected_gene_region:
            regions = build_scope_regions(
                gene,
                selected_gene_region,
                genome_build=selected_build,
                knowledge_base=knowledge_base,
            )
            active_region = regions.get(scope) or regions["promoter_plus_gene"] or selected_gene_region
            return {
                "gene": gene,
                "genome_build": selected_build,
                "region": active_region,
                "scope": scope,
                "scope_regions": regions,
                "selected_gene_region": selected_gene_region,
                "selected_sources": ["Local curated promoter/gene intervals"],
                "candidate_regions": [],
                "curated_coordinates": True,
            }

    lookup = find_gene_region(gene, genome_build=selected_build)
    selected_gene_region = str(lookup["selected_region"])
    regions = build_scope_regions(
        gene,
        selected_gene_region,
        genome_build=selected_build,
        knowledge_base=knowledge_base,
    )
    active_region = regions.get(scope) or regions["promoter_plus_gene"] or selected_gene_region
    return {
        "gene": gene,
        "genome_build": selected_build,
        "region": active_region,
        "scope": scope,
        "scope_regions": regions,
        "selected_gene_region": selected_gene_region,
        "selected_sources": list(lookup.get("selected_sources", [])) + ["Generic upstream promoter heuristic"],
        "candidate_regions": list(lookup.get("candidate_regions", [])),
        "curated_coordinates": False,
    }


def select_profile_variant_source(
    profile: dict[str, Any],
    genome_build: str,
) -> dict[str, Any]:
    """Prefer a matching-build VCF and otherwise return a matching BAM."""
    normalized_build = normalize_genome_build(genome_build)
    for key, source_type in (("vcf_sources", "vcf"), ("bam_sources", "bam")):
        for source in profile.get(key, []):
            if normalize_genome_build(source.get("genome_build")) == normalized_build:
                selected = dict(source)
                selected["type"] = source_type
                selected["path"] = str(Path(selected["path"]))
                return selected
    raise ValueError(
        f"Sample profile '{profile.get('id', '')}' has no VCF or BAM source for {normalized_build}."
    )
