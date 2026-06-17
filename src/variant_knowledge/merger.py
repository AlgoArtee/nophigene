"""Merge dynamic variant knowledge bases with local curated gene bundles."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _parse_region(region: str) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"(?:chr)?(?P<chrom>[^:]+):(?P<start>[\d,]+)-(?P<end>[\d,]+)", str(region).strip())
    if match is None:
        return None
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    return match.group("chrom"), min(start, end), max(start, end)


def load_dynamic_knowledge_base(path: str | Path | None) -> dict[str, Any] | None:
    """Load a dynamic knowledge-base artifact if a path is available."""
    if not path:
        return None
    payload_path = Path(path)
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _minimal_knowledge_base(
    *,
    gene_name: str,
    region: str,
    dynamic_payload: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_region(region)
    chrom, start, end = parsed if parsed is not None else ("", 1, 1)
    return {
        "database_name": f"NophiGene {gene_name} Dynamic Interpretation Database",
        "version": str(dynamic_payload.get("generated_at") or "dynamic"),
        "gene_context": {
            "gene_name": gene_name,
            "assembly": dynamic_payload.get("genome_build", ""),
            "cytoband": "not curated",
            "chromosome": chrom,
            "gene_region": {
                "label": f"{gene_name} selected interval",
                "start": start,
                "end": end,
                "definition": "Selected preprocessing interval used as a dynamic gene-review span.",
            },
            "promoter_review_region": {
                "label": f"{gene_name} dynamic promoter review unavailable",
                "start": start,
                "end": end,
                "definition": "No curated promoter interval is bundled; dynamic KB uses the selected interval.",
            },
            "promoter_hotspot_region": {
                "label": f"{gene_name} dynamic regulatory review span",
                "start": start,
                "end": end,
                "definition": "Selected interval used for dynamic regulatory review.",
            },
            "recommended_promoter_plus_gene_region": region,
            "gene_summary": (
                f"{gene_name} is being interpreted with a dynamic knowledge base generated during preprocessing."
            ),
            "clinical_context": (
                "Dynamic external-source aggregation is for research triage and provenance, not a diagnostic call."
            ),
            "variant_effect_overview": [
                "Variant records were generated from selected external sources and the current sample loci.",
            ],
            "condition_research_overview": [],
            "relevant_methylation_probe_ids": [
                str(row.get("probe_id"))
                for row in dynamic_payload.get("epigenetic_locus_records", [])
                if row.get("probe_id")
            ][:10],
            "methylation_interpretation": (
                "Methylation values provide local regulatory context only unless a source explicitly supports "
                "a probe-variant mechanism."
            ),
            "methylation_effects": [],
            "methylation_condition_research": [],
            "evidence": [],
        },
        "variant_records": [],
    }


def _lookup_key_set(record: dict[str, Any]) -> set[str]:
    keys = {_clean_text(record.get("variant")).casefold(), _clean_text(record.get("display_name")).casefold()}
    keys.update(_clean_text(value).casefold() for value in record.get("lookup_keys", []) if _clean_text(value))
    return {key for key in keys if key}


def merge_dynamic_knowledge_base(
    knowledge_base: dict[str, Any] | None,
    dynamic_payload: dict[str, Any] | None,
    *,
    gene_name: str,
    region: str,
) -> dict[str, Any] | None:
    """Return a knowledge base augmented with dynamic variant records."""
    if not dynamic_payload:
        return knowledge_base
    merged = copy.deepcopy(knowledge_base) if knowledge_base else _minimal_knowledge_base(
        gene_name=gene_name,
        region=region,
        dynamic_payload=dynamic_payload,
    )
    if merged is None:
        return None

    existing_keys: set[str] = set()
    for record in merged.get("variant_records", []):
        existing_keys.update(_lookup_key_set(record))

    added_records: list[dict[str, Any]] = []
    for record in dynamic_payload.get("variant_records", []):
        if not isinstance(record, dict):
            continue
        keys = _lookup_key_set(record)
        if keys & existing_keys:
            continue
        candidate = copy.deepcopy(record)
        candidate.setdefault("source_database", dynamic_payload.get("database_name", "Dynamic variant knowledge base"))
        candidate.setdefault("dynamic_knowledge_base", True)
        added_records.append(candidate)
        existing_keys.update(keys)

    merged.setdefault("variant_records", []).extend(added_records)
    gene_context = merged.setdefault("gene_context", {})
    gene_context["dynamic_knowledge_base"] = {
        "database_name": dynamic_payload.get("database_name", "Dynamic variant knowledge base"),
        "generated_at": dynamic_payload.get("generated_at", ""),
        "provider_count": len(dynamic_payload.get("provider_statuses", [])),
        "variant_record_count": len(dynamic_payload.get("variant_records", [])),
        "merged_variant_record_count": len(added_records),
    }
    merged["dynamic_knowledge_base"] = {
        "generated_at": dynamic_payload.get("generated_at", ""),
        "provider_statuses": dynamic_payload.get("provider_statuses", []),
        "population_records": dynamic_payload.get("population_records", []),
        "literature_records": dynamic_payload.get("literature_records", []),
        "epigenetic_locus_records": dynamic_payload.get("epigenetic_locus_records", []),
    }
    return merged
