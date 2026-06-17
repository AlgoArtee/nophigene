"""Orchestrate dynamic variant knowledge-base generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .client import RequestClient
from .connectors import connector_for
from .credentials import CredentialResolver, credential_status_for_specs
from .imports import SourceImportBundle, filter_import_records_for_query, parse_source_import
from .models import EpigeneticLocus, KnowledgeQuery, QueryVariant, SourceResult, SourceSpec, utc_now_iso
from .registry import LANE_LABELS, list_source_specs, select_source_specs


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _variant_rsid(value: Any) -> str:
    text = _clean_text(value)
    return text if text.lower().startswith("rs") else ""


def variants_from_dataframe(variants: pd.DataFrame | None) -> tuple[QueryVariant, ...]:
    """Normalize a VCF-derived dataframe into query variants."""
    if variants is None or variants.empty:
        return ()
    normalized: list[QueryVariant] = []
    for _, row in variants.iterrows():
        chrom = _clean_text(row.get("chrom") or row.get("CHROM")).removeprefix("chr")
        pos = _safe_int(row.get("pos") or row.get("POS"))
        if not chrom or pos is None:
            continue
        normalized.append(
            QueryVariant(
                chrom=chrom,
                pos=pos,
                ref=_clean_text(row.get("ref") or row.get("REF")).upper(),
                alt=_clean_text(row.get("alt") or row.get("ALT")).upper(),
                rsid=_variant_rsid(row.get("id") or row.get("ID")),
                sample=_clean_text(row.get("sample")),
                gt_raw=_clean_text(row.get("gt_raw")),
                zygosity=_clean_text(row.get("zygosity")),
            )
        )
    return tuple(normalized)


def epigenetic_loci_from_dataframe(manifest: pd.DataFrame | None, *, limit: int = 80) -> tuple[EpigeneticLocus, ...]:
    """Normalize a filtered manifest dataframe into epigenetic loci."""
    if manifest is None or manifest.empty:
        return ()
    loci: list[EpigeneticLocus] = []
    for _, row in manifest.head(limit).iterrows():
        probe_id = _clean_text(row.get("probe_id") or row.get("IlmnID"))
        chrom = _clean_text(row.get("chrom") or row.get("CHR") or row.get("CHR_hg38")).removeprefix("chr")
        pos = _safe_int(row.get("pos") or row.get("MAPINFO") or row.get("Start_hg38"))
        if not probe_id or not chrom or pos is None:
            continue
        loci.append(
            EpigeneticLocus(
                probe_id=probe_id,
                chrom=chrom,
                pos=pos,
                gene=_clean_text(row.get("gene") or row.get("UCSC_RefGene_Name")),
                relation=_clean_text(row.get("Relation_to_UCSC_CpG_Island") or row.get("UCSC_RefGene_Group")),
            )
        )
    return tuple(loci)


def _source_result_to_records(source_results: list[SourceResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in source_results:
        for record in result.records:
            row = dict(record)
            row.setdefault("source_key", result.source_key)
            rows.append(row)
    return rows


def _record_mentions_variant(record: dict[str, Any], variant: QueryVariant) -> bool:
    haystack = " ".join(_clean_text(value).lower() for value in record.values() if not isinstance(value, (dict, list)))
    if variant.rsid and variant.rsid.lower() in haystack:
        return True
    if variant.coordinate_key.lower() in haystack or variant.change_key.lower() in haystack:
        return True
    return False


def _evidence_from_record(record: dict[str, Any], spec_lookup: dict[str, SourceSpec]) -> dict[str, Any]:
    source_key = str(record.get("source_key", "")).strip()
    spec = spec_lookup.get(source_key)
    label = _clean_text(record.get("label") or record.get("source") or source_key)
    return {
        "label": label[:220],
        "url": _clean_text(record.get("url") or (spec.homepage if spec else "")),
        "source": spec.name if spec else _clean_text(record.get("source") or source_key),
        "source_key": source_key,
        "category": _clean_text(record.get("category")),
        "summary": _clean_text(record.get("summary"))[:600],
        "license_note": spec.license_note if spec else "",
    }


def _clinical_significance_for_variant(records: list[dict[str, Any]]) -> str:
    for record in records:
        text = _clean_text(
            record.get("clinical_significance")
            or record.get("summary")
            or record.get("label")
        )
        if any(token in text.lower() for token in ("pathogenic", "benign", "risk", "drug", "response", "association")):
            return text[:300]
    return "No selected source returned a definitive clinical classification in this dynamic query."


def _build_dynamic_variant_records(
    query: KnowledgeQuery,
    source_records: list[dict[str, Any]],
    specs: tuple[SourceSpec, ...],
) -> list[dict[str, Any]]:
    spec_lookup = {spec.key: spec for spec in specs}
    fallback_records = [
        record for record in source_records if record.get("category") not in {"source_metadata", "literature"}
    ]
    dynamic_records: list[dict[str, Any]] = []
    for variant in query.variants:
        matched_records = [
            record for record in source_records if _record_mentions_variant(record, variant)
        ] or fallback_records[:8]
        evidence = [
            _evidence_from_record(record, spec_lookup)
            for record in matched_records[:12]
        ]
        provider_names = sorted(
            {
                evidence_item["source"]
                for evidence_item in evidence
                if evidence_item.get("source")
            }
        )
        display = variant.label
        dynamic_records.append(
            {
                "variant": display,
                "display_name": display,
                "common_name": f"Dynamic source aggregation for {display}",
                "gene_name": query.gene,
                "chromosome": variant.chrom,
                "position": variant.pos,
                "lookup_keys": variant.lookup_terms(query.gene),
                "region_class": "dynamic_query_variant",
                "interpretation_scope": "Dynamic source aggregation / research triage context",
                "clinical_interpretation": (
                    f"The dynamic knowledge-base builder queried selected external sources for {display}. "
                    "Use this as provenance-rich triage context; do not treat it as a standalone diagnosis "
                    "or prescribing recommendation."
                ),
                "clinical_significance": _clinical_significance_for_variant(matched_records),
                "functional_effects": [
                    f"Selected provider evidence count: {len(evidence)}",
                    "Genotype-specific interpretation still comes from FORMAT/GT and call QC in the analysis layer.",
                ],
                "associated_conditions": [],
                "research_context": [
                    "Dynamic records are generated from live/source metadata at preprocessing time.",
                    "Licensed or unavailable sources are represented by status metadata rather than scraped records.",
                    f"Providers contributing evidence: {', '.join(provider_names) if provider_names else 'none returned records'}",
                ],
                "usual_variant_note": "Automatically generated from the sample's queried variant locus.",
                "methylation_interpretation": (
                    "Pair this sequence variant with nearby methylation only as local regulatory context unless "
                    "a source record explicitly supports a probe-variant mechanism."
                ),
                "is_assayable_in_snp_vcf": True,
                "evidence": evidence,
                "dynamic_source_count": len(provider_names),
            }
        )
    return dynamic_records


def _build_epigenetic_locus_records(query: KnowledgeQuery) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for locus in query.epigenetic_loci:
        rows.append(
            {
                "probe_id": locus.probe_id,
                "chromosome": locus.chrom,
                "position": locus.pos,
                "gene_annotation": locus.gene,
                "relation": locus.relation,
                "summary": (
                    f"{locus.probe_id} is present in the selected manifest subset for {query.gene} "
                    f"at chr{locus.chrom}:{locus.pos}."
                ),
            }
        )
    return rows


def _filter_source_records(source_records: list[dict[str, Any]], categories: set[str]) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in source_records
        if _clean_text(record.get("category")) in categories
    ]


def _parse_region(region: str) -> dict[str, Any]:
    match = re.fullmatch(r"(?:chr)?(?P<chrom>[^:]+):(?P<start>[\d,]+)-(?P<end>[\d,]+)", str(region).strip())
    if not match:
        return {"chrom": "", "start": None, "end": None}
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    return {"chrom": match.group("chrom"), "start": min(start, end), "end": max(start, end)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _has_official_records(result: SourceResult) -> bool:
    if result.status != "ok":
        return False
    return any(record.get("category") != "source_metadata" for record in result.records)


def _source_metadata_record(spec: SourceSpec) -> dict[str, Any]:
    return {
        "category": "source_metadata",
        "source": spec.name,
        "label": spec.name,
        "summary": spec.description or spec.license_note,
        "url": spec.homepage,
        "license_note": spec.license_note,
    }


def _needs_export_result(spec: SourceSpec) -> SourceResult:
    return SourceResult(
        source_key=spec.key,
        status="needs_export",
        message=(
            f"{spec.name} is a licensed or non-open source. Upload a permitted CSV/JSON export "
            "or configure a supported official API connector; scraping is not performed."
        ),
        records=[_source_metadata_record(spec)],
    )


def _import_result(
    spec: SourceSpec,
    bundle: SourceImportBundle,
    query: KnowledgeQuery,
) -> SourceResult:
    records = filter_import_records_for_query(bundle.records, query)
    warnings = list(bundle.warnings)
    if bundle.records and not records:
        warnings.append("The import parsed successfully but no records matched the active gene/variant context.")
    return SourceResult(
        source_key=spec.key,
        status="imported",
        message=f"Imported {len(records)} normalized record(s) from a user-provided {spec.name} export.",
        records=records,
        warnings=warnings,
    )


def _parse_source_imports(
    *,
    specs: tuple[SourceSpec, ...],
    source_imports: dict[str, str | Path] | None,
    generated_at: str,
) -> tuple[dict[str, SourceImportBundle], dict[str, str]]:
    if not source_imports:
        return {}, {}
    spec_lookup = {spec.key: spec for spec in specs}
    bundles: dict[str, SourceImportBundle] = {}
    errors: dict[str, str] = {}
    for raw_key, raw_path in source_imports.items():
        key = _clean_text(raw_key)
        spec = spec_lookup.get(key)
        if spec is None:
            errors[key] = f"Unknown or unselected knowledge source import key: {key}"
            continue
        if "user_export" not in spec.ingestion_modes:
            errors[key] = f"{spec.name} does not support user-export ingestion."
            continue
        try:
            bundles[key] = parse_source_import(
                key,
                raw_path,
                spec=spec,
                imported_at=generated_at,
            )
        except Exception as exc:
            errors[key] = str(exc)
    return bundles, errors


def build_dynamic_knowledge_base(
    *,
    gene: str,
    region: str,
    genome_build: str,
    variants: pd.DataFrame | None = None,
    manifest_subset: pd.DataFrame | None = None,
    selected_sources: list[str] | tuple[str, ...] | None = None,
    credentials: dict[str, str] | None = None,
    source_imports: dict[str, str | Path] | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    request_client: RequestClient | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a dynamic variant knowledge-base payload and optionally write it."""
    normalized_gene = _clean_text(gene).upper()
    normalized_build = _clean_text(genome_build) or "hg19"
    specs = select_source_specs(selected_sources)
    query = KnowledgeQuery(
        gene=normalized_gene,
        region=region,
        genome_build=normalized_build,
        variants=variants_from_dataframe(variants),
        epigenetic_loci=epigenetic_loci_from_dataframe(manifest_subset),
    )
    timestamp = generated_at or utc_now_iso()
    import_bundles, import_errors = _parse_source_imports(
        specs=specs,
        source_imports=source_imports,
        generated_at=timestamp,
    )
    resolver = CredentialResolver(credentials)
    client = request_client or RequestClient()
    source_results: list[SourceResult] = []
    for spec in specs:
        credential = resolver.resolve(spec)
        connector = connector_for(spec, client, credential)
        try:
            source_results.append(connector.query(query))
        except Exception as exc:
            source_results.append(
                SourceResult(
                    source_key=spec.key,
                    status="failed",
                    message=str(exc),
                    errors=[str(exc)],
                )
            )
            continue

        if _has_official_records(source_results[-1]):
            continue
        if spec.key in import_bundles:
            source_results[-1] = _import_result(spec, import_bundles[spec.key], query)
            continue
        if spec.key in import_errors:
            source_results[-1] = SourceResult(
                source_key=spec.key,
                status="failed",
                message=f"Could not parse {spec.name} export: {import_errors[spec.key]}",
                errors=[import_errors[spec.key]],
            )
            continue
        if source_results[-1].status == "needs_credentials":
            continue
        if spec.requires_export:
            source_results[-1] = _needs_export_result(spec)
            continue

    source_records = _source_result_to_records(source_results)
    spec_lookup = {spec.key: spec for spec in specs}
    provider_statuses = [
        result.to_status(spec_lookup[result.source_key])
        for result in source_results
        if result.source_key in spec_lookup
    ]
    payload = {
        "schema_version": "1.0",
        "database_name": f"NophiGene Dynamic {normalized_gene} Variant Knowledge Base",
        "gene_name": normalized_gene,
        "region": region,
        "region_parsed": _parse_region(region),
        "genome_build": normalized_build,
        "generated_at": timestamp,
        "variant_records": _build_dynamic_variant_records(query, source_records, specs),
        "epigenetic_locus_records": _build_epigenetic_locus_records(query),
        "population_records": _filter_source_records(
            source_records,
            {"population", "population_frequency", "population_association", "polygenic_score"},
        ),
        "literature_records": _filter_source_records(source_records, {"literature"}),
        "source_records": source_records,
        "provider_statuses": provider_statuses,
        "provenance": {
            "selected_source_keys": [spec.key for spec in specs],
            "credential_statuses": credential_status_for_specs(list_source_specs(), credentials),
            "variant_count": len(query.variants),
            "epigenetic_locus_count": len(query.epigenetic_loci),
            "cache_dir": str(cache_dir) if cache_dir else "",
            "source_imports": [
                bundle.to_provenance()
                for bundle in import_bundles.values()
            ],
            "source_import_errors": dict(sorted(import_errors.items())),
        },
        "source_lanes": LANE_LABELS,
        "license_notes": {
            spec.key: {
                "source": spec.name,
                "access_type": spec.access_type,
                "license_note": spec.license_note,
                "homepage": spec.homepage,
            }
            for spec in specs
        },
    }
    if output_dir is not None:
        output_path = Path(output_dir) / "variant_kb.json"
        payload["artifact_path"] = str(output_path)
        _write_json(output_path, payload)
    return payload
