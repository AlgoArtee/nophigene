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
from .local_articles import (
    LOCAL_ARTICLE_SOURCE_KEY,
    LOCAL_ARTICLE_SOURCE_SPEC,
    LOCAL_ARTICLE_WORKFLOW_KEY,
    extract_local_article_evidence,
    source_result_from_local_article_evidence,
    write_local_article_artifacts,
)
from .models import EpigeneticLocus, KnowledgeQuery, QueryVariant, SourceResult, SourceSpec, utc_now_iso
from .registry import LANE_LABELS, list_source_specs, select_source_specs
from .workflows import WorkflowSpec, get_workflow_spec, select_workflow_specs, source_keys_for_workflows


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
        for index, record in enumerate(result.records, start=1):
            row = dict(record)
            row.setdefault("source_key", result.source_key)
            row.setdefault("evidence_id", f"{result.source_key}:{index:03d}")
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
        "evidence_id": _clean_text(record.get("evidence_id")),
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


def _selected_source_specs(
    *,
    selected_sources: list[str] | tuple[str, ...] | None,
    workflows: tuple[WorkflowSpec, ...],
) -> tuple[SourceSpec, ...]:
    """Return source specs from an explicit source selection or workflow union."""
    if selected_sources:
        return select_source_specs(selected_sources)
    workflow_source_keys = source_keys_for_workflows(workflows)
    if workflow_source_keys:
        return select_source_specs(workflow_source_keys)
    return ()


def _workflow_source_matrix(
    *,
    workflows: tuple[WorkflowSpec, ...],
    selected_source_keys: list[str],
) -> dict[str, list[str]]:
    """Return source-to-workflow attribution for the final selected source set."""
    selected = set(selected_source_keys)
    matrix: dict[str, list[str]] = {source_key: [] for source_key in selected_source_keys}
    for workflow in workflows:
        for source_key in workflow.valid_source_keys():
            if source_key in selected:
                matrix.setdefault(source_key, []).append(workflow.key)
    for source_key in selected_source_keys:
        if not matrix[source_key]:
            matrix[source_key] = ["manual_source_override"]
    return matrix


def _execution_specs_for_workflows(
    *,
    specs: tuple[SourceSpec, ...],
    workflows: tuple[WorkflowSpec, ...],
) -> tuple[SourceSpec, ...]:
    """Order provider execution by workflow sequence, then manual-only additions."""
    spec_lookup = {spec.key: spec for spec in specs}
    ordered: list[SourceSpec] = []
    seen: set[str] = set()
    for workflow in workflows:
        for source_key in workflow.valid_source_keys():
            spec = spec_lookup.get(source_key)
            if spec is not None and source_key not in seen:
                seen.add(source_key)
                ordered.append(spec)
    for spec in specs:
        if spec.key not in seen:
            seen.add(spec.key)
            ordered.append(spec)
    return tuple(ordered)


def _local_article_requested(
    *,
    workflows: tuple[WorkflowSpec, ...],
    selected_sources: list[str] | tuple[str, ...] | None,
    use_local_article_evidence: bool,
    article_pdf_folder: str | Path | None,
) -> bool:
    if use_local_article_evidence or _clean_text(article_pdf_folder):
        return True
    if selected_sources and any(_clean_text(source_key) == LOCAL_ARTICLE_SOURCE_KEY for source_key in selected_sources):
        return True
    return any(workflow.key == LOCAL_ARTICLE_WORKFLOW_KEY for workflow in workflows)


def _append_local_article_workflow(workflows: tuple[WorkflowSpec, ...]) -> tuple[WorkflowSpec, ...]:
    if any(workflow.key == LOCAL_ARTICLE_WORKFLOW_KEY for workflow in workflows):
        return workflows
    workflow = get_workflow_spec(LOCAL_ARTICLE_WORKFLOW_KEY)
    return workflows + ((workflow,) if workflow is not None else ())


def _workflow_status(statuses: list[dict[str, Any]]) -> str:
    if not statuses:
        return "skipped"
    usable_statuses = {"ok", "imported", "metadata_only"}
    needs_input_statuses = {"needs_credentials", "needs_export", "needs_folder"}
    warning_statuses = {"failed", "skipped"}
    has_usable = any(status.get("status") in usable_statuses for status in statuses)
    has_needs_input = any(status.get("status") in needs_input_statuses for status in statuses)
    has_warning = any(status.get("status") in warning_statuses for status in statuses)
    if all(status.get("status") == "skipped" for status in statuses):
        return "skipped"
    if has_usable and not has_needs_input and not has_warning:
        return "ok"
    if has_usable:
        return "partial"
    if has_needs_input:
        return "needs_input"
    if has_warning:
        return "failed"
    return "skipped"


def _workflow_record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    categories = [_clean_text(record.get("category")) for record in records]
    return {
        "source_records": len(records),
        "variant_evidence": sum(
            category
            not in {
                "",
                "source_metadata",
                "literature",
                "population",
                "population_frequency",
                "population_association",
                "polygenic_score",
            }
            for category in categories
        ),
        "population_records": sum(
            category in {"population", "population_frequency", "population_association", "polygenic_score"}
            for category in categories
        ),
        "literature_records": sum(category == "literature" for category in categories),
        "metadata_records": sum(category in {"", "source_metadata"} for category in categories),
    }


def _workflow_summary(
    *,
    workflow: WorkflowSpec,
    status: str,
    source_count: int,
    provider_statuses: list[dict[str, Any]],
    record_counts: dict[str, int],
) -> str:
    if source_count == 0:
        return f"{workflow.label} had no selected sources after manual source filtering."
    needs_input = sum(
        status_row.get("status") in {"needs_credentials", "needs_export", "needs_folder"}
        for status_row in provider_statuses
    )
    failures = sum(status_row.get("status") == "failed" for status_row in provider_statuses)
    usable = sum(status_row.get("status") in {"ok", "imported", "metadata_only"} for status_row in provider_statuses)
    parts = [
        f"{workflow.label} queried {source_count} selected source(s)",
        f"with {usable} usable or metadata response(s)",
        f"and {record_counts['source_records']} normalized record(s).",
    ]
    if needs_input:
        parts.append(f"{needs_input} source(s) need credentials or permitted exports.")
    if failures:
        parts.append(f"{failures} source(s) failed without stopping other workflows.")
    if status == "ok":
        parts.append("No provider failures were reported for this workflow.")
    return " ".join(parts)


def _build_workflow_runs(
    *,
    workflows: tuple[WorkflowSpec, ...],
    selected_source_keys: list[str],
    provider_statuses: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    timestamp: str,
) -> list[dict[str, Any]]:
    """Build deterministic per-workflow status and evidence summaries."""
    status_lookup = {str(status.get("source_key")): status for status in provider_statuses}
    selected = set(selected_source_keys)
    runs: list[dict[str, Any]] = []
    for index, workflow in enumerate(workflows, start=1):
        source_keys = [source_key for source_key in workflow.valid_source_keys() if source_key in selected]
        workflow_statuses = [status_lookup[source_key] for source_key in source_keys if source_key in status_lookup]
        workflow_records = [
            record for record in source_records if str(record.get("source_key", "")).strip() in set(source_keys)
        ]
        record_counts = _workflow_record_counts(workflow_records)
        status = _workflow_status(workflow_statuses)
        warnings: list[str] = []
        errors: list[str] = []
        for provider_status in workflow_statuses:
            source_key = str(provider_status.get("source_key", ""))
            source_name = str(provider_status.get("name") or source_key)
            for warning in provider_status.get("warnings", []) or []:
                warnings.append(f"{source_name}: {_clean_text(warning)}")
            for error in provider_status.get("errors", []) or []:
                errors.append(f"{source_name}: {_clean_text(error)}")
            if provider_status.get("status") in {"needs_credentials", "needs_export"}:
                warnings.append(f"{source_name}: {provider_status.get('message', provider_status.get('status'))}")
        evidence_ids = sorted(
            {
                _clean_text(record.get("evidence_id"))
                for record in workflow_records
                if _clean_text(record.get("evidence_id"))
            }
        )
        runs.append(
            {
                "run_order": index,
                "workflow_key": workflow.key,
                "label": workflow.label,
                "purpose": workflow.purpose,
                "status": status,
                "report_section": workflow.report_section,
                "requires_vcf": workflow.requires_vcf,
                "requires_manifest": workflow.requires_manifest,
                "evidence_lanes": list(workflow.evidence_lanes),
                "selected_source_keys": source_keys,
                "source_count": len(source_keys),
                "provider_statuses": workflow_statuses,
                "record_counts": record_counts,
                "warnings": warnings,
                "errors": errors,
                "summary": _workflow_summary(
                    workflow=workflow,
                    status=status,
                    source_count=len(source_keys),
                    provider_statuses=workflow_statuses,
                    record_counts=record_counts,
                ),
                "evidence_ids": evidence_ids,
                "started_at": timestamp,
                "completed_at": timestamp,
                "licensed_notes": list(workflow.licensed_notes),
            }
        )
    return runs


def build_dynamic_knowledge_base(
    *,
    gene: str,
    region: str,
    genome_build: str,
    variants: pd.DataFrame | None = None,
    manifest_subset: pd.DataFrame | None = None,
    selected_workflows: list[str] | tuple[str, ...] | None = None,
    selected_sources: list[str] | tuple[str, ...] | None = None,
    credentials: dict[str, str] | None = None,
    source_imports: dict[str, str | Path] | None = None,
    use_local_article_evidence: bool = False,
    article_pdf_folder: str | Path | None = None,
    article_pdf_recursive: bool = True,
    max_article_pdfs: int = 100,
    gene_aliases: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    request_client: RequestClient | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a dynamic variant knowledge-base payload and optionally write it."""
    normalized_gene = _clean_text(gene).upper()
    normalized_build = _clean_text(genome_build) or "hg19"
    workflow_specs = select_workflow_specs(selected_workflows)
    local_article_requested = _local_article_requested(
        workflows=workflow_specs,
        selected_sources=selected_sources,
        use_local_article_evidence=use_local_article_evidence,
        article_pdf_folder=article_pdf_folder,
    )
    if local_article_requested:
        workflow_specs = _append_local_article_workflow(workflow_specs)
    specs = _selected_source_specs(selected_sources=selected_sources, workflows=workflow_specs)
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
    for spec in _execution_specs_for_workflows(specs=specs, workflows=workflow_specs):
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

    runtime_specs = specs
    local_article_evidence: dict[str, Any] = {}
    if local_article_requested:
        local_article_evidence = extract_local_article_evidence(
            gene=normalized_gene,
            pdf_folder=article_pdf_folder,
            gene_aliases=gene_aliases,
            recursive=article_pdf_recursive,
            max_pdfs=max_article_pdfs,
            generated_at=timestamp,
        )
        source_results.append(source_result_from_local_article_evidence(local_article_evidence))
        runtime_specs = specs + (LOCAL_ARTICLE_SOURCE_SPEC,)

    source_records = _source_result_to_records(source_results)
    spec_lookup = {spec.key: spec for spec in runtime_specs}
    provider_statuses = [
        result.to_status(spec_lookup[result.source_key])
        for result in source_results
        if result.source_key in spec_lookup
    ]
    selected_source_keys = [spec.key for spec in runtime_specs]
    workflow_source_matrix = _workflow_source_matrix(
        workflows=workflow_specs,
        selected_source_keys=selected_source_keys,
    )
    workflow_runs = _build_workflow_runs(
        workflows=workflow_specs,
        selected_source_keys=selected_source_keys,
        provider_statuses=provider_statuses,
        source_records=source_records,
        timestamp=timestamp,
    )
    payload = {
        "schema_version": "1.0",
        "database_name": f"NophiGene Dynamic {normalized_gene} Variant Knowledge Base",
        "gene_name": normalized_gene,
        "region": region,
        "region_parsed": _parse_region(region),
        "genome_build": normalized_build,
        "generated_at": timestamp,
        "variant_records": _build_dynamic_variant_records(query, source_records, runtime_specs),
        "epigenetic_locus_records": _build_epigenetic_locus_records(query),
        "population_records": _filter_source_records(
            source_records,
            {"population", "population_frequency", "population_association", "polygenic_score"},
        ),
        "literature_records": _filter_source_records(source_records, {"literature"}),
        "source_records": source_records,
        "provider_statuses": provider_statuses,
        "workflow_runs": workflow_runs,
        "workflow_source_matrix": workflow_source_matrix,
        "provenance": {
            "selected_workflow_keys": [workflow.key for workflow in workflow_specs],
            "selected_source_keys": selected_source_keys,
            "credential_statuses": credential_status_for_specs(list_source_specs(), credentials),
            "variant_count": len(query.variants),
            "epigenetic_locus_count": len(query.epigenetic_loci),
            "cache_dir": str(cache_dir) if cache_dir else "",
            "source_imports": [
                bundle.to_provenance()
                for bundle in import_bundles.values()
            ],
            "source_import_errors": dict(sorted(import_errors.items())),
            "local_article_evidence": local_article_evidence.get("provenance", {}) if local_article_evidence else {},
        },
        "source_lanes": LANE_LABELS,
        "license_notes": {
            spec.key: {
                "source": spec.name,
                "access_type": spec.access_type,
                "license_note": spec.license_note,
                "homepage": spec.homepage,
            }
            for spec in runtime_specs
        },
    }
    if local_article_evidence:
        payload["local_article_evidence"] = local_article_evidence
    if output_dir is not None:
        output_path = Path(output_dir) / "variant_kb.json"
        if local_article_evidence:
            payload["local_article_evidence_artifacts"] = write_local_article_artifacts(output_dir, local_article_evidence)
        payload["artifact_path"] = str(output_path)
        _write_json(output_path, payload)
    return payload
