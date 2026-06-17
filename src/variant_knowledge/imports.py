"""User-export ingestion for licensed dynamic knowledge sources.

This module intentionally parses only a small, documented set of summary fields.
Unknown columns from licensed exports are discarded so dynamic KB artifacts do
not become raw database dumps.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import KnowledgeQuery, SourceSpec, utc_now_iso

CANONICAL_IMPORT_FIELDS = (
    "source_key",
    "record_id",
    "gene",
    "variant",
    "rsid",
    "title",
    "summary",
    "assertion",
    "phenotype",
    "drug",
    "score",
    "url",
    "citation",
    "evidence_level",
    "license_note",
)

ACCEPTED_IMPORT_FORMATS = ("csv", "json")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "source_key": ("source", "database", "provider", "source_key"),
    "record_id": (
        "record_id",
        "id",
        "identifier",
        "accession",
        "variant_id",
        "hgmd_id",
        "article_id",
    ),
    "gene": ("gene", "gene_symbol", "symbol"),
    "variant": (
        "variant",
        "variant_name",
        "mutation",
        "alteration",
        "hgvs",
        "dna_change",
        "protein_change",
    ),
    "rsid": ("rsid", "rs_id", "dbsnp", "dbsnp_id"),
    "title": ("title", "article_title", "name", "gene_card"),
    "summary": ("summary", "description", "interpretation", "notes", "abstract", "evidence"),
    "assertion": ("assertion", "clinical_significance", "classification", "pathogenicity"),
    "phenotype": ("phenotype", "disease", "condition", "trait"),
    "drug": ("drug", "medication", "therapy"),
    "score": ("score", "rank", "rating"),
    "url": ("url", "link", "source_url"),
    "citation": ("citation", "reference", "pmid", "doi"),
    "evidence_level": ("evidence_level", "level", "tier"),
    "license_note": ("license_note", "license", "terms"),
}

SOURCE_SPECIFIC_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "hgmd": {
        "record_id": ("acc_num", "hgmd_accession"),
        "assertion": ("class", "tag"),
        "phenotype": ("disease",),
        "variant": ("mutation", "hgvs"),
    },
    "genecards": {
        "record_id": ("gc_id", "card_id"),
        "summary": ("summaries", "gene_summary"),
        "phenotype": ("disorders", "related_diseases"),
    },
    "varsome": {
        "record_id": ("varsome_id", "variant_id"),
        "assertion": ("acmg_classification", "classification"),
        "score": ("pathogenicity_score",),
    },
    "franklin": {
        "record_id": ("franklin_id", "variant_id"),
        "assertion": ("classification", "acmg_classification"),
        "evidence_level": ("acmg_evidence",),
    },
    "mastermind": {
        "record_id": ("article_id", "publication_id"),
        "citation": ("pmid", "doi", "reference"),
        "summary": ("abstract", "snippet", "evidence"),
    },
    "google_scholar": {
        "record_id": ("result_id", "article_id"),
        "citation": ("citation", "cited_by", "doi"),
        "summary": ("snippet", "abstract"),
    },
}


@dataclass(frozen=True)
class SourceImportBundle:
    """Normalized evidence imported from a user-provided export."""

    source_key: str
    source_name: str
    filename: str
    filename_sha256: str
    file_sha256: str
    row_count: int
    normalized_record_count: int
    records: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    license_note: str
    imported_at: str

    def to_provenance(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "source_name": self.source_name,
            "filename": self.filename,
            "filename_sha256": self.filename_sha256,
            "file_sha256": self.file_sha256,
            "row_count": self.row_count,
            "normalized_record_count": self.normalized_record_count,
            "license_note": self.license_note,
            "imported_at": self.imported_at,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def import_schema() -> dict[str, Any]:
    """Return the public schema advertised by cards and OpenAPI."""
    return {
        "formats": list(ACCEPTED_IMPORT_FORMATS),
        "fields": list(CANONICAL_IMPORT_FIELDS),
        "raw_payload_policy": "Unknown columns are discarded; dynamic KB artifacts store normalized summaries only.",
    }


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalized_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in ACCEPTED_IMPORT_FORMATS:
        raise ValueError(f"Unsupported source import format '.{suffix}'. Use CSV or JSON.")
    warnings: list[str] = []
    if suffix == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Source import CSV has no header row: {path.name}")
            return [dict(row) for row in reader], warnings

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("records", "items", "results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Source import JSON must be a list or an object with records/items/results/data.")
    rows = [item for item in payload if isinstance(item, dict)]
    if len(rows) != len(payload):
        warnings.append("Ignored non-object JSON entries in the source import.")
    return rows, warnings


def _aliases_for(source_key: str) -> dict[str, tuple[str, ...]]:
    aliases = {field: values for field, values in FIELD_ALIASES.items()}
    for field, values in SOURCE_SPECIFIC_ALIASES.get(source_key, {}).items():
        aliases[field] = tuple(dict.fromkeys(values + aliases.get(field, ())))
    return aliases


def _normalize_row(
    row: dict[str, Any],
    *,
    source_key: str,
    source_name: str,
    source_lane: str,
    license_note: str,
) -> dict[str, Any] | None:
    normalized_lookup = {_normalized_key(key): value for key, value in row.items()}
    aliases = _aliases_for(source_key)
    record: dict[str, Any] = {}
    for field in CANONICAL_IMPORT_FIELDS:
        value = ""
        for alias in aliases.get(field, (field,)):
            normalized_alias = _normalized_key(alias)
            if normalized_alias in normalized_lookup:
                value = _clean_text(normalized_lookup[normalized_alias])
                if value:
                    break
        if value:
            record[field] = value
    record["source_key"] = source_key
    record["source"] = source_name
    record["license_note"] = record.get("license_note") or license_note
    record["category"] = _category_for_record(source_lane, record)
    if record.get("assertion"):
        record["clinical_significance"] = record["assertion"]
    record["label"] = (
        record.get("title")
        or record.get("variant")
        or record.get("rsid")
        or record.get("record_id")
        or source_name
    )[:220]
    summary_parts = [
        record.get("summary", ""),
        record.get("assertion", ""),
        record.get("phenotype", ""),
        record.get("drug", ""),
        record.get("evidence_level", ""),
    ]
    record["summary"] = _clean_text(" | ".join(part for part in summary_parts if part))[:600]
    if not any(record.get(field) for field in CANONICAL_IMPORT_FIELDS if field not in {"source_key", "license_note"}):
        return None
    return record


def _category_for_record(source_lane: str, record: dict[str, Any]) -> str:
    if source_lane == "literature":
        return "literature"
    if record.get("citation") and record.get("title") and not any(
        record.get(field) for field in ("assertion", "phenotype", "drug")
    ):
        return "literature"
    if source_lane == "pharmacogenomics" or record.get("drug"):
        return "pharmacogenomics"
    if source_lane == "population":
        return "population"
    return "clinical_variant"


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        key = tuple(
            _clean_text(record.get(field)).casefold()
            for field in ("source_key", "record_id", "gene", "variant", "rsid", "title", "url", "citation")
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def parse_source_import(
    source_key: str,
    path: str | Path,
    *,
    spec: SourceSpec | None = None,
    imported_at: str | None = None,
) -> SourceImportBundle:
    """Parse one licensed-source export into normalized summary records."""
    normalized_key = _clean_text(source_key)
    if not normalized_key:
        raise ValueError("Source import requires a source key.")
    import_path = Path(path)
    if not import_path.exists():
        raise FileNotFoundError(f"Source import file not found: {import_path}")

    data = import_path.read_bytes()
    rows, warnings = _load_rows(import_path)
    source_name = spec.name if spec else normalized_key
    source_lane = spec.lane if spec else "licensed"
    license_note = spec.license_note if spec else ""
    normalized_records = [
        record
        for row in rows
        if (
            record := _normalize_row(
                row,
                source_key=normalized_key,
                source_name=source_name,
                source_lane=source_lane,
                license_note=license_note,
            )
        )
    ]
    normalized_records = _dedupe_records(normalized_records)
    if rows and not normalized_records:
        warnings.append("No import rows contained supported normalized evidence fields.")
    return SourceImportBundle(
        source_key=normalized_key,
        source_name=source_name,
        filename=import_path.name,
        filename_sha256=_hash_bytes(import_path.name.encode("utf-8", errors="ignore")),
        file_sha256=_hash_bytes(data),
        row_count=len(rows),
        normalized_record_count=len(normalized_records),
        records=normalized_records,
        warnings=warnings,
        errors=[],
        license_note=license_note,
        imported_at=imported_at or utc_now_iso(),
    )


def filter_import_records_for_query(
    records: list[dict[str, Any]],
    query: KnowledgeQuery,
) -> list[dict[str, Any]]:
    """Keep imported evidence that matches the active gene or observed variants."""
    gene = query.gene.casefold()
    variant_terms: set[str] = set()
    for variant in query.variants:
        variant_terms.update(term.casefold() for term in variant.lookup_terms(query.gene))
        variant_terms.add(variant.label.casefold())
    filtered: list[dict[str, Any]] = []
    for record in records:
        record_gene = _clean_text(record.get("gene")).casefold()
        if record_gene and record_gene != gene:
            continue
        record_variant_terms = {
            _clean_text(record.get("variant")).casefold(),
            _clean_text(record.get("rsid")).casefold(),
        }
        record_variant_terms = {term for term in record_variant_terms if term}
        if variant_terms and record_variant_terms and not (record_variant_terms & variant_terms):
            continue
        filtered.append(dict(record))
    return filtered
