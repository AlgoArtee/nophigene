"""Typed records used by the dynamic variant knowledge-base builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO timestamp with stable UTC notation."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SourceSpec:
    """A queryable or linkout-only database described by ``resources.txt``."""

    key: str
    name: str
    description: str
    lane: str
    access_type: str
    connector_kind: str
    homepage: str = ""
    env_var: str = ""
    license_note: str = ""
    rate_limit_per_second: float | None = None
    supports_variant: bool = True
    supports_gene: bool = True
    supports_region: bool = True
    supports_literature: bool = False
    ingestion_modes: tuple[str, ...] = ("linkout_only",)
    requires_export: bool = False
    accepted_import_formats: tuple[str, ...] = ("csv", "json")
    import_schema: tuple[str, ...] = (
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

    def to_card(
        self,
        *,
        credential_status: str = "not_required",
        selected: bool = True,
        import_status: str = "",
        import_path: str = "",
    ) -> dict[str, Any]:
        """Return a redacted UI/API representation."""
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "lane": self.lane,
            "access_type": self.access_type,
            "connector_kind": self.connector_kind,
            "homepage": self.homepage,
            "env_var": self.env_var,
            "license_note": self.license_note,
            "rate_limit_per_second": self.rate_limit_per_second,
            "ingestion_modes": list(self.ingestion_modes),
            "requires_export": self.requires_export,
            "accepted_import_formats": list(self.accepted_import_formats),
            "import_schema": list(self.import_schema),
            "supports": {
                "variant": self.supports_variant,
                "gene": self.supports_gene,
                "region": self.supports_region,
                "literature": self.supports_literature,
            },
            "credential_status": credential_status,
            "import_status": import_status,
            "import_path": import_path,
            "selected": selected,
        }


@dataclass(frozen=True)
class QueryVariant:
    """A normalized sample variant row used for external source lookups."""

    chrom: str
    pos: int
    ref: str = ""
    alt: str = ""
    rsid: str = ""
    sample: str = ""
    gt_raw: str = ""
    zygosity: str = ""

    @property
    def coordinate_key(self) -> str:
        return f"{self.chrom.removeprefix('chr')}:{self.pos}"

    @property
    def change_key(self) -> str:
        if self.ref and self.alt:
            return f"{self.coordinate_key}:{self.ref}>{self.alt}"
        return self.coordinate_key

    @property
    def label(self) -> str:
        return self.rsid or self.change_key

    def lookup_terms(self, gene: str) -> list[str]:
        terms = [self.coordinate_key, self.change_key]
        if self.rsid:
            terms.extend([self.rsid, f"{gene}:{self.rsid}"])
        return terms


@dataclass(frozen=True)
class EpigeneticLocus:
    """A methylation/regulatory locus extracted from a filtered manifest."""

    probe_id: str
    chrom: str
    pos: int
    gene: str = ""
    relation: str = ""


@dataclass(frozen=True)
class KnowledgeQuery:
    """The normalized context sent to all database connectors."""

    gene: str
    region: str
    genome_build: str
    variants: tuple[QueryVariant, ...] = ()
    epigenetic_loci: tuple[EpigeneticLocus, ...] = ()

    @property
    def rsids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for variant in self.variants:
            rsid = variant.rsid.strip()
            if rsid and rsid.lower().startswith("rs") and rsid.lower() not in seen:
                seen.add(rsid.lower())
                ordered.append(rsid)
        return ordered


@dataclass
class SourceResult:
    """Normalized connector result with redacted status metadata."""

    source_key: str
    status: str
    message: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    queried_urls: list[str] = field(default_factory=list)
    elapsed_ms: int | None = None

    def to_status(self, spec: SourceSpec) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "name": spec.name,
            "lane": spec.lane,
            "access_type": spec.access_type,
            "status": self.status,
            "message": self.message,
            "record_count": len(self.records),
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "queried_urls": list(self.queried_urls),
            "elapsed_ms": self.elapsed_ms,
            "license_note": spec.license_note,
            "homepage": spec.homepage,
        }
