"""Database connectors used by dynamic variant knowledge-base generation."""

from __future__ import annotations

import csv
import io
import os
import time
from typing import Any

from .client import KnowledgeRequestError, RequestClient
from .credentials import ResolvedCredential
from .models import KnowledgeQuery, SourceResult, SourceSpec


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _first_variant(query: KnowledgeQuery):
    return query.variants[0] if query.variants else None


def _first_rsid(query: KnowledgeQuery) -> str:
    return query.rsids[0] if query.rsids else ""


def _literature_query(query: KnowledgeQuery) -> str:
    rsid = _first_rsid(query)
    if rsid:
        return f"{query.gene} {rsid}"
    return query.gene


NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL_NAME = "NophiGeneDynamicKB"
NLM_CLINICAL_TABLES_VARIANTS_URL = "https://clinicaltables.nlm.nih.gov/api/variants/v4/search"
CLINGEN_VALIDITY_URL = "https://search.clinicalgenome.org/kb/gene-validity/download"
CLINGEN_DOSAGE_URL = "https://search.clinicalgenome.org/kb/gene-dosage/download"
CLINGEN_SUMMARY_URL = "https://search.clinicalgenome.org/kb/reports/curation-activity-summary-report"
CLINGEN_ACTIONABILITY_URLS = (
    ("Adult", "https://actionability.clinicalgenome.org/ac/Adult/api/summ?flavor=flat"),
    ("Pediatric", "https://actionability.clinicalgenome.org/ac/Pediatric/api/summ?flavor=flat"),
)
CLINGEN_PRIMARY_TIMEOUT_SECONDS = 20
CLINGEN_OPTIONAL_TIMEOUT_SECONDS = 8


def _ncbi_request_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    tool = os.environ.get("NOPHIGENE_NCBI_TOOL", NCBI_TOOL_NAME).strip() or NCBI_TOOL_NAME
    out["tool"] = tool.replace(" ", "_")
    email = os.environ.get("NOPHIGENE_NCBI_EMAIL", "").strip()
    if email:
        out["email"] = email
    api_key = os.environ.get("NOPHIGENE_NCBI_API_KEY", "").strip() or os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        out["api_key"] = api_key
    return out


def _request_failure_result(
    spec: SourceSpec,
    exc: KnowledgeRequestError,
    *,
    queried_urls: list[str],
    started: float,
) -> SourceResult:
    remediation = getattr(exc, "remediation", "")
    code = getattr(exc, "code", "request_failed")
    message = str(exc)
    return SourceResult(
        source_key=spec.key,
        status="failed",
        message=message,
        errors=[message],
        queried_urls=queried_urls,
        elapsed_ms=_elapsed_ms(started),
        error_code=code,
        remediation=remediation,
    )


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(_clean_cell(item) for item in value if _clean_cell(item))
    if isinstance(value, dict):
        for key in ("description", "name", "label", "title", "trait_name", "variation_name"):
            cleaned = _clean_cell(value.get(key))
            if cleaned:
                return cleaned
        return ""
    return str(value).strip()


class BaseConnector:
    """Base connector contract."""

    def __init__(self, spec: SourceSpec, client: RequestClient, credential: ResolvedCredential) -> None:
        self.spec = spec
        self.client = client
        self.credential = credential

    def query(self, context: KnowledgeQuery) -> SourceResult:
        return SourceResult(
            source_key=self.spec.key,
            status="metadata_only",
            message=self.spec.license_note,
            records=[self._metadata_record()],
        )

    def _metadata_record(self) -> dict[str, Any]:
        return {
            "category": "source_metadata",
            "source": self.spec.name,
            "label": self.spec.name,
            "summary": self.spec.description or self.spec.license_note,
            "url": self.spec.homepage,
            "license_note": self.spec.license_note,
        }


class AuthMetadataConnector(BaseConnector):
    """Credential-aware connector for sources whose official API is not implemented in v1."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        if not self.credential.present:
            return SourceResult(
                source_key=self.spec.key,
                status="needs_credentials",
                message=f"Set {self.spec.env_var} or enter a session token before querying {self.spec.name}.",
                records=[self._metadata_record()],
            )
        return SourceResult(
            source_key=self.spec.key,
            status="metadata_only",
            message=(
                f"Credentials are available for {self.spec.name}, but this v1 connector only records "
                "metadata until the licensed endpoint contract is configured."
            ),
            records=[self._metadata_record()],
        )


class LicensedMetadataConnector(BaseConnector):
    """Explicitly non-scraping connector for licensed databases/search portals."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        return SourceResult(
            source_key=self.spec.key,
            status="metadata_only",
            message="Licensed or non-open source: no scraping is performed.",
            records=[self._metadata_record()],
        )


class NcbiEutilsConnector(BaseConnector):
    """NCBI E-Utilities connector for ClinVar, dbSNP, Gene, PubMed, PMC, GEO, and LitVar-like evidence."""

    DB_BY_KIND = {
        "clinvar": "clinvar",
        "dbsnp": "snp",
        "ncbi_gene": "gene",
        "pubmed": "pubmed",
        "pmc": "pmc",
        "geo": "gds",
        "litvar": "pubmed",
    }

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        db = self.DB_BY_KIND.get(self.spec.connector_kind, "pubmed")
        term = self._term(context)
        records: list[dict[str, Any]] = []
        urls: list[str] = []
        try:
            search_url = f"{NCBI_EUTILS_BASE_URL}/esearch.fcgi"
            search = self.client.get_json(
                search_url,
                params=_ncbi_request_params({"db": db, "term": term, "retmode": "json", "retmax": 5}),
                rate_limit_per_second=self.spec.rate_limit_per_second,
            )
            urls.append(search_url)
            ids = list(search.get("esearchresult", {}).get("idlist", []))
            if ids:
                summary_url = f"{NCBI_EUTILS_BASE_URL}/esummary.fcgi"
                summary = self.client.get_json(
                    summary_url,
                    params=_ncbi_request_params({"db": db, "id": ",".join(ids[:5]), "retmode": "json"}),
                    rate_limit_per_second=self.spec.rate_limit_per_second,
                )
                urls.append(summary_url)
                result = summary.get("result", {})
                for item_id in ids[:5]:
                    item = result.get(str(item_id), {})
                    title = str(item.get("title") or item.get("name") or item.get("uid") or item_id)
                    records.append(
                        {
                            "category": self._category(),
                            "source": self.spec.name,
                            "label": title,
                            "summary": str(item.get("description") or item.get("fulljournalname") or title),
                            "source_id": str(item_id),
                            "url": self._record_url(db, str(item_id)),
                            "variant": _first_rsid(context),
                        }
                    )
            return SourceResult(
                source_key=self.spec.key,
                status="ok",
                message=f"Queried NCBI {db}; {len(records)} record(s) returned.",
                records=records,
                queried_urls=urls,
                elapsed_ms=_elapsed_ms(started),
            )
        except KnowledgeRequestError as exc:
            return _request_failure_result(self.spec, exc, queried_urls=urls, started=started)

    def _term(self, context: KnowledgeQuery) -> str:
        rsid = _first_rsid(context)
        if self.spec.connector_kind in {"clinvar", "dbsnp"} and rsid:
            return rsid
        if self.spec.connector_kind == "ncbi_gene":
            return f"{context.gene}[sym] AND Homo sapiens[orgn]"
        if self.spec.connector_kind == "litvar":
            return f'("{context.gene}"[Title/Abstract]) AND ({rsid or context.region})'
        return _literature_query(context)

    def _category(self) -> str:
        if self.spec.connector_kind in {"pubmed", "pmc", "litvar", "geo"}:
            return "literature"
        if self.spec.connector_kind == "dbsnp":
            return "population"
        return "clinical_variant"

    def _record_url(self, db: str, item_id: str) -> str:
        if db == "pubmed":
            return f"https://pubmed.ncbi.nlm.nih.gov/{item_id}/"
        if db == "pmc":
            return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{item_id}/"
        if db == "snp":
            return f"https://www.ncbi.nlm.nih.gov/snp/{item_id}"
        if db == "gene":
            return f"https://www.ncbi.nlm.nih.gov/gene/{item_id}"
        if db == "clinvar":
            return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{item_id}/"
        return f"https://www.ncbi.nlm.nih.gov/{db}/{item_id}"


class ClinVarConnector(BaseConnector):
    """ClinVar connector using official NCBI query patterns."""

    RETMAX = 10

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        queried_urls: list[str] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        records: list[dict[str, Any]] = []
        try:
            for rsid in context.rsids:
                ids = self._esearch_ids(rsid, queried_urls)
                records.extend(self._esummary_records(ids, context, seen_ids, queried_urls))

            gene_ids = self._esearch_ids(f"{context.gene}[gene] AND single_gene[prop]", queried_urls)
            if not gene_ids:
                gene_ids = self._esearch_ids(f"{context.gene}[gene]", queried_urls)
            records.extend(self._esummary_records(gene_ids, context, seen_ids, queried_urls))

            if not records:
                try:
                    records.extend(self._clinical_tables_records(context, seen_ids, queried_urls))
                except KnowledgeRequestError as exc:
                    warnings.append(str(exc))

            return SourceResult(
                source_key=self.spec.key,
                status="ok",
                message=f"Queried ClinVar; {len(records)} clinical variant record(s) returned.",
                records=records,
                warnings=warnings,
                queried_urls=queried_urls,
                elapsed_ms=_elapsed_ms(started),
            )
        except KnowledgeRequestError as exc:
            return _request_failure_result(self.spec, exc, queried_urls=queried_urls, started=started)

    def _esearch_ids(self, term: str, queried_urls: list[str]) -> list[str]:
        url = f"{NCBI_EUTILS_BASE_URL}/esearch.fcgi"
        payload = self.client.get_json(
            url,
            params=_ncbi_request_params(
                {
                    "db": "clinvar",
                    "term": term,
                    "retmode": "json",
                    "retmax": self.RETMAX,
                }
            ),
            rate_limit_per_second=self.spec.rate_limit_per_second,
        )
        queried_urls.append(url)
        idlist = payload.get("esearchresult", {}).get("idlist", [])
        return [str(item_id) for item_id in idlist[: self.RETMAX]]

    def _esummary_records(
        self,
        ids: list[str],
        context: KnowledgeQuery,
        seen_ids: set[str],
        queried_urls: list[str],
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        url = f"{NCBI_EUTILS_BASE_URL}/esummary.fcgi"
        payload = self.client.get_json(
            url,
            params=_ncbi_request_params(
                {
                    "db": "clinvar",
                    "id": ",".join(ids[: self.RETMAX]),
                    "retmode": "json",
                }
            ),
            rate_limit_per_second=self.spec.rate_limit_per_second,
        )
        queried_urls.append(url)
        result = payload.get("result", {})
        records: list[dict[str, Any]] = []
        for item_id in ids[: self.RETMAX]:
            item = result.get(str(item_id), {})
            if not isinstance(item, dict):
                continue
            source_id = self._variation_id(item, item_id)
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            records.append(self._record_from_summary(item, source_id, context))
        return records

    def _clinical_tables_records(
        self,
        context: KnowledgeQuery,
        seen_ids: set[str],
        queried_urls: list[str],
    ) -> list[dict[str, Any]]:
        terms = context.rsids[0] if context.rsids else context.gene
        payload = self.client.get_json(
            NLM_CLINICAL_TABLES_VARIANTS_URL,
            params={
                "terms": terms,
                "maxList": 5,
                "df": "VariationID,GeneSymbol,dbSNP,Name",
            },
            rate_limit_per_second=self.spec.rate_limit_per_second,
        )
        queried_urls.append(NLM_CLINICAL_TABLES_VARIANTS_URL)
        if not isinstance(payload, list) or len(payload) < 4 or not isinstance(payload[3], list):
            return []
        records: list[dict[str, Any]] = []
        for row in payload[3][:5]:
            if not isinstance(row, list):
                continue
            cells = [_clean_cell(value) for value in row]
            source_id = cells[0] if cells else ""
            if not source_id or source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            gene_symbol = cells[1] if len(cells) > 1 else context.gene
            rsid = cells[2] if len(cells) > 2 else ""
            label = cells[3] if len(cells) > 3 else f"ClinVar variation {source_id}"
            records.append(
                {
                    "category": "clinical_variant",
                    "source": self.spec.name,
                    "label": label or f"ClinVar variation {source_id}",
                    "summary": f"ClinVar Clinical Tables record for {gene_symbol or context.gene}.",
                    "source_id": source_id,
                    "url": self._record_url(source_id),
                    "variant": rsid or source_id,
                    "rsid": rsid,
                    "gene": gene_symbol or context.gene,
                }
            )
        return records

    def _record_from_summary(
        self,
        item: dict[str, Any],
        source_id: str,
        context: KnowledgeQuery,
    ) -> dict[str, Any]:
        title = (
            _clean_cell(item.get("title"))
            or self._first_variation_name(item)
            or _clean_cell(item.get("accession"))
            or f"ClinVar variation {source_id}"
        )
        significance = self._clinical_significance(item)
        traits = self._traits(item)
        rsid = self._rsid(item) or _first_rsid(context)
        summary_parts = []
        if significance:
            summary_parts.append(f"Clinical significance: {significance}")
        if traits:
            summary_parts.append(f"Phenotype/trait: {'; '.join(traits[:3])}")
        variant_type = _clean_cell(item.get("variant_type") or item.get("obj_type"))
        if variant_type:
            summary_parts.append(f"Variant type: {variant_type}")
        summary = "; ".join(summary_parts) or _clean_cell(item.get("description")) or title
        record = {
            "category": "clinical_variant",
            "source": self.spec.name,
            "label": title,
            "summary": summary,
            "source_id": source_id,
            "url": self._record_url(source_id),
            "variant": rsid or source_id,
            "rsid": rsid,
            "gene": context.gene,
        }
        if significance:
            record["clinical_significance"] = significance
            record["assertion"] = significance
        if traits:
            record["phenotype"] = "; ".join(traits[:5])
        return record

    def _variation_id(self, item: dict[str, Any], fallback: str) -> str:
        for key in ("variation_id", "uid", "id"):
            value = _clean_cell(item.get(key))
            if value:
                return value
        return str(fallback)

    def _first_variation_name(self, item: dict[str, Any]) -> str:
        variation_set = item.get("variation_set")
        if not isinstance(variation_set, list):
            return ""
        for variation in variation_set:
            if isinstance(variation, dict):
                name = _clean_cell(variation.get("variation_name"))
                if name:
                    return name
        return ""

    def _clinical_significance(self, item: dict[str, Any]) -> str:
        for key in (
            "clinical_significance",
            "germline_classification",
            "somatic_clinical_impact",
            "oncogenicity_classification",
        ):
            value = item.get(key)
            if isinstance(value, dict):
                cleaned = _clean_cell(value.get("description") or value.get("review_status"))
            else:
                cleaned = _clean_cell(value)
            if cleaned:
                return cleaned
        return ""

    def _traits(self, item: dict[str, Any]) -> list[str]:
        traits: list[str] = []
        for key in ("trait_set", "traits"):
            value = item.get(key)
            if isinstance(value, list):
                for trait in value:
                    cleaned = _clean_cell(trait)
                    if cleaned:
                        traits.append(cleaned)
            else:
                cleaned = _clean_cell(value)
                if cleaned:
                    traits.append(cleaned)
        seen: set[str] = set()
        ordered: list[str] = []
        for trait in traits:
            normalized = trait.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(trait)
        return ordered

    def _rsid(self, item: dict[str, Any]) -> str:
        for key in ("rsid", "dbsnp", "refsnp_id"):
            value = _clean_cell(item.get(key))
            if value:
                return value if value.lower().startswith("rs") else f"rs{value}"
        variation_set = item.get("variation_set")
        if not isinstance(variation_set, list):
            return ""
        for variation in variation_set:
            if not isinstance(variation, dict):
                continue
            xrefs = variation.get("variation_xrefs")
            if not isinstance(xrefs, list):
                continue
            for xref in xrefs:
                if not isinstance(xref, dict):
                    continue
                source = _clean_cell(xref.get("db_source") or xref.get("db"))
                if source.lower() != "dbsnp":
                    continue
                value = _clean_cell(xref.get("db_id") or xref.get("id"))
                if value:
                    return value if value.lower().startswith("rs") else f"rs{value}"
        return ""

    def _record_url(self, source_id: str) -> str:
        return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{source_id}/"


class EnsemblConnector(BaseConnector):
    """Ensembl REST connector for gene and regional variant context."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        server = "https://rest.ensembl.org" if context.genome_build == "hg38" else "https://grch37.rest.ensembl.org"
        headers = {"Content-Type": "application/json"}
        urls: list[str] = []
        records: list[dict[str, Any]] = []
        try:
            lookup_url = f"{server}/lookup/symbol/homo_sapiens/{context.gene}"
            gene_payload = self.client.get_json(lookup_url, params={"expand": 1}, headers=headers)
            urls.append(lookup_url)
            records.append(
                {
                    "category": "gene_annotation",
                    "source": self.spec.name,
                    "label": f"Ensembl {context.gene}",
                    "summary": f"{context.gene} resolved to {gene_payload.get('seq_region_name')}:{gene_payload.get('start')}-{gene_payload.get('end')}.",
                    "source_id": gene_payload.get("id"),
                    "url": f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={gene_payload.get('id')}",
                }
            )
            variant = _first_variant(context)
            if variant is not None:
                region = f"{variant.chrom.removeprefix('chr')}:{variant.pos}-{variant.pos}"
                overlap_url = f"{server}/overlap/region/human/{region}"
                overlap = self.client.get_json(overlap_url, params={"feature": "variation"}, headers=headers)
                urls.append(overlap_url)
                for item in overlap[:5] if isinstance(overlap, list) else []:
                    records.append(
                        {
                            "category": "variant_annotation",
                            "source": self.spec.name,
                            "label": str(item.get("id") or variant.label),
                            "summary": str(item.get("consequence_type") or item.get("source") or "Ensembl variation overlap"),
                            "source_id": item.get("id"),
                            "url": f"https://www.ensembl.org/Homo_sapiens/Variation/Explore?v={item.get('id')}",
                            "variant": variant.label,
                        }
                    )
            return SourceResult(self.spec.key, "ok", f"Queried Ensembl; {len(records)} record(s).", records, queried_urls=urls, elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=urls, elapsed_ms=_elapsed_ms(started))


class UcscConnector(BaseConnector):
    """UCSC Genome Browser public API connector."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        genome = "hg38" if context.genome_build == "hg38" else "hg19"
        urls: list[str] = []
        records: list[dict[str, Any]] = []
        try:
            search_url = "https://api.genome.ucsc.edu/search"
            payload = self.client.get_json(search_url, params={"search": context.gene, "genome": genome})
            urls.append(search_url)
            for group in payload.get("positionMatches", [])[:3]:
                for match in group.get("matches", [])[:3]:
                    records.append(
                        {
                            "category": "gene_annotation",
                            "source": self.spec.name,
                            "label": str(match.get("posName") or context.gene),
                            "summary": str(match.get("position") or ""),
                            "url": f"https://genome.ucsc.edu/cgi-bin/hgTracks?db={genome}&position={match.get('position', context.region)}",
                        }
                    )
            return SourceResult(self.spec.key, "ok", f"Queried UCSC; {len(records)} record(s).", records, queried_urls=urls, elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=urls, elapsed_ms=_elapsed_ms(started))


class LiteratureSearchConnector(BaseConnector):
    """Connector for open literature APIs with straightforward search endpoints."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        kind = self.spec.connector_kind
        term = _literature_query(context)
        urls: list[str] = []
        records: list[dict[str, Any]] = []
        try:
            if kind == "europe_pmc":
                url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                payload = self.client.get_json(url, params={"query": term, "format": "json", "pageSize": 5})
                urls.append(url)
                for item in payload.get("resultList", {}).get("result", [])[:5]:
                    records.append(
                        {
                            "category": "literature",
                            "source": self.spec.name,
                            "label": str(item.get("title") or item.get("id") or term),
                            "summary": str(item.get("abstractText") or item.get("journalTitle") or ""),
                            "source_id": item.get("id"),
                            "url": str(item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url") or f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}"),
                            "variant": _first_rsid(context),
                        }
                    )
            elif kind == "openalex":
                url = "https://api.openalex.org/works"
                payload = self.client.get_json(url, params={"search": term, "per-page": 5})
                urls.append(url)
                for item in payload.get("results", [])[:5]:
                    records.append(
                        {
                            "category": "literature",
                            "source": self.spec.name,
                            "label": str(item.get("title") or item.get("id") or term),
                            "summary": str((item.get("primary_location") or {}).get("source", {}).get("display_name") or ""),
                            "source_id": item.get("id"),
                            "url": str(item.get("doi") or item.get("id") or ""),
                            "variant": _first_rsid(context),
                        }
                    )
            elif kind == "crossref":
                url = "https://api.crossref.org/works"
                payload = self.client.get_json(url, params={"query": term, "rows": 5})
                urls.append(url)
                for item in payload.get("message", {}).get("items", [])[:5]:
                    title = " ".join(item.get("title", [])[:1])
                    records.append(
                        {
                            "category": "literature",
                            "source": self.spec.name,
                            "label": title or str(item.get("DOI") or term),
                            "summary": str(item.get("container-title", [""])[0] if item.get("container-title") else ""),
                            "source_id": item.get("DOI"),
                            "url": str(item.get("URL") or ""),
                            "variant": _first_rsid(context),
                        }
                    )
            elif kind == "semantic_scholar":
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                payload = self.client.get_json(url, params={"query": term, "limit": 5, "fields": "title,year,url,abstract"})
                urls.append(url)
                for item in payload.get("data", [])[:5]:
                    records.append(
                        {
                            "category": "literature",
                            "source": self.spec.name,
                            "label": str(item.get("title") or term),
                            "summary": str(item.get("abstract") or ""),
                            "source_id": item.get("paperId"),
                            "url": str(item.get("url") or ""),
                            "variant": _first_rsid(context),
                        }
                    )
            elif kind in {"biorxiv", "medrxiv"}:
                # The public details endpoint is date-based, so v1 records a precise search link.
                records.append(
                    {
                        "category": "literature",
                        "source": self.spec.name,
                        "label": f"{self.spec.name} search for {term}",
                        "summary": "Preprint search linkout; article API is date-window based.",
                        "url": f"https://www.{kind}.org/search/{term.replace(' ', '%2520')}",
                        "variant": _first_rsid(context),
                    }
                )
            return SourceResult(self.spec.key, "ok", f"Queried {self.spec.name}; {len(records)} record(s).", records, queried_urls=urls, elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=urls, elapsed_ms=_elapsed_ms(started))


class GwasCatalogConnector(BaseConnector):
    """NHGRI-EBI GWAS Catalog connector."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        rsid = _first_rsid(context)
        records: list[dict[str, Any]] = []
        urls: list[str] = []
        if not rsid:
            return SourceResult(self.spec.key, "skipped", "No rsID was available for GWAS Catalog lookup.")
        try:
            url = f"https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/{rsid}"
            payload = self.client.get_json(url)
            urls.append(url)
            records.append(
                {
                    "category": "population_association",
                    "source": self.spec.name,
                    "label": rsid,
                    "summary": str(payload.get("rsId") or rsid),
                    "source_id": payload.get("rsId"),
                    "url": f"https://www.ebi.ac.uk/gwas/variants/{rsid}",
                    "variant": rsid,
                }
            )
            return SourceResult(self.spec.key, "ok", f"Queried GWAS Catalog for {rsid}.", records, queried_urls=urls, elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=urls, elapsed_ms=_elapsed_ms(started))


class PgsCatalogConnector(BaseConnector):
    """PGS Catalog connector."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        rsid = _first_rsid(context)
        if not rsid:
            return SourceResult(self.spec.key, "skipped", "No rsID was available for PGS Catalog lookup.")
        url = f"https://www.pgscatalog.org/rest/variant/{rsid}"
        try:
            payload = self.client.get_json(url)
            records = [
                {
                    "category": "polygenic_score",
                    "source": self.spec.name,
                    "label": rsid,
                    "summary": f"PGS Catalog variant record with {len(payload.get('associated_pgs_ids', []) or [])} linked score(s).",
                    "source_id": rsid,
                    "url": f"https://www.pgscatalog.org/variant/{rsid}/",
                    "variant": rsid,
                }
            ]
            return SourceResult(self.spec.key, "ok", f"Queried PGS Catalog for {rsid}.", records, queried_urls=[url], elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=[url], elapsed_ms=_elapsed_ms(started))


class EncodeConnector(BaseConnector):
    """ENCODE portal search connector."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        url = "https://www.encodeproject.org/search/"
        try:
            payload = self.client.get_json(
                url,
                params={"type": "Experiment", "searchTerm": context.gene, "format": "json", "limit": "5"},
            )
            records = [
                {
                    "category": "regulatory_experiment",
                    "source": self.spec.name,
                    "label": str(item.get("accession") or item.get("@id") or context.gene),
                    "summary": str(item.get("assay_title") or item.get("description") or "ENCODE experiment"),
                    "source_id": item.get("accession"),
                    "url": f"https://www.encodeproject.org{item.get('@id', '')}",
                }
                for item in payload.get("@graph", [])[:5]
            ]
            return SourceResult(self.spec.key, "ok", f"Queried ENCODE; {len(records)} record(s).", records, queried_urls=[url], elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=[url], elapsed_ms=_elapsed_ms(started))


class BioStudiesConnector(BaseConnector):
    """BioStudies search connector."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        url = "https://www.ebi.ac.uk/biostudies/api/v1/search"
        try:
            payload = self.client.get_json(url, params={"query": context.gene, "pageSize": 5})
            records = [
                {
                    "category": "public_dataset",
                    "source": self.spec.name,
                    "label": str(item.get("title") or item.get("accession") or context.gene),
                    "summary": str(item.get("description") or ""),
                    "source_id": item.get("accession"),
                    "url": f"https://www.ebi.ac.uk/biostudies/studies/{item.get('accession', '')}",
                }
                for item in payload.get("hits", [])[:5]
            ]
            return SourceResult(self.spec.key, "ok", f"Queried BioStudies; {len(records)} record(s).", records, queried_urls=[url], elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=[url], elapsed_ms=_elapsed_ms(started))


class GraphqlConnector(BaseConnector):
    """Best-effort GraphQL connector for gnomAD and CIViC."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        variant = _first_variant(context)
        urls: list[str] = []
        records: list[dict[str, Any]] = []
        if self.spec.connector_kind == "gnomad":
            if variant is None:
                return SourceResult(self.spec.key, "skipped", "No variant was available for gnomAD lookup.")
            variant_id = f"{variant.chrom.removeprefix('chr')}-{variant.pos}-{variant.ref}-{variant.alt.split(',')[0]}"
            url = "https://gnomad.broadinstitute.org/api/"
            query = """
            query Variant($variantId: String!, $dataset: DatasetId!) {
              variant(variantId: $variantId, dataset: $dataset) {
                variantId
                genome { ac an af }
                exome { ac an af }
              }
            }
            """
            variables = {"variantId": variant_id, "dataset": "gnomad_r4"}
            try:
                payload = self.client.post_json(url, json_payload={"query": query, "variables": variables})
                urls.append(url)
                item = payload.get("data", {}).get("variant")
                if item:
                    records.append(
                        {
                            "category": "population_frequency",
                            "source": self.spec.name,
                            "label": variant_id,
                            "summary": f"gnomAD frequency context for {variant_id}.",
                            "source_id": item.get("variantId"),
                            "url": f"https://gnomad.broadinstitute.org/variant/{variant_id}",
                            "variant": variant.label,
                            "frequencies": {"genome": item.get("genome"), "exome": item.get("exome")},
                        }
                    )
                return SourceResult(self.spec.key, "ok", f"Queried gnomAD; {len(records)} record(s).", records, queried_urls=urls, elapsed_ms=_elapsed_ms(started))
            except KnowledgeRequestError as exc:
                return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=urls, elapsed_ms=_elapsed_ms(started))

        url = "https://civicdb.org/api/graphql"
        query = """
        query Gene($name: String!) {
          gene(name: $name) {
            id
            name
            variants(first: 5) { nodes { id name } }
          }
        }
        """
        try:
            payload = self.client.post_json(url, json_payload={"query": query, "variables": {"name": context.gene}})
            urls.append(url)
            gene = payload.get("data", {}).get("gene") or {}
            for item in (gene.get("variants") or {}).get("nodes", [])[:5]:
                records.append(
                    {
                        "category": "cancer_variant",
                        "source": self.spec.name,
                        "label": str(item.get("name") or context.gene),
                        "summary": f"CIViC variant entry for {context.gene}.",
                        "source_id": item.get("id"),
                        "url": f"https://civicdb.org/links/variants/{item.get('id')}",
                    }
                )
            return SourceResult(self.spec.key, "ok", f"Queried CIViC; {len(records)} record(s).", records, queried_urls=urls, elapsed_ms=_elapsed_ms(started))
        except KnowledgeRequestError as exc:
            return SourceResult(self.spec.key, "failed", str(exc), errors=[str(exc)], queried_urls=urls, elapsed_ms=_elapsed_ms(started))


class ClinGenConnector(BaseConnector):
    """ClinGen connector backed by official gene-centered downloads and APIs."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        started = time.monotonic()
        urls: list[str] = []
        records: list[dict[str, Any]] = []
        warnings: list[str] = []
        request_errors: list[KnowledgeRequestError] = []

        download_steps = (
            ("Gene-Disease Validity", CLINGEN_VALIDITY_URL, self._validity_records, CLINGEN_PRIMARY_TIMEOUT_SECONDS),
            ("Dosage Sensitivity", CLINGEN_DOSAGE_URL, self._dosage_records, CLINGEN_PRIMARY_TIMEOUT_SECONDS),
            (
                "Curation Activity Summary",
                CLINGEN_SUMMARY_URL,
                self._summary_records,
                CLINGEN_OPTIONAL_TIMEOUT_SECONDS,
            ),
        )
        for label, url, parser, timeout_seconds in download_steps:
            try:
                text = self.client.get_text(
                    url,
                    rate_limit_per_second=self.spec.rate_limit_per_second,
                    timeout=timeout_seconds,
                )
                urls.append(url)
                records.extend(parser(text, context))
            except KnowledgeRequestError as exc:
                request_errors.append(exc)
                warnings.append(f"{label} request failed: {exc}")
            except (ValueError, csv.Error) as exc:
                warnings.append(f"{label} response could not be parsed: {exc}")

        for context_label, url in CLINGEN_ACTIONABILITY_URLS:
            try:
                payload = self.client.get_json(
                    url,
                    rate_limit_per_second=self.spec.rate_limit_per_second,
                    timeout=CLINGEN_OPTIONAL_TIMEOUT_SECONDS,
                )
                urls.append(url)
                records.extend(self._actionability_records(payload, context, context_label))
            except KnowledgeRequestError as exc:
                request_errors.append(exc)
                warnings.append(f"Clinical Actionability {context_label} request failed: {exc}")
            except (TypeError, ValueError) as exc:
                warnings.append(f"Clinical Actionability {context_label} response could not be parsed: {exc}")

        records = self._dedupe_records(records)
        if request_errors and not records and len(request_errors) == len(download_steps) + len(CLINGEN_ACTIONABILITY_URLS):
            first_error = request_errors[0]
            return _request_failure_result(self.spec, first_error, queried_urls=urls, started=started)

        if records:
            message = f"Queried ClinGen; {len(records)} gene-centered curation record(s) returned for {context.gene}."
        else:
            message = f"Queried ClinGen; no ClinGen curation records found for {context.gene}."
        return SourceResult(
            source_key=self.spec.key,
            status="ok",
            message=message,
            records=records,
            warnings=warnings,
            queried_urls=urls,
            elapsed_ms=_elapsed_ms(started),
        )

    def _validity_records(self, text: str, context: KnowledgeQuery) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in self._csv_dict_rows(text, "GENE SYMBOL"):
            if not self._gene_matches(row.get("GENE SYMBOL"), context):
                continue
            disease = _clean_cell(row.get("DISEASE LABEL"))
            classification = _clean_cell(row.get("CLASSIFICATION"))
            moi = _clean_cell(row.get("MOI"))
            gcep = _clean_cell(row.get("GCEP"))
            date = _clean_cell(row.get("CLASSIFICATION DATE"))
            report = _clean_cell(row.get("ONLINE REPORT"))
            summary_parts = [
                f"{classification} gene-disease validity" if classification else "Gene-disease validity curation",
                f"for {disease}" if disease else "",
                f"({moi})" if moi else "",
            ]
            record = {
                "category": "gene_disease_validity",
                "source": self.spec.name,
                "label": f"ClinGen validity: {context.gene}{' - ' + disease if disease else ''}",
                "summary": " ".join(part for part in summary_parts if part).strip(),
                "gene": context.gene,
                "hgnc_id": _clean_cell(row.get("GENE ID (HGNC)")),
                "disease": disease,
                "mondo_id": _clean_cell(row.get("DISEASE ID (MONDO)")),
                "mode_of_inheritance": moi,
                "classification": classification,
                "assertion": classification,
                "sop": _clean_cell(row.get("SOP")),
                "expert_panel": gcep,
                "date": date,
                "url": report or self.spec.homepage,
                "provenance_url": CLINGEN_VALIDITY_URL,
            }
            records.append(record)
        return records

    def _dosage_records(self, text: str, context: KnowledgeQuery) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in self._csv_dict_rows(text, "GENE SYMBOL"):
            if not self._gene_matches(row.get("GENE SYMBOL"), context):
                continue
            haplo = _clean_cell(row.get("HAPLOINSUFFICIENCY"))
            triplo = _clean_cell(row.get("TRIPLOSENSITIVITY"))
            report = _clean_cell(row.get("ONLINE REPORT"))
            date = _clean_cell(row.get("DATE"))
            records.append(
                {
                    "category": "dosage_sensitivity",
                    "source": self.spec.name,
                    "label": f"ClinGen dosage sensitivity: {context.gene}",
                    "summary": f"Haploinsufficiency: {haplo or 'not reported'}; triplosensitivity: {triplo or 'not reported'}.",
                    "gene": context.gene,
                    "hgnc_id": _clean_cell(row.get("HGNC ID")),
                    "haploinsufficiency": haplo,
                    "triplosensitivity": triplo,
                    "date": date,
                    "url": report or self.spec.homepage,
                    "provenance_url": CLINGEN_DOSAGE_URL,
                }
            )
        return records

    def _summary_records(self, text: str, context: KnowledgeQuery) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in self._csv_dict_rows(text, "gene_symbol"):
            if not self._gene_matches(row.get("gene_symbol"), context):
                continue
            disease = _clean_cell(row.get("disease_label"))
            validity = _clean_cell(row.get("gene_disease_validity_assertion_classifications"))
            actionability = _clean_cell(row.get("actionability_assertion_classifications"))
            dosage_haplo = _clean_cell(row.get("dosage_haploinsufficiency_assertion"))
            dosage_triplo = _clean_cell(row.get("dosage_triplosensitivity_assertion"))
            parts = []
            if validity:
                parts.append(f"validity: {validity}")
            if actionability:
                parts.append(f"actionability: {actionability}")
            if dosage_haplo or dosage_triplo:
                parts.append(f"dosage: HI {dosage_haplo or 'n/a'}, TS {dosage_triplo or 'n/a'}")
            records.append(
                {
                    "category": "clinical_gene_curation_summary",
                    "source": self.spec.name,
                    "label": f"ClinGen curation summary: {context.gene}{' - ' + disease if disease else ''}",
                    "summary": "; ".join(parts) or "ClinGen curation summary row.",
                    "gene": context.gene,
                    "hgnc_id": _clean_cell(row.get("hgnc_id")),
                    "disease": disease,
                    "mondo_id": _clean_cell(row.get("mondo_id")),
                    "mode_of_inheritance": _clean_cell(row.get("mode_of_inheritance")),
                    "classification": validity,
                    "actionability": actionability,
                    "haploinsufficiency": dosage_haplo,
                    "triplosensitivity": dosage_triplo,
                    "expert_panel": _clean_cell(row.get("gene_disease_validity_gceps")),
                    "actionability_group": _clean_cell(row.get("actionability_groups")),
                    "url": _clean_cell(row.get("gene_url")) or self.spec.homepage,
                    "report_url": _clean_cell(row.get("gene_disease_validity_assertion_reports")),
                    "actionability_report_url": _clean_cell(row.get("actionability_assertion_reports")),
                    "dosage_report_url": _clean_cell(row.get("dosage_report")),
                    "provenance_url": CLINGEN_SUMMARY_URL,
                }
            )
        return records

    def _actionability_records(
        self,
        payload: Any,
        context: KnowledgeQuery,
        actionability_context: str,
    ) -> list[dict[str, Any]]:
        rows = self._flat_json_rows(payload)
        records: list[dict[str, Any]] = []
        for row in rows:
            if not self._gene_matches(row.get("geneOrVariant"), context):
                continue
            disease = _clean_cell(row.get("disease"))
            overall = _clean_cell(row.get("overall"))
            intervention = _clean_cell(row.get("intervention"))
            doc_id = _clean_cell(row.get("docId"))
            records.append(
                {
                    "category": "clinical_actionability",
                    "source": self.spec.name,
                    "label": f"ClinGen actionability ({actionability_context}): {context.gene}{' - ' + disease if disease else ''}",
                    "summary": (
                        f"Overall actionability score {overall or 'not reported'}"
                        f"{'; intervention: ' + intervention if intervention else ''}."
                    ),
                    "gene": context.gene,
                    "disease": disease,
                    "classification": _clean_cell(row.get("status-overall")),
                    "actionability_score": overall,
                    "outcome": _clean_cell(row.get("outcome")),
                    "intervention": intervention,
                    "severity": _clean_cell(row.get("severity")),
                    "likelihood": _clean_cell(row.get("likelihood")),
                    "nature_of_intervention": _clean_cell(row.get("natureOfIntervention")),
                    "effectiveness": _clean_cell(row.get("effectiveness")),
                    "context": actionability_context,
                    "date": _clean_cell(row.get("releaseDate") or row.get("lastUpdated")),
                    "url": _clean_cell(row.get("contextIri")) or self.spec.homepage,
                    "source_id": doc_id,
                    "provenance_url": dict(CLINGEN_ACTIONABILITY_URLS).get(actionability_context, ""),
                }
            )
        return records

    def _csv_dict_rows(self, text: str, first_column: str) -> list[dict[str, str]]:
        reader = csv.reader(io.StringIO(text))
        header: list[str] | None = None
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]
            if not any(row):
                continue
            if header is None:
                if row and row[0] == first_column:
                    header = row
                continue
            if row[0].startswith("+"):
                continue
            normalized = row + [""] * max(0, len(header) - len(row))
            rows.append(dict(zip(header, normalized[: len(header)])))
        if header is None:
            raise ValueError(f"Could not find ClinGen CSV header starting with {first_column!r}.")
        return rows

    def _flat_json_rows(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("ClinGen actionability response was not a JSON object.")
        columns = payload.get("columns")
        rows = payload.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            raise ValueError("ClinGen actionability response did not include columns and rows.")
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            padded = row + [""] * max(0, len(columns) - len(row))
            normalized_rows.append(dict(zip([str(column) for column in columns], padded[: len(columns)])))
        return normalized_rows

    def _gene_matches(self, value: Any, context: KnowledgeQuery) -> bool:
        return _clean_cell(value).upper() == context.gene.upper()

    def _dedupe_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for record in records:
            key = (
                _clean_cell(record.get("category")),
                _clean_cell(record.get("label")).lower(),
                _clean_cell(record.get("url")),
                _clean_cell(record.get("provenance_url")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped


class LinkoutOfficialConnector(BaseConnector):
    """Official-source linkout connector for open resources without a compact stable API path."""

    def query(self, context: KnowledgeQuery) -> SourceResult:
        record = self._metadata_record()
        term = _literature_query(context)
        record["label"] = f"{self.spec.name} query context for {term}"
        record["summary"] = (
            f"{self.spec.name} is included in the dynamic KB registry. This v1 connector records "
            "source metadata and uses direct linkout/provenance until a compact official API shape is added."
        )
        return SourceResult(
            source_key=self.spec.key,
            status="metadata_only",
            message="Official source registered; linkout metadata recorded.",
            records=[record],
        )


CONNECTOR_CLASSES = {
    "metadata": BaseConnector,
    "auth_metadata": AuthMetadataConnector,
    "licensed_metadata": LicensedMetadataConnector,
    "clinvar": ClinVarConnector,
    "dbsnp": NcbiEutilsConnector,
    "ncbi_gene": NcbiEutilsConnector,
    "pubmed": NcbiEutilsConnector,
    "pmc": NcbiEutilsConnector,
    "geo": NcbiEutilsConnector,
    "litvar": NcbiEutilsConnector,
    "clingen": ClinGenConnector,
    "ensembl": EnsemblConnector,
    "ucsc": UcscConnector,
    "europe_pmc": LiteratureSearchConnector,
    "openalex": LiteratureSearchConnector,
    "crossref": LiteratureSearchConnector,
    "semantic_scholar": LiteratureSearchConnector,
    "biorxiv": LiteratureSearchConnector,
    "medrxiv": LiteratureSearchConnector,
    "gwas_catalog": GwasCatalogConnector,
    "pgs_catalog": PgsCatalogConnector,
    "encode": EncodeConnector,
    "screen": EncodeConnector,
    "biostudies": BioStudiesConnector,
    "gnomad": GraphqlConnector,
    "civic": GraphqlConnector,
    "pharmgkb": LinkoutOfficialConnector,
    "mavedb": LinkoutOfficialConnector,
    "panelapp": LinkoutOfficialConnector,
    "pharmvar": LinkoutOfficialConnector,
    "cpic": LinkoutOfficialConnector,
    "fda_pgx": LinkoutOfficialConnector,
    "dgidb": LinkoutOfficialConnector,
}


def connector_for(
    spec: SourceSpec,
    client: RequestClient,
    credential: ResolvedCredential,
) -> BaseConnector:
    """Return a connector instance for one source spec."""
    connector_class = CONNECTOR_CLASSES.get(spec.connector_kind, BaseConnector)
    return connector_class(spec, client, credential)
