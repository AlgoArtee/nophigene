"""Database connectors used by dynamic variant knowledge-base generation."""

from __future__ import annotations

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
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search = self.client.get_json(
                search_url,
                params={"db": db, "term": term, "retmode": "json", "retmax": 5},
                rate_limit_per_second=self.spec.rate_limit_per_second,
            )
            urls.append(search_url)
            ids = list(search.get("esearchresult", {}).get("idlist", []))
            if ids:
                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                summary = self.client.get_json(
                    summary_url,
                    params={"db": db, "id": ",".join(ids[:5]), "retmode": "json"},
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
            return SourceResult(
                source_key=self.spec.key,
                status="failed",
                message=str(exc),
                errors=[str(exc)],
                queried_urls=urls,
                elapsed_ms=_elapsed_ms(started),
            )

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
    "clinvar": NcbiEutilsConnector,
    "dbsnp": NcbiEutilsConnector,
    "ncbi_gene": NcbiEutilsConnector,
    "pubmed": NcbiEutilsConnector,
    "pmc": NcbiEutilsConnector,
    "geo": NcbiEutilsConnector,
    "litvar": NcbiEutilsConnector,
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
