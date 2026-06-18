"""Fetch candidate genomic intervals from multiple public annotation APIs."""

from __future__ import annotations

import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import requests

DEFAULT_TIMEOUT_SECONDS = 30
GENE_DATA_DIR = Path(__file__).resolve().parent / "gene_data"
GENE_DATA_BUNDLE_PATH = GENE_DATA_DIR / "gene_data_bundle.zip"
GENE_DATA_INDEX_PATH = GENE_DATA_DIR / "gene_data_index.json"
LOCAL_CURATED_SOURCE = "Local curated gene bundle"


def fetch_refseq_region(gene_symbol: str = "DRD4") -> str | None:
    """Look up a gene interval with the NCBI Entrez Gene APIs.

    The function first resolves the gene symbol to an Entrez Gene identifier and
    then fetches the gene summary payload to extract the genomic coordinates.
    Returning ``None`` instead of raising on lookup problems lets the caller
    combine several public sources opportunistically and decide later which
    interval to trust.

    Parameters
    ----------
    gene_symbol : str, optional
        HGNC-style gene symbol to search for. Defaults to ``"DRD4"``.

    Returns
    -------
    str | None
        Region string in ``chrom:start-end`` format, or ``None`` when the API
        does not return a usable interval.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    try:
        search_response = requests.get(
            base_url + "esearch.fcgi",
            params={
                "db": "gene",
                "term": f"{gene_symbol}[sym] AND Homo sapiens[orgn]",
                "retmode": "json",
            },
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        search_response.raise_for_status()
        gene_ids = search_response.json().get("esearchresult", {}).get("idlist", [])
        if not gene_ids:
            return None

        gene_id = gene_ids[0]
        summary_response = requests.get(
            base_url + "esummary.fcgi",
            params={"db": "gene", "id": gene_id, "retmode": "json"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        summary_response.raise_for_status()
        genomic_info = summary_response.json().get("result", {}).get(gene_id, {}).get("genomicinfo", [])
        if not genomic_info:
            return None

        location = genomic_info[0]
        chr_num = location.get("chr")
        start = location.get("chrstart")
        end = location.get("chrstop")
        if chr_num is None or start is None or end is None:
            return None
    except (requests.RequestException, ValueError):
        return None

    return f"{chr_num}:{min(start, end)}-{max(start, end)}"


def fetch_ensembl_region(
    gene_symbol: str = "DRD4",
    server: str = "https://grch37.rest.ensembl.org",
) -> str | None:
    """Resolve a gene interval from the Ensembl REST API.

    Parameters
    ----------
    gene_symbol : str, optional
        Gene symbol to resolve. Defaults to ``"DRD4"``.
    server : str, optional
        Base URL for the Ensembl REST server. The default points to the GRCh37
        archive because the rest of the repository is largely hg19/GRCh37-based.

    Returns
    -------
    str | None
        Region string in ``chrom:start-end`` format, or ``None`` when Ensembl
        does not return a successful lookup payload.
    """
    try:
        response = requests.get(
            f"{server}/lookup/symbol/homo_sapiens/{gene_symbol}",
            params={"expand": 1},
            headers={"Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    seq_region_name = data.get("seq_region_name")
    start = data.get("start")
    end = data.get("end")
    if seq_region_name is None or start is None or end is None:
        return None
    return f"{seq_region_name}:{start}-{end}"


_UCSC_PREFERRED_TRACKS = [
    "hgnc",
    "ncbiRefSeqCurated",
    "knownGene",
    "refGene",
    "wgEncodeGencodeBasicV49",
    "wgEncodeGencodeBasicV48",
    "wgEncodeGencodeCompV49",
    "wgEncodeGencodeCompV48",
    "wgEncodeGencodeBasicV49lift37",
    "wgEncodeGencodeBasicV48lift37",
    "wgEncodeGencodeCompV49lift37",
    "wgEncodeGencodeCompV48lift37",
    "ensGene",
]


def _parse_ucsc_position(position: str) -> tuple[str, int, int] | None:
    """Parse a UCSC position string into normalized chromosome coordinates."""
    match = re.fullmatch(
        r"chr(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)",
        str(position).replace(",", "").strip(),
    )
    if match is None:
        return None

    start = int(match.group("start"))
    end = int(match.group("end"))
    if start > end:
        start, end = end, start
    return match.group("chrom"), start, end


def _ucsc_match_symbol(match: dict[str, Any]) -> str:
    """Extract the leading gene symbol from a UCSC search match."""
    pos_name = str(match.get("posName", "")).strip()
    if not pos_name:
        return ""
    return re.split(r"[\s(]", pos_name, maxsplit=1)[0].strip()


def fetch_ucsc_region(gene_symbol: str = "DRD4", genome: str = "hg19") -> str | None:
    """Look up a gene interval from UCSC search results for the chosen assembly.

    Parameters
    ----------
    gene_symbol : str, optional
        HGNC-style gene symbol to match against UCSC position search results.
    genome : str, optional
        UCSC genome assembly name. Defaults to ``"hg19"`` to match the rest of
        this project and the Illumina EPIC manifest coordinates.

    Returns
    -------
    str | None
        Preferred exact-symbol interval returned by UCSC, or ``None`` if no
        matching genomic position is found.
    """
    cleaned_symbol = _validate_gene_symbol(gene_symbol).upper()

    try:
        response = requests.get(
            "https://api.genome.ucsc.edu/search",
            params={"search": cleaned_symbol, "genome": genome},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    regions_by_track: dict[str, list[tuple[str, int, int]]] = {}
    for track_group in data.get("positionMatches", []):
        track_name = str(track_group.get("trackName") or track_group.get("name") or "")
        if not track_name:
            continue
        for match in track_group.get("matches", []):
            if _ucsc_match_symbol(match).upper() != cleaned_symbol:
                continue
            parsed = _parse_ucsc_position(str(match.get("position", "")))
            if parsed is None:
                continue
            regions_by_track.setdefault(track_name, []).append(parsed)

    for track_name in _UCSC_PREFERRED_TRACKS:
        regions = regions_by_track.get(track_name)
        if not regions:
            continue
        chroms = {chrom for chrom, _start, _end in regions}
        if len(chroms) != 1:
            continue
        chrom = next(iter(chroms))
        start = min(start for _chrom, start, _end in regions)
        end = max(end for _chrom, _start, end in regions)
        return f"{chrom}:{start}-{end}"

    return None


def _validate_gene_symbol(gene_symbol: str) -> str:
    """Normalize and validate a gene symbol before external API lookups."""
    cleaned = gene_symbol.strip()
    if not cleaned:
        raise ValueError("Enter a gene symbol before requesting genomic coordinates.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise ValueError(
            "Gene symbols may only contain letters, digits, dots, underscores, and hyphens."
        )
    return cleaned


def _normalize_genome_build(genome_build: str) -> str:
    """Normalize user-facing build labels to the app's internal keys."""
    cleaned = str(genome_build or "hg19").strip().lower().replace(" ", "")
    if cleaned in {"hg38", "grch38", "grch38/hg38", "hg38/grch38"}:
        return "hg38"
    if cleaned in {"hg19", "grch37", "grch37/hg19", "hg19/grch37"}:
        return "hg19"
    raise ValueError("Genome build must be hg19/GRCh37 or hg38/GRCh38.")


def _sanitize_gene_name_for_filename(gene_symbol: str) -> str:
    """Return the filename-safe gene stem used by bundled gene data."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", gene_symbol.strip())
    return sanitized.strip("_") or "gene"


def _candidate_interpretation_filenames(gene_symbol: str) -> list[str]:
    """Return likely bundled interpretation DB filenames for a gene."""
    sanitized = _sanitize_gene_name_for_filename(gene_symbol)
    candidates = [
        f"{sanitized.lower()}_interpretation_db.json",
        f"{sanitized}_interpretation_db.json",
        f"{sanitized.upper()}_interpretation_db.json",
    ]
    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=8)
def _gene_data_bundle_members(bundle_path: str) -> frozenset[str]:
    """Return filenames available in the compressed curated bundle."""
    path = Path(bundle_path)
    if not path.exists():
        return frozenset()
    try:
        with zipfile.ZipFile(path) as bundle:
            return frozenset(info.filename for info in bundle.infolist() if not info.is_dir())
    except (OSError, zipfile.BadZipFile):
        return frozenset()


@lru_cache(maxsize=4096)
def _read_gene_data_bundle_member(bundle_path: str, member_name: str) -> bytes | None:
    """Read one curated bundle member when it exists."""
    if member_name not in _gene_data_bundle_members(bundle_path):
        return None
    try:
        with zipfile.ZipFile(bundle_path) as bundle:
            return bundle.read(member_name)
    except (OSError, zipfile.BadZipFile, KeyError):
        return None


@lru_cache(maxsize=8)
def _gene_data_index(index_path: str) -> dict[str, Any]:
    """Return the sharded bulk gene-data index when available."""
    path = Path(index_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=4096)
def _read_bulk_gene_data_member(
    index_path: str,
    gene_data_dir: str,
    member_name: str,
) -> bytes | None:
    """Read one sharded bulk gene-data member when indexed."""
    index = _gene_data_index(index_path)
    files = index.get("files", {})
    shards = index.get("shards", {})
    if not isinstance(files, dict) or not isinstance(shards, dict):
        return None
    shard_name = files.get(member_name)
    if not isinstance(shard_name, str):
        return None
    shard_info = shards.get(shard_name, {})
    relative_path = shard_info.get("path") if isinstance(shard_info, dict) else None
    if not isinstance(relative_path, str):
        relative_path = f"bulk_gene_data_shards/{shard_name}"
    shard_path = Path(gene_data_dir) / relative_path
    if not shard_path.exists():
        return None
    try:
        with zipfile.ZipFile(shard_path) as bundle:
            return bundle.read(member_name)
    except (OSError, zipfile.BadZipFile, KeyError):
        return None


def _read_gene_data_text(filename: str) -> str | None:
    """Read one gene-data artifact from loose, zipped, or sharded storage."""
    loose_path = GENE_DATA_DIR / filename
    if loose_path.exists():
        try:
            return loose_path.read_text(encoding="utf-8")
        except OSError:
            return None
    bundled_bytes = _read_gene_data_bundle_member(str(GENE_DATA_BUNDLE_PATH), filename)
    if bundled_bytes is not None:
        return bundled_bytes.decode("utf-8")
    bulk_bytes = _read_bulk_gene_data_member(
        str(GENE_DATA_INDEX_PATH),
        str(GENE_DATA_DIR),
        filename,
    )
    if bulk_bytes is not None:
        return bulk_bytes.decode("utf-8")
    return None


def _load_local_interpretation_database(gene_symbol: str) -> dict[str, Any] | None:
    """Load a bundled gene interpretation DB without importing the full analysis stack."""
    for filename in _candidate_interpretation_filenames(gene_symbol):
        payload_text = _read_gene_data_text(filename)
        if payload_text is None:
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _knowledge_base_genome_build(knowledge_base: dict[str, Any]) -> str | None:
    """Return the assembly declared by a local curated gene database."""
    context = knowledge_base.get("gene_context", {})
    assembly = str(context.get("assembly", "") if isinstance(context, dict) else "").lower()
    if "hg38" in assembly or "grch38" in assembly:
        return "hg38"
    if "hg19" in assembly or "grch37" in assembly:
        return "hg19"
    return None


def _format_local_interval(record: Any, *, default_chrom: str = "") -> str:
    """Normalize a curated interval record to chrom:start-end text."""
    if isinstance(record, str):
        cleaned = record.replace(",", "").strip()
        match = re.fullmatch(r"(?:chr)?(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)", cleaned)
        if match is None:
            return ""
        chrom = match.group("chrom")
        start = int(match.group("start"))
        end = int(match.group("end"))
        return f"{chrom}:{min(start, end)}-{max(start, end)}"

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
    return f"{chrom.removeprefix('chr')}:{min(start, end)}-{max(start, end)}"


def _fetch_local_curated_region(
    cleaned_symbol: str,
    normalized_build: str,
) -> dict[str, str] | None:
    """Return a bundled curated gene interval for offline/public-API fallback."""
    knowledge_base = _load_local_interpretation_database(cleaned_symbol)
    if not knowledge_base:
        return None
    context = knowledge_base.get("gene_context", {})
    if not isinstance(context, dict):
        return None
    curated_gene = str(context.get("gene_name") or knowledge_base.get("gene_name") or "").upper()
    if curated_gene and curated_gene != cleaned_symbol.upper():
        return None
    declared_build = _knowledge_base_genome_build(knowledge_base)
    if declared_build and declared_build != normalized_build:
        return None
    chrom = str(context.get("chromosome", "")).strip()
    region = _format_local_interval(context.get("gene_region"), default_chrom=chrom)
    if not region:
        region = _format_local_interval(
            context.get("recommended_promoter_plus_gene_region"),
            default_chrom=chrom,
        )
    if not region:
        return None
    return {"source": LOCAL_CURATED_SOURCE, "region": region}


def find_gene_region(gene_symbol: str = "DRD4", genome_build: str = "hg19") -> dict[str, object]:
    """Resolve a gene symbol to the widest candidate interval across public sources.

    Parameters
    ----------
    gene_symbol : str, optional
        HGNC-style gene symbol to look up.

    Returns
    -------
    dict[str, object]
        Structured lookup result containing the selected region plus the
        individual source candidates that were found.

    Raises
    ------
    ValueError
        Raised when the symbol is blank or when none of the configured sources
        returns a usable interval.
    """
    cleaned_symbol = _validate_gene_symbol(gene_symbol)
    normalized_build = _normalize_genome_build(genome_build)
    if normalized_build == "hg38":
        ensembl_label = "Ensembl GRCh38"
        ensembl_server = "https://rest.ensembl.org"
        ucsc_label = "UCSC hg38"
        ucsc_genome = "hg38"
    else:
        ensembl_label = "Ensembl GRCh37"
        ensembl_server = "https://grch37.rest.ensembl.org"
        ucsc_label = "UCSC knownGene"
        ucsc_genome = "hg19"

    source_candidates = [
        ("NCBI RefSeq", fetch_refseq_region(cleaned_symbol)),
        (ensembl_label, fetch_ensembl_region(cleaned_symbol, server=ensembl_server)),
        (ucsc_label, fetch_ucsc_region(cleaned_symbol, genome=ucsc_genome)),
    ]
    candidates = [
        {"source": source_name, "region": region}
        for source_name, region in source_candidates
        if region
    ]
    if not candidates:
        local_candidate = _fetch_local_curated_region(cleaned_symbol, normalized_build)
        if local_candidate:
            candidates.append(local_candidate)
    if not candidates:
        raise ValueError(
            f"No genomic interval could be resolved for gene symbol '{cleaned_symbol}'."
        )

    selected_region = get_widest_region(candidate["region"] for candidate in candidates)
    selected_sources = [
        candidate["source"] for candidate in candidates if candidate["region"] == selected_region
    ]
    return {
        "gene_name": cleaned_symbol.upper(),
        "genome_build": normalized_build,
        "selected_region": selected_region,
        "selected_sources": selected_sources,
        "candidate_regions": candidates,
    }


def get_widest_region(regions: Iterable[str]) -> str | None:
    """Return the widest interval from an iterable of ``chrom:start-end`` strings.

    Parameters
    ----------
    regions : Iterable[str]
        Candidate genomic intervals collected from one or more sources.

    Returns
    -------
    str | None
        The interval with the greatest span, or ``None`` when the iterable is
        empty.
    """
    max_span = -1
    widest = None

    for region in regions:
        _chrom, genomic_range = region.split(":")
        start, end = map(int, genomic_range.split("-"))
        span = end - start
        if span > max_span:
            max_span = span
            widest = region

    return widest


def main() -> None:
    """Fetch DRD4 intervals from all configured sources and print a summary."""
    refseq_region = fetch_refseq_region()
    ensembl_region = fetch_ensembl_region()
    ucsc_region = fetch_ucsc_region()

    all_regions = [region for region in [refseq_region, ensembl_region, ucsc_region] if region]
    widest = get_widest_region(all_regions)

    print("RefSeq region:", refseq_region)
    print("Ensembl region:", ensembl_region)
    print("UCSC region:", ucsc_region)
    print("Widest region:", widest)


if __name__ == "__main__":
    main()
