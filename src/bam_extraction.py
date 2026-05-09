"""Helpers for extracting regional VCFs from GRCh38-aligned BAM files."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HG38_REFERENCE_DIR = DATA_DIR / "reference" / "hg38"
EXTRACTED_VCF_DIR = DATA_DIR / "extracted"

UCSC_HG38_ANALYSIS_SET_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/analysisSet"
)
HG38_FASTA_GZ_NAME = "hg38.analysisSet.fa.gz"
HG38_FASTA_NAME = "hg38.analysisSet.fa"
HG38_MD5_NAME = "md5sum.txt"

HG38_FASTA_GZ = HG38_REFERENCE_DIR / HG38_FASTA_GZ_NAME
HG38_FASTA = HG38_REFERENCE_DIR / HG38_FASTA_NAME
HG38_FASTA_FAI = HG38_REFERENCE_DIR / f"{HG38_FASTA_NAME}.fai"
HG38_MD5SUM = HG38_REFERENCE_DIR / HG38_MD5_NAME

DEFAULT_EXTRACTION_TOOLS = ("samtools", "bcftools")
_REGION_PATTERN = re.compile(
    r"^(?P<chrom>[^:\s]+):(?P<start>[\d,]+)-(?P<end>[\d,]+)$"
)


class ExtractionError(RuntimeError):
    """Raised when BAM extraction cannot be prepared or executed."""


@dataclass(frozen=True)
class ExtractionCommandPlan:
    """Shell command pieces used for BAM-to-VCF extraction."""

    bam_path: Path
    reference_fasta: Path
    output_vcf: Path
    input_region: str
    resolved_region: str
    samtools_index: list[str]
    mpileup: list[str]
    call: list[str]
    index: list[str]


def is_docker_runtime() -> bool:
    """Return whether the app appears to be running inside its Docker runtime."""
    return Path("/.dockerenv").exists() or os.environ.get("NOPHIGENE_IN_DOCKER") == "1"


def get_extraction_tool_status(
    tool_names: Sequence[str] = DEFAULT_EXTRACTION_TOOLS,
) -> dict[str, object]:
    """Report Docker/tool availability for the Extraction tab."""
    tools = {tool_name: shutil.which(tool_name) for tool_name in tool_names}
    missing_tools = [tool_name for tool_name, tool_path in tools.items() if not tool_path]
    docker_runtime = is_docker_runtime()
    local_override = os.environ.get("NOPHIGENE_ENABLE_LOCAL_EXTRACTION") == "1"
    available = not missing_tools and (docker_runtime or local_override)

    if available:
        message = "Extraction tools are available."
    elif missing_tools:
        message = (
            "Extraction is unavailable because samtools/bcftools are not on PATH. "
            "Start the Docker UI after rebuilding the image to enable this workflow."
        )
    else:
        message = (
            "Extraction is Docker-only by default. Start the Docker UI, or set "
            "NOPHIGENE_ENABLE_LOCAL_EXTRACTION=1 for a local expert setup."
        )

    return {
        "available": available,
        "docker_runtime": docker_runtime,
        "local_override": local_override,
        "tools": tools,
        "missing_tools": missing_tools,
        "message": message,
    }


def parse_region(region: str) -> tuple[str, int, int]:
    """Parse ``chrom:start-end`` text into chromosome, start, and end."""
    match = _REGION_PATTERN.fullmatch(str(region).replace(",", "").strip())
    if match is None:
        raise ExtractionError("Enter a genomic region in chrom:start-end format.")

    chrom = match.group("chrom")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start < 1 or end < 1 or start > end:
        raise ExtractionError("The extraction region must have a positive start before its end.")
    return chrom, start, end


def format_region(chrom: str, start: int, end: int) -> str:
    """Format a region without thousands separators."""
    return f"{chrom}:{int(start)}-{int(end)}"


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value and value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _ucsc_chrom_alias(chrom: str) -> str:
    cleaned = chrom.strip()
    if cleaned.lower().startswith("chr"):
        return "chrM" if cleaned.lower() in {"chrm", "chrmt"} else cleaned
    if cleaned.upper() in {"M", "MT"}:
        return "chrM"
    return f"chr{cleaned}"


def chromosome_aliases(chrom: str) -> list[str]:
    """Return likely contig aliases for a chromosome label."""
    cleaned = chrom.strip()
    aliases = [cleaned]
    if cleaned.lower().startswith("chr"):
        aliases.append(cleaned[3:])
    else:
        aliases.append(f"chr{cleaned}")

    if cleaned.upper() in {"M", "MT"} or cleaned.lower() in {"chrm", "chrmt"}:
        aliases.extend(["MT", "M", "chrM", "chrMT"])

    aliases.append(_ucsc_chrom_alias(cleaned))
    return _unique(aliases)


def resolve_bam_region(region: str, bam_contigs: Sequence[str] | None = None) -> str:
    """Resolve a user region to the contig naming style found in the BAM header."""
    chrom, start, end = parse_region(region)
    if not bam_contigs:
        return format_region(_ucsc_chrom_alias(chrom), start, end)

    contig_set = set(bam_contigs)
    for alias in chromosome_aliases(chrom):
        if alias in contig_set:
            return format_region(alias, start, end)

    alias_text = ", ".join(chromosome_aliases(chrom))
    raise ExtractionError(
        f"BAM header does not contain a contig matching {chrom}. Checked aliases: {alias_text}."
    )


def default_extracted_vcf_path(
    gene_name: str,
    genome_build: str = "hg38",
    analysis_scope: str = "promoter_plus_gene",
) -> Path:
    """Return the default extracted regional VCF path."""
    gene_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene_name.strip().upper()).strip("_")
    scope_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", analysis_scope.strip().lower()).strip("_")
    build_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", genome_build.strip().lower()).strip("_")
    return EXTRACTED_VCF_DIR / f"{gene_slug or 'GENE'}_{build_slug or 'hg38'}_{scope_slug or 'region'}.vcf.gz"


def _read_expected_md5(md5_path: Path, filename: str = HG38_FASTA_GZ_NAME) -> str | None:
    if not md5_path.exists():
        return None
    for line in md5_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        listed_name = parts[-1].lstrip("*").replace("\\", "/")
        if listed_name.endswith(filename):
            return parts[0].lower()
    return None


def compute_md5(path: Path) -> str:
    """Compute an MD5 digest with streaming reads."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_hg38_reference_status(
    reference_dir: Path = HG38_REFERENCE_DIR,
    *,
    verify_checksum: bool = False,
) -> dict[str, object]:
    """Return the local hg38 analysis-set reference status."""
    reference_dir = Path(reference_dir)
    fasta_gz = reference_dir / HG38_FASTA_GZ_NAME
    fasta = reference_dir / HG38_FASTA_NAME
    fai = reference_dir / f"{HG38_FASTA_NAME}.fai"
    md5sum = reference_dir / HG38_MD5_NAME

    expected_md5 = _read_expected_md5(md5sum)
    observed_md5 = (
        compute_md5(fasta_gz)
        if verify_checksum and fasta_gz.exists() and expected_md5
        else None
    )
    md5_verified = observed_md5 == expected_md5 if observed_md5 and expected_md5 else None
    checksum_mismatch = md5_verified is False
    ready = fasta.exists() and fai.exists() and not checksum_mismatch

    missing: list[str] = []
    if not fasta_gz.exists():
        missing.append(HG38_FASTA_GZ_NAME)
    if not md5sum.exists():
        missing.append(HG38_MD5_NAME)
    if not fasta.exists():
        missing.append(HG38_FASTA_NAME)
    if not fai.exists():
        missing.append(f"{HG38_FASTA_NAME}.fai")

    if checksum_mismatch:
        message = "Reference checksum mismatch. Re-download the hg38 analysis-set FASTA."
    elif ready and md5_verified is True:
        message = "Reference FASTA, checksum, and index are ready."
    elif ready:
        message = "Reference FASTA and index are ready."
    elif missing:
        message = "Reference is not ready. Missing: " + ", ".join(missing) + "."
    else:
        message = "Reference is not ready."

    return {
        "reference_dir": reference_dir,
        "fasta_gz": fasta_gz,
        "fasta": fasta,
        "fai": fai,
        "md5sum": md5sum,
        "fasta_gz_exists": fasta_gz.exists(),
        "fasta_exists": fasta.exists(),
        "fai_exists": fai.exists(),
        "md5sum_exists": md5sum.exists(),
        "expected_md5": expected_md5 or "",
        "observed_md5": observed_md5 or "",
        "md5_verified": md5_verified,
        "checksum_mismatch": checksum_mismatch,
        "ready": ready,
        "checksum_checked": verify_checksum and bool(expected_md5 and fasta_gz.exists()),
        "message": message,
    }


def _download_file(url: str, destination: Path, *, force: bool = False) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ExtractionError(
            f"Command failed ({command_to_string(command)}): {stderr or 'no command output'}"
        )
    return completed


def prepare_hg38_reference(
    reference_dir: Path = HG38_REFERENCE_DIR,
    *,
    force_download: bool = False,
    samtools: str | None = None,
) -> dict[str, object]:
    """Download, checksum, decompress, and index the UCSC hg38 analysis-set FASTA."""
    samtools_path = samtools or shutil.which("samtools")
    if not samtools_path:
        raise ExtractionError("samtools is required to index the hg38 reference FASTA.")

    reference_dir = Path(reference_dir)
    fasta_gz = reference_dir / HG38_FASTA_GZ_NAME
    fasta = reference_dir / HG38_FASTA_NAME
    fai = reference_dir / f"{HG38_FASTA_NAME}.fai"
    md5sum = reference_dir / HG38_MD5_NAME

    reference_dir.mkdir(parents=True, exist_ok=True)
    _download_file(f"{UCSC_HG38_ANALYSIS_SET_URL}/{HG38_MD5_NAME}", md5sum, force=force_download)
    _download_file(
        f"{UCSC_HG38_ANALYSIS_SET_URL}/{HG38_FASTA_GZ_NAME}",
        fasta_gz,
        force=force_download,
    )

    expected_md5 = _read_expected_md5(md5sum)
    if not expected_md5:
        raise ExtractionError("Could not find hg38.analysisSet.fa.gz in UCSC md5sum.txt.")
    observed_md5 = compute_md5(fasta_gz)
    if observed_md5 != expected_md5:
        raise ExtractionError(
            "Downloaded hg38.analysisSet.fa.gz did not match the UCSC md5 checksum."
        )

    if force_download or not fasta.exists():
        partial_fasta = fasta.with_suffix(fasta.suffix + ".partial")
        if partial_fasta.exists():
            partial_fasta.unlink()
        with gzip.open(fasta_gz, "rb") as source, partial_fasta.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        partial_fasta.replace(fasta)

    if force_download or not fai.exists():
        _run_command([samtools_path, "faidx", str(fasta)])

    return get_hg38_reference_status(reference_dir, verify_checksum=True)


def read_bam_contigs(bam_path: Path | str, *, samtools: str = "samtools") -> list[str]:
    """Read contig names from a BAM header with ``samtools view -H``."""
    bam_path = Path(bam_path)
    completed = _run_command([samtools, "view", "-H", str(bam_path)])
    contigs: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("@SQ"):
            continue
        for field in line.split("\t"):
            if field.startswith("SN:"):
                contigs.append(field.removeprefix("SN:"))
                break
    return contigs


def bam_index_candidates(bam_path: Path | str) -> list[Path]:
    """Return the common BAM index paths for a BAM file."""
    bam_path = Path(bam_path)
    candidates = [Path(f"{bam_path}.bai")]
    if bam_path.suffix.lower() == ".bam":
        candidates.append(bam_path.with_suffix(".bai"))
    return _unique_path(candidates)


def _unique_path(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths


def ensure_bam_index(bam_path: Path | str, *, samtools: str = "samtools") -> Path:
    """Return an existing BAM index, or create one with ``samtools index``."""
    bam_path = Path(bam_path)
    for index_path in bam_index_candidates(bam_path):
        if index_path.exists():
            return index_path
    _run_command([samtools, "index", str(bam_path)])
    for index_path in bam_index_candidates(bam_path):
        if index_path.exists():
            return index_path
    return Path(f"{bam_path}.bai")


def build_extraction_commands(
    *,
    bam_path: Path | str,
    region: str,
    reference_fasta: Path | str = HG38_FASTA,
    output_vcf: Path | str,
    bam_contigs: Sequence[str] | None = None,
    samtools: str = "samtools",
    bcftools: str = "bcftools",
) -> ExtractionCommandPlan:
    """Build the command plan for extracting a regional VCF."""
    bam = Path(bam_path)
    reference = Path(reference_fasta)
    output = Path(output_vcf)
    resolved_region = resolve_bam_region(region, bam_contigs)
    return ExtractionCommandPlan(
        bam_path=bam,
        reference_fasta=reference,
        output_vcf=output,
        input_region=str(region),
        resolved_region=resolved_region,
        samtools_index=[samtools, "index", str(bam)],
        mpileup=[
            bcftools,
            "mpileup",
            "-Ou",
            "-f",
            str(reference),
            "-r",
            resolved_region,
            "-a",
            "FORMAT/DP,FORMAT/AD",
            str(bam),
        ],
        call=[bcftools, "call", "-mv", "-Oz", "-o", str(output)],
        index=[bcftools, "index", "-t", str(output)],
    )


def command_to_string(command: Sequence[str]) -> str:
    """Render a command list for logs without invoking a shell."""
    return " ".join(str(part) for part in command)


def extract_region_vcf(
    *,
    bam_path: Path | str,
    region: str,
    output_vcf: Path | str,
    reference_fasta: Path | str = HG38_FASTA,
    samtools: str | None = None,
    bcftools: str | None = None,
) -> dict[str, object]:
    """Extract and index a regional VCF from a coordinate-sorted BAM."""
    samtools_path = samtools or shutil.which("samtools")
    bcftools_path = bcftools or shutil.which("bcftools")
    if not samtools_path or not bcftools_path:
        raise ExtractionError("samtools and bcftools are required for BAM extraction.")

    bam = Path(bam_path)
    reference = Path(reference_fasta)
    output = Path(output_vcf)
    if not bam.exists():
        raise ExtractionError(f"BAM file does not exist: {bam}")
    if not reference.exists():
        raise ExtractionError(f"Reference FASTA does not exist: {reference}")
    if not Path(f"{reference}.fai").exists():
        raise ExtractionError(f"Reference FASTA index does not exist: {reference}.fai")

    output.parent.mkdir(parents=True, exist_ok=True)
    bam_index = ensure_bam_index(bam, samtools=samtools_path)
    contigs = read_bam_contigs(bam, samtools=samtools_path)
    plan = build_extraction_commands(
        bam_path=bam,
        region=region,
        reference_fasta=reference,
        output_vcf=output,
        bam_contigs=contigs,
        samtools=samtools_path,
        bcftools=bcftools_path,
    )

    mpileup_proc = subprocess.Popen(
        plan.mpileup,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    call_proc = subprocess.run(
        plan.call,
        stdin=mpileup_proc.stdout,
        capture_output=True,
        check=False,
    )
    if mpileup_proc.stdout is not None:
        mpileup_proc.stdout.close()
    mpileup_stderr = mpileup_proc.stderr.read().decode("utf-8", errors="replace") if mpileup_proc.stderr else ""
    mpileup_returncode = mpileup_proc.wait()
    call_stderr = call_proc.stderr.decode("utf-8", errors="replace")
    call_stdout = call_proc.stdout.decode("utf-8", errors="replace")

    if mpileup_returncode != 0:
        raise ExtractionError(
            f"bcftools mpileup failed: {mpileup_stderr.strip() or 'no command output'}"
        )
    if call_proc.returncode != 0:
        raise ExtractionError(
            f"bcftools call failed: {call_stderr.strip() or call_stdout.strip() or 'no command output'}"
        )
    if not output.exists():
        raise ExtractionError("bcftools completed but did not create an output VCF.")

    index_result = _run_command(plan.index)
    return {
        "bam_path": bam,
        "bam_index": bam_index,
        "output_vcf": output,
        "output_index": Path(f"{output}.tbi"),
        "reference_fasta": reference,
        "input_region": region,
        "resolved_region": plan.resolved_region,
        "bam_contigs": contigs,
        "commands": {
            "samtools_index": command_to_string(plan.samtools_index),
            "mpileup": command_to_string(plan.mpileup),
            "call": command_to_string(plan.call),
            "index": command_to_string(plan.index),
        },
        "stdout": "\n".join(part for part in [call_stdout, index_result.stdout] if part).strip(),
        "stderr": "\n".join(part for part in [mpileup_stderr, call_stderr, index_result.stderr] if part).strip(),
    }


def write_empty_vcf(
    output_vcf: Path | str,
    *,
    reference_contigs: Sequence[str] | None = None,
) -> Path:
    """Write a valid header-only VCF for tests and no-call fallbacks."""
    output = Path(output_vcf)
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        for contig in reference_contigs or []:
            handle.write(f"##contig=<ID={contig}>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    return output
