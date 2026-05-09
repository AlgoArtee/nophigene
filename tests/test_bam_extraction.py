"""Tests for GRCh38 BAM extraction helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.bam_extraction import (
    HG38_FASTA_GZ_NAME,
    HG38_FASTA_NAME,
    HG38_MD5_NAME,
    build_extraction_commands,
    default_extracted_vcf_path,
    get_hg38_reference_status,
    resolve_bam_region,
)


def test_reference_status_reports_missing_resources(tmp_path: Path) -> None:
    """A fresh reference directory should clearly report missing files."""
    status = get_hg38_reference_status(tmp_path, verify_checksum=True)

    assert status["ready"] is False
    assert status["fasta_gz_exists"] is False
    assert HG38_FASTA_GZ_NAME in status["message"]
    assert HG38_FASTA_NAME in status["message"]


def test_reference_status_detects_checksum_mismatch(tmp_path: Path) -> None:
    """Checksum verification should catch a stale or corrupt downloaded FASTA."""
    fasta_gz = tmp_path / HG38_FASTA_GZ_NAME
    fasta_gz.write_bytes(b"not the hg38 reference")
    (tmp_path / HG38_FASTA_NAME).write_text(">chr1\nACGT\n", encoding="utf-8")
    (tmp_path / f"{HG38_FASTA_NAME}.fai").write_text("chr1\t4\t6\t4\t5\n", encoding="utf-8")
    (tmp_path / HG38_MD5_NAME).write_text(
        "00000000000000000000000000000000  hg38.analysisSet.fa.gz\n",
        encoding="utf-8",
    )

    status = get_hg38_reference_status(tmp_path, verify_checksum=True)

    assert status["checksum_mismatch"] is True
    assert status["ready"] is False
    assert status["observed_md5"] == hashlib.md5(b"not the hg38 reference").hexdigest()


def test_reference_status_ready_with_fasta_and_index(tmp_path: Path) -> None:
    """A decompressed FASTA plus .fai should be enough to mark the reference usable."""
    (tmp_path / HG38_FASTA_NAME).write_text(">chr1\nACGT\n", encoding="utf-8")
    (tmp_path / f"{HG38_FASTA_NAME}.fai").write_text("chr1\t4\t6\t4\t5\n", encoding="utf-8")

    status = get_hg38_reference_status(tmp_path)

    assert status["ready"] is True
    assert status["message"] == "Reference FASTA and index are ready."


def test_command_construction_resolves_chr_prefixed_bam_contig() -> None:
    """A numeric region should become chr-prefixed when the BAM header uses UCSC names."""
    plan = build_extraction_commands(
        bam_path=Path("data/sample.hg38.bam"),
        region="15:21405401-21441499",
        reference_fasta=Path("data/reference/hg38/hg38.analysisSet.fa"),
        output_vcf=Path("data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz"),
        bam_contigs=["chr15", "chrM"],
    )

    assert plan.resolved_region == "chr15:21405401-21441499"
    assert plan.mpileup == [
        "bcftools",
        "mpileup",
        "-Ou",
        "-f",
        str(Path("data/reference/hg38/hg38.analysisSet.fa")),
        "-r",
        "chr15:21405401-21441499",
        "-a",
        "FORMAT/DP,FORMAT/AD",
        str(Path("data/sample.hg38.bam")),
    ]
    assert plan.call == [
        "bcftools",
        "call",
        "-mv",
        "-Oz",
        "-o",
        str(Path("data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz")),
    ]
    assert plan.index == [
        "bcftools",
        "index",
        "-t",
        str(Path("data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz")),
    ]


def test_contig_alias_resolution_handles_numeric_and_mitochondrial_names() -> None:
    """The extractor should bridge 15/chr15 and MT/chrM naming differences."""
    assert resolve_bam_region("15:21405401-21441499", ["chr15"]) == "chr15:21405401-21441499"
    assert resolve_bam_region("chr15:21405401-21441499", ["15"]) == "15:21405401-21441499"
    assert resolve_bam_region("MT:561-1601", ["chrM"]) == "chrM:561-1601"
    assert resolve_bam_region("chrM:561-1601", ["MT"]) == "MT:561-1601"


def test_default_resolution_prefers_ucsc_hg38_names_without_bam_header() -> None:
    """Before a BAM is inspected, default region formatting should match UCSC hg38."""
    assert resolve_bam_region("15:1-2") == "chr15:1-2"
    assert resolve_bam_region("MT:1-2") == "chrM:1-2"


def test_default_extracted_vcf_path_uses_gene_build_and_scope() -> None:
    """The UI default should land extracted calls in data/extracted."""
    output = default_extracted_vcf_path("POTEB3", "hg38", "promoter_plus_gene")

    assert output.as_posix().endswith("data/extracted/POTEB3_hg38_promoter_plus_gene.vcf.gz")
