"""Evidence-calibrated interpretation helpers.

This module deliberately keeps call quality, evidence strength, and study
applicability separate.  It is used by the schema-v2 report layer while the
legacy report fields remain available for compatibility.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


INTERPRETATION_SCHEMA_VERSION = "2.0"
INTERPRETATION_MODES = {"research", "clinical_support", "dual"}
CALL_QC_POLICY_VERSION = "vcf-call-qc-v1"
METHYLATION_POLICY_VERSION = "methylation-context-v1"

_CLINICAL_SOURCE_KEYS = {"clinvar", "clingen", "panelapp"}
_CRITICAL_QC_FLAGS = {
    "missing_gt",
    "filter_non_pass",
    "low_depth",
    "low_gq",
    "very_low_qual",
    "heterozygous_allelic_imbalance_severe",
}
_SYMBOLIC_ALLELE_PREFIXES = ("<", "*", ".")


def utc_now_iso() -> str:
    """Return a stable UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_interpretation_mode(value: Any) -> str:
    """Normalize a public report mode and reject unsafe implicit modes."""
    normalized = str(value or "research").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        normalized = "research"
    if normalized not in INTERPRETATION_MODES:
        raise ValueError("interpretation_mode must be research, clinical_support, or dual.")
    return normalized


def normalize_genome_build(value: Any) -> str:
    """Return ``hg19``, ``hg38``, or ``unknown`` without guessing a build."""
    normalized = str(value or "").strip().lower().replace(" ", "")
    if normalized in {"hg19", "grch37", "grch37/hg19", "hg19/grch37"}:
        return "hg19"
    if normalized in {"hg38", "grch38", "grch38/hg38", "hg38/grch38"}:
        return "hg38"
    return "unknown"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clean_allele(value: Any) -> str:
    return _clean_text(value).upper()


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _alt_alleles(row: pd.Series | dict[str, Any]) -> list[str]:
    source = row.get("alt_alleles") if hasattr(row, "get") else None
    values = _as_list(source)
    if not values:
        values = _as_list(row.get("alt") if hasattr(row, "get") else None)
    alleles: list[str] = []
    for value in values:
        for allele in str(value or "").split(","):
            normalized = _clean_allele(allele)
            if normalized and normalized not in {".", "NAN", "NONE"} and normalized not in alleles:
                alleles.append(normalized)
    return alleles


def _gt_codes(row: pd.Series | dict[str, Any]) -> list[int | None]:
    raw_codes = row.get("gt_codes") if hasattr(row, "get") else None
    codes: list[int | None] = []
    if raw_codes is not None and not (isinstance(raw_codes, float) and pd.isna(raw_codes)):
        for value in _as_list(raw_codes):
            text = _clean_text(value)
            if not text or text in {".", "NONE", "NAN"}:
                codes.append(None)
            else:
                try:
                    codes.append(int(float(text)))
                except ValueError:
                    codes.append(None)
        if codes:
            return codes
    raw_gt = _clean_text(row.get("gt_raw") if hasattr(row, "get") else "")
    if not raw_gt:
        return []
    return [int(token) if token.isdigit() else None for token in re.split(r"[|/]", raw_gt)]


def _minimize_allele(position: int, ref: str, alt: str) -> tuple[int, str, str, str]:
    """Trim a VCF allele to minimal representation without reference lookups."""
    if not ref or not alt or any(allele.startswith(_SYMBOLIC_ALLELE_PREFIXES) for allele in (ref, alt)):
        return position, ref, alt, "unsupported"
    normalized_position = position
    normalized_ref = ref
    normalized_alt = alt
    while len(normalized_ref) > 1 and len(normalized_alt) > 1 and normalized_ref[-1] == normalized_alt[-1]:
        normalized_ref = normalized_ref[:-1]
        normalized_alt = normalized_alt[:-1]
    while len(normalized_ref) > 1 and len(normalized_alt) > 1 and normalized_ref[0] == normalized_alt[0]:
        normalized_ref = normalized_ref[1:]
        normalized_alt = normalized_alt[1:]
        normalized_position += 1
    status = "minimal" if (normalized_position, normalized_ref, normalized_alt) != (position, ref, alt) else "as_reported"
    return normalized_position, normalized_ref, normalized_alt, status


def _format_spdi(build: str, chrom: str, pos: int, ref: str, alt: str) -> tuple[str, str]:
    """Return a SPDI only when a reference accession is supplied upstream.

    A contig label such as ``11`` is not a valid SPDI sequence identifier, so
    this function intentionally records why a canonical SPDI is unavailable
    instead of emitting a misleading identifier.
    """
    _ = (build, chrom, pos, ref, alt)
    return "", "reference_accession_required"


def canonical_variant_alleles(
    variants: pd.DataFrame,
    *,
    genome_build: str | None,
) -> list[dict[str, Any]]:
    """Split VCF sites into exact alternate-allele findings.

    The legacy variant table remains site-oriented.  This derived view is the
    only view used by schema-v2 external evidence matching.
    """
    build = normalize_genome_build(genome_build)
    alleles: list[dict[str, Any]] = []
    for source_index, (_, row) in enumerate(variants.iterrows()):
        chrom = _clean_text(row.get("chrom") or row.get("CHROM")).removeprefix("chr")
        position = _safe_int(row.get("pos") if "pos" in row else row.get("POS"))
        ref = _clean_allele(row.get("ref") if "ref" in row else row.get("REF"))
        rsid = _clean_text(row.get("id") if "id" in row else row.get("ID"))
        alt_values = _alt_alleles(row)
        gt_codes = _gt_codes(row)
        gt_raw = _clean_text(row.get("gt_raw"))
        is_phased = bool(row.get("phased", False)) or "|" in gt_raw
        allele_dosage = row.get("allele_dosage_per_alt") if hasattr(row, "get") else {}
        if not isinstance(allele_dosage, dict):
            allele_dosage = {}
        for alt_index, alt in enumerate(alt_values, start=1):
            dosage = sum(1 for code in gt_codes if code == alt_index)
            if not gt_codes:
                dosage = _safe_int(allele_dosage.get(alt)) or 0
            normalized_pos, normalized_ref, normalized_alt, minimal_status = _minimize_allele(
                position or 0, ref, alt
            )
            spdi, spdi_status = _format_spdi(build, chrom, normalized_pos, normalized_ref, normalized_alt)
            symbolic = minimal_status == "unsupported"
            identity_status = (
                "ready_for_external_matching"
                if build != "unknown" and chrom and position and ref and alt and not symbolic and len(ref) == len(alt)
                else "requires_reference_normalization"
                if build != "unknown" and chrom and position and ref and alt and not symbolic
                else "incomplete_identity"
            )
            allele_key = "|".join(
                [build, chrom, str(normalized_pos or ""), normalized_ref, normalized_alt]
            )
            alleles.append(
                {
                    "finding_key": f"allele:{source_index}:{alt_index}",
                    "source_site_index": source_index,
                    "sample": _clean_text(row.get("sample")),
                    "genome_build": build,
                    "chrom": chrom,
                    "pos": position,
                    "ref": ref,
                    "alt": alt,
                    "alt_index": alt_index,
                    "alt_dosage": dosage,
                    "observed_non_reference": dosage > 0,
                    "rsid": rsid if rsid.lower().startswith("rs") else "",
                    "allele_key": allele_key,
                    "normalization": {
                        "minimal_representation": minimal_status,
                        "normalized_pos": normalized_pos or None,
                        "normalized_ref": normalized_ref,
                        "normalized_alt": normalized_alt,
                        "reference_allele_validation": "not_performed",
                        "left_alignment": "not_performed_without_reference" if len(ref) != len(alt) else "not_required",
                        "identity_status": identity_status,
                    },
                    "spdi": spdi,
                    "spdi_status": spdi_status,
                    "vrs_id": "",
                    "vrs_status": "not_computed",
                    "genotype": _clean_text(row.get("genotype")),
                    "gt_raw": gt_raw or "./.",
                    "zygosity": _clean_text(row.get("zygosity")) or "missing",
                    "phased": is_phased,
                    "phase_status": "phased_call_present" if is_phased else "unphased_or_not_proven",
                    "haplotype_status": "not_inferred_from_small_variant_vcf",
                    "raw_call": {
                        "filter_status": _clean_text(row.get("filter_status")),
                        "filter_pass": bool(row.get("filter_pass", False)),
                        "qual": _safe_float(row.get("qual")),
                        "dp": _safe_int(row.get("dp")),
                        "gq": _safe_float(row.get("gq")),
                        "confidence_score": _safe_float(row.get("confidence_score")),
                        "qc_flags": [str(flag) for flag in _as_list(row.get("qc_flags")) if str(flag).strip()],
                    },
                }
            )
    return alleles


def assess_call_quality(allele: dict[str, Any]) -> dict[str, Any]:
    """Assess whether a decoded allele call is eligible for clinical support."""
    raw = allele.get("raw_call", {})
    flags = {str(flag) for flag in raw.get("qc_flags", [])}
    blockers: list[str] = []
    if not allele.get("observed_non_reference"):
        blockers.append("reference_or_unobserved_allele")
    if allele.get("gt_raw") in {"", "./.", ".", "None"} or "missing_gt" in flags:
        blockers.append("missing_gt")
    if not raw.get("filter_pass") and raw.get("filter_status", "").upper() != "PASS":
        blockers.append("non_pass_filter")
    score = raw.get("confidence_score")
    if score is None or score < 0.70:
        blockers.append("insufficient_call_confidence")
    blockers.extend(sorted(_CRITICAL_QC_FLAGS & flags))
    blockers = list(dict.fromkeys(blockers))
    return {
        "policy_version": CALL_QC_POLICY_VERSION,
        "status": "pass" if not blockers else "not_eligible",
        "eligible": not blockers,
        "confidence_score": score,
        "qc_flags": sorted(flags),
        "blockers": blockers,
    }


def _record_text_values(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("rsid", "variant", "display_name", "label", "source_id"):
        value = _clean_text(record.get(key)).casefold()
        if value:
            values.add(value)
    for value in _as_list(record.get("lookup_keys")):
        text = _clean_text(value).casefold()
        if text:
            values.add(text)
    return values


def source_record_matches_allele(record: dict[str, Any], allele: dict[str, Any]) -> bool:
    """Match evidence only to an exact rsID or coordinate+allele identity."""
    rsid = _clean_text(allele.get("rsid")).casefold()
    record_values = _record_text_values(record)
    source_build = normalize_genome_build(
        record.get("genome_build") or record.get("assembly") or record.get("assembly_name")
    )
    allele_build = normalize_genome_build(allele.get("genome_build"))
    if source_build != "unknown" and allele_build != "unknown" and source_build != allele_build:
        return False
    if rsid and rsid in record_values:
        return True
    chrom = _clean_text(allele.get("chrom")).removeprefix("chr")
    pos = _safe_int(allele.get("pos"))
    ref = _clean_allele(allele.get("ref"))
    alt = _clean_allele(allele.get("alt"))
    record_chrom = _clean_text(record.get("chromosome") or record.get("chrom")).removeprefix("chr")
    record_pos = _safe_int(record.get("position") or record.get("pos"))
    record_ref = _clean_allele(record.get("ref") or record.get("reference_allele"))
    record_alt = _clean_allele(record.get("alt") or record.get("alternate_allele"))
    if chrom and pos and ref and alt and record_chrom == chrom and record_pos == pos:
        if record_ref and record_alt:
            return record_ref == ref and record_alt == alt
        coordinate_tokens = {
            f"{chrom}:{pos}:{ref}>{alt}".casefold(),
            f"{chrom}:{pos}:{ref}>{alt.replace('>', '')}".casefold(),
        }
        return bool(coordinate_tokens & record_values)
    return False


def _evidence_item(record: dict[str, Any], *, source_key: str = "") -> dict[str, Any]:
    return {
        "source_key": _clean_text(source_key or record.get("source_key")),
        "source": _clean_text(record.get("source")),
        "category": _clean_text(record.get("category")),
        "source_id": _clean_text(record.get("source_id") or record.get("evidence_id")),
        "label": _clean_text(record.get("label") or record.get("variant")),
        "url": _clean_text(record.get("url")),
        "clinical_significance": record.get("clinical_significance") or record.get("assertion") or "",
        "review_status": _clean_text(record.get("review_status")),
        "assertion_criteria": _clean_text(record.get("assertion_criteria")),
        "conditions": record.get("phenotype") or record.get("traits") or [],
        "last_evaluated": _clean_text(record.get("last_evaluated") or record.get("updated_at")),
        "tissue": _clean_text(record.get("tissue")),
        "cohort": _clean_text(record.get("cohort") or record.get("study_id")),
        "effect_size": _clean_text(record.get("effect_size") or record.get("beta")),
        "confidence_interval": _clean_text(record.get("confidence_interval")),
        "p_value": _clean_text(record.get("p_value")),
        "replication_status": _clean_text(record.get("replication_status")),
        "transcript_consequence": record.get("transcript_consequence") or {},
        "genome_build": normalize_genome_build(
            record.get("genome_build") or record.get("assembly") or record.get("assembly_name")
        ),
        "reference_allele_validation": _clean_text(record.get("reference_allele_validation")),
        "spdi": _clean_text(record.get("spdi")),
        "vrs_id": _clean_text(record.get("vrs_id") or record.get("allele_registry_id")),
        "summary": _clean_text(record.get("summary"))[:600],
        "match": "exact_allele",
    }


def _dynamic_evidence_for_allele(dynamic_payload: dict[str, Any] | None, allele: dict[str, Any]) -> list[dict[str, Any]]:
    payload = dynamic_payload or {}
    rows = payload.get("source_records", [])
    return [
        _evidence_item(record)
        for record in rows
        if isinstance(record, dict) and source_record_matches_allele(record, allele)
    ]


def _curated_evidence_for_allele(knowledge_base: dict[str, Any], allele: dict[str, Any]) -> list[dict[str, Any]]:
    records = knowledge_base.get("variant_records", []) if isinstance(knowledge_base, dict) else []
    evidence: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not source_record_matches_allele(record, allele):
            continue
        # ``merge_dynamic_knowledge_base`` exposes generated dynamic records
        # to legacy renderers. They are not curated evidence and must not be
        # re-ingested here; the original source records below are matched
        # directly and allele-specifically instead.
        if _clean_text(record.get("region_class")) == "dynamic_query_variant":
            continue
        for source in _as_list(record.get("evidence")):
            if not isinstance(source, dict):
                continue
            evidence.append(
                {
                    "source_key": "local_curated",
                    "source": "Local curated bundle",
                    "category": "curated_research",
                    "source_id": _clean_text(source.get("url") or source.get("label")),
                    "label": _clean_text(source.get("label")),
                    "url": _clean_text(source.get("url")),
                    "clinical_significance": _clean_text(record.get("clinical_significance")),
                    "review_status": "",
                    "assertion_criteria": "",
                    "conditions": record.get("associated_conditions", []),
                    "last_evaluated": "",
                    "summary": _clean_text(record.get("clinical_interpretation"))[:600],
                    "match": "exact_allele",
                }
            )
    return evidence


def assess_evidence_strength(evidence: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Classify evidence without deriving a clinical class from text alone."""
    rows = list(evidence)
    clinical_rows = [
        row
        for row in rows
        if row.get("source_key") in _CLINICAL_SOURCE_KEYS and row.get("category") == "clinical_variant"
    ]
    reviewed_clinical = [
        row for row in clinical_rows if _clean_text(row.get("review_status"))
    ]
    if reviewed_clinical:
        return {
            "status": "source_assertion_available",
            "tier": "clinical_source_assertion",
            "clinical_assertion_available": True,
            "evidence_count": len(rows),
            "clinical_evidence_count": len(reviewed_clinical),
        }
    if rows:
        return {
            "status": "research_or_ungraded",
            "tier": "research_context",
            "clinical_assertion_available": False,
            "evidence_count": len(rows),
            "clinical_evidence_count": 0,
        }
    return {
        "status": "not_assessed",
        "tier": "not_assessed",
        "clinical_assertion_available": False,
        "evidence_count": 0,
        "clinical_evidence_count": 0,
    }


def _select_transcript_annotation(evidence: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select a MANE-first transcript annotation already returned by a source."""
    candidates = [
        row.get("transcript_consequence")
        for row in evidence
        if isinstance(row.get("transcript_consequence"), dict) and row.get("transcript_consequence")
    ]
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            not bool(row.get("is_mane_select")),
            not bool(row.get("is_canonical")),
            not _clean_text(row.get("transcript_id") or row.get("id")),
        )
    )
    transcript = candidates[0]
    return {
        "transcript_id": _clean_text(transcript.get("transcript_id") or transcript.get("id")),
        "mane_select": bool(transcript.get("is_mane_select")),
        "canonical": bool(transcript.get("is_canonical") or transcript.get("canonical")),
        "hgvsc": _clean_text(transcript.get("hgvsc")),
        "hgvsp": _clean_text(transcript.get("hgvsp")),
        "consequence": _clean_text(transcript.get("major_consequence"))
        or ", ".join(str(item) for item in _as_list(transcript.get("consequence_terms")) if str(item).strip()),
    }


def _external_identifiers(evidence: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Collect source-supplied canonical identifiers without fabricating them."""
    for row in evidence:
        spdi = _clean_text(row.get("spdi"))
        vrs_id = _clean_text(row.get("vrs_id"))
        if spdi or vrs_id:
            return {"spdi": spdi, "vrs_id": vrs_id}
    return {"spdi": "", "vrs_id": ""}


def _study_applicability(sample_context: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [
        field
        for field in ("tissue", "ancestry", "phenotype_terms")
        if not _clean_text(sample_context.get(field)) and not sample_context.get(field)
    ]
    study_fields = {
        "has_tissue_context": any(_clean_text(row.get("tissue")) for row in evidence),
        "has_cohort_context": any(_clean_text(row.get("cohort")) for row in evidence),
        "has_effect_estimate": any(
            _clean_text(row.get("effect_size") or row.get("beta") or row.get("p_value")) for row in evidence
        ),
    }
    return {
        "status": "context_limited" if missing else "sample_context_available",
        "missing_sample_context": missing,
        "study_metadata": study_fields,
    }


def _clinical_blockers(
    allele: dict[str, Any],
    call_quality: dict[str, Any],
    evidence_strength: dict[str, Any],
    sample_context: dict[str, Any],
    transcript: dict[str, Any],
    gene_disease_context_available: bool,
) -> list[str]:
    blockers = list(call_quality.get("blockers", []))
    normalization = allele.get("normalization", {})
    if allele.get("genome_build") == "unknown":
        blockers.append("unknown_genome_build")
    if normalization.get("identity_status") != "ready_for_external_matching":
        blockers.append("incomplete_variant_identity")
    if normalization.get("reference_allele_validation") != "validated":
        blockers.append("reference_allele_not_validated")
    if not _clean_text(transcript.get("transcript_id")):
        blockers.append("mane_transcript_not_annotated")
    if not evidence_strength.get("clinical_assertion_available"):
        blockers.append("no_reviewed_clinical_assertion")
    if not gene_disease_context_available:
        blockers.append("missing_relevant_gene_disease_context")
    if not sample_context.get("phenotype_terms"):
        blockers.append("missing_phenotype_context")
    return list(dict.fromkeys(blockers))


def _has_relevant_gene_disease_context(
    dynamic_payload: dict[str, Any] | None,
    *,
    gene_name: str,
) -> bool:
    """Recognize gene-level context without promoting it to a variant claim."""
    for record in (dynamic_payload or {}).get("source_records", []):
        if not isinstance(record, dict):
            continue
        source_key = _clean_text(record.get("source_key")).casefold()
        category = _clean_text(record.get("category")).casefold()
        record_gene = _clean_text(record.get("gene") or record.get("gene_symbol")).casefold()
        if record_gene != _clean_text(gene_name).casefold():
            continue
        if source_key in {"clingen", "panelapp", "clinvar"} and category in {
            "gene_disease_validity",
            "gene_disease",
            "diagnostic_panel",
        }:
            return True
    return False


def build_findings(
    *,
    variants: pd.DataFrame,
    gene_name: str,
    genome_build: str | None,
    knowledge_base: dict[str, Any],
    dynamic_payload: dict[str, Any] | None,
    sample_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build exact-allele findings with explicit clinical downgrade blockers."""
    context = dict(sample_context or {})
    gene_disease_context_available = _has_relevant_gene_disease_context(
        dynamic_payload,
        gene_name=gene_name,
    )
    findings: list[dict[str, Any]] = []
    for allele in canonical_variant_alleles(variants, genome_build=genome_build):
        if not allele["observed_non_reference"]:
            continue
        evidence = [
            *_curated_evidence_for_allele(knowledge_base, allele),
            *_dynamic_evidence_for_allele(dynamic_payload, allele),
        ]
        if any(item.get("reference_allele_validation") == "validated" for item in evidence):
            allele = {
                **allele,
                "normalization": {
                    **dict(allele.get("normalization", {})),
                    "reference_allele_validation": "validated",
                },
            }
        call_quality = assess_call_quality(allele)
        evidence_strength = assess_evidence_strength(evidence)
        transcript = _select_transcript_annotation(evidence)
        external_identifiers = _external_identifiers(evidence)
        applicability = _study_applicability(context, evidence)
        blockers = _clinical_blockers(
            allele,
            call_quality,
            evidence_strength,
            context,
            transcript,
            gene_disease_context_available,
        )
        findings.append(
            {
                "finding_key": allele["finding_key"],
                "gene": str(gene_name or "").upper(),
                "variant": {
                    **{key: value for key, value in allele.items() if key not in {"raw_call"}},
                    "transcript": transcript,
                    "external_identifiers": external_identifiers,
                },
                "call_quality": call_quality,
                "evidence_strength": evidence_strength,
                "study_applicability": applicability,
                "evidence": evidence,
                "clinical_support": {
                    "status": "eligible" if not blockers else "downgraded_to_research",
                    "eligible": not blockers,
                    "gene_disease_context_available": gene_disease_context_available,
                    "eligibility_blockers": blockers,
                },
            }
        )
    return findings


def build_methylation_assessment(
    methylation: pd.DataFrame,
    *,
    sample_context: dict[str, Any] | None = None,
    alleles: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return probe-level methylation context without phenotype predictions."""
    context = dict(sample_context or {})
    observed_alleles = [
        allele for allele in (alleles or []) if isinstance(allele, dict) and allele.get("observed_non_reference")
    ]
    required_context = (
        "tissue",
        "batch_id",
        "cell_composition_method",
        "methylation_reference_cohort_id",
    )
    missing_context = [field for field in required_context if not _clean_text(context.get(field))]
    available = set(methylation.columns)
    qc_columns = {"detection_p_value", "bead_count", "normalization_method"}
    missing_qc = sorted(qc_columns - available)
    probe_rows: list[dict[str, Any]] = []
    masked_probe_count = 0
    for _, row in methylation.iterrows():
        beta = _safe_float(row.get("beta"))
        detection_p = _safe_float(
            row.get("detection_p_value")
            if "detection_p_value" in row
            else row.get("detection_p")
            if "detection_p" in row
            else row.get("Detection Pval")
        )
        bead_count = _safe_int(
            row.get("bead_count")
            if "bead_count" in row
            else row.get("beadcount")
            if "beadcount" in row
            else row.get("Bead_Count")
        )
        mask_reasons: list[str] = []
        if detection_p is not None and detection_p > 0.01:
            mask_reasons.append("detection_p_above_0.01")
        if bead_count is not None and bead_count < 3:
            mask_reasons.append("bead_count_below_3")
        for field, reason in (
            ("cross_reactive", "cross_reactive_probe"),
            ("non_unique_mapping", "non_unique_mapping"),
            ("probe_snp_overlap", "probe_snp_overlap"),
        ):
            if _is_truthy(row.get(field, False)):
                mask_reasons.append(reason)
        if mask_reasons:
            masked_probe_count += 1
        variant_relationship = _probe_variant_relationship(row, observed_alleles, context)
        probe_rows.append(
            {
                "probe_id": _clean_text(row.get("probe_id")),
                "beta": beta,
                "chrom": _clean_text(row.get("chrom")),
                "pos": _safe_int(row.get("pos")),
                "gene_annotation": _clean_text(row.get("GencodeBasicV12_NAME") or row.get("UCSC_RefGene_Name")),
                "gene_region": _clean_text(row.get("UCSC_RefGene_Group")),
                "cpg_island_relation": _clean_text(row.get("Relation_to_UCSC_CpG_Island")),
                "interpretation": "descriptive_only",
                "detection_p_value": detection_p,
                "bead_count": bead_count,
                "mask_reasons": mask_reasons,
                "included_for_reference_comparison": not mask_reasons,
                "variant_relationship": variant_relationship,
            }
        )
    clinical_blockers = [*missing_context]
    if missing_qc:
        clinical_blockers.append("missing_idat_or_probe_qc:" + ",".join(missing_qc))
    return {
        "policy_version": METHYLATION_POLICY_VERSION,
        "status": "research_context_only" if clinical_blockers else "reference_comparison_ready",
        "clinical_support_eligible": not clinical_blockers,
        "eligibility_blockers": clinical_blockers,
        "missing_qc_columns": missing_qc,
        "masking_policy": {
            "detection_p_value_max": 0.01,
            "bead_count_min": 3,
            "masked_probe_count": masked_probe_count,
        },
        "probe_findings": probe_rows,
        "raw_mean_beta_is_interpreted": False,
        "reference_comparison": {
            "status": "not_available" if not context.get("methylation_reference_cohort_id") else "requires_external_reference_computation",
            "reference_cohort_id": _clean_text(context.get("methylation_reference_cohort_id")),
        },
    }


def _probe_variant_relationship(
    row: pd.Series,
    alleles: Iterable[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Gate mQTL claims on explicit probe-variant and study-context fields."""
    declared_rsid = _clean_text(
        row.get("probe_variant_rsid")
        or row.get("methylation_qtl_rsid")
        or row.get("linked_variant_rsid")
    )
    declared_chrom = _clean_text(row.get("probe_variant_chrom")).removeprefix("chr")
    declared_pos = _safe_int(row.get("probe_variant_pos"))
    declared_ref = _clean_allele(row.get("probe_variant_ref"))
    declared_alt = _clean_allele(row.get("probe_variant_alt"))
    exact_match = False
    matched_allele = ""
    for allele in alleles:
        if declared_rsid and declared_rsid.casefold() == _clean_text(allele.get("rsid")).casefold():
            exact_match = True
        elif (
            declared_chrom
            and declared_pos
            and declared_ref
            and declared_alt
            and declared_chrom == _clean_text(allele.get("chrom")).removeprefix("chr")
            and declared_pos == _safe_int(allele.get("pos"))
            and declared_ref == _clean_allele(allele.get("ref"))
            and declared_alt == _clean_allele(allele.get("alt"))
        ):
            exact_match = True
        if exact_match:
            matched_allele = _clean_text(allele.get("rsid")) or _clean_text(allele.get("allele_key"))
            break
    confounders = [
        label
        for field, label in (
            ("probe_binding_polymorphism", "probe_binding_polymorphism"),
            ("ld_confounded", "linkage_disequilibrium"),
        )
        if _is_truthy(row.get(field, False))
    ]
    same_tissue_cohort = _is_truthy(row.get("same_tissue_cohort", False))
    supporting_tissue = _clean_text(row.get("supporting_tissue"))
    supporting_cohort = _clean_text(row.get("supporting_cohort"))
    if not same_tissue_cohort and supporting_tissue and supporting_cohort:
        same_tissue_cohort = (
            supporting_tissue.casefold() == _clean_text(context.get("tissue")).casefold()
            and supporting_cohort.casefold()
            == _clean_text(context.get("methylation_reference_cohort_id")).casefold()
        )
    if not (declared_rsid or declared_chrom):
        status = "not_assessed"
    elif not exact_match:
        status = "declared_variant_not_observed"
    elif not same_tissue_cohort:
        status = "research_context_only"
    elif confounders:
        status = "exact_relationship_with_confounders"
    else:
        status = "exact_probe_variant_relationship"
    return {
        "status": status,
        "matched_allele": matched_allele,
        "supporting_tissue": supporting_tissue,
        "supporting_cohort": supporting_cohort,
        "same_tissue_cohort": same_tissue_cohort,
        "confounding_alternatives": confounders,
    }


def build_model_assessments(
    *,
    sample_context: dict[str, Any] | None = None,
    requested_models: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Gate probabilistic outputs behind explicit registered-model metadata."""
    context = dict(sample_context or {})
    assessments: list[dict[str, Any]] = []
    models = [model for model in (requested_models or []) if isinstance(model, dict)]
    if not models:
        return [
            {
                "model_id": "",
                "status": "model_not_eligible",
                "probability_emitted": False,
                "calibrated_probability": None,
                "eligibility_blockers": ["no_registered_model_requested"],
                "weighted_call_coverage": None,
                "performance_metric": "",
                "calibration": "",
            }
        ]
    for model in models:
        blockers: list[str] = []
        for field in (
            "model_id",
            "model_version",
            "source",
            "evaluation_ancestry",
            "performance_metric",
            "calibration",
            "baseline_risk",
        ):
            if not _clean_text(model.get(field)):
                blockers.append(f"missing_{field}")
        coverage = _safe_float(model.get("weighted_call_coverage"))
        if coverage is None or coverage < 0.95:
            blockers.append("insufficient_weighted_call_coverage")
        if not bool(model.get("alleles_harmonized", False)):
            blockers.append("alleles_not_harmonized")
        if not _clean_text(context.get("ancestry")):
            blockers.append("missing_sample_ancestry")
        elif _clean_text(context.get("ancestry")).casefold() != _clean_text(model.get("evaluation_ancestry")).casefold():
            blockers.append("ancestry_evaluation_mismatch")
        calibrated_probability = _safe_float(model.get("calibrated_probability"))
        assessments.append(
            {
                "model_id": _clean_text(model.get("model_id")),
                "status": "eligible" if not blockers else "model_not_eligible",
                "probability_emitted": not blockers and calibrated_probability is not None,
                "calibrated_probability": calibrated_probability if not blockers else None,
                "eligibility_blockers": blockers,
                "weighted_call_coverage": coverage,
                "performance_metric": _clean_text(model.get("performance_metric")),
                "calibration": _clean_text(model.get("calibration")),
                "baseline_risk": _clean_text(model.get("baseline_risk")),
            }
        )
    return assessments


def build_evidence_snapshot(
    dynamic_payload: dict[str, Any] | None,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create immutable report provenance from an optional dynamic KB payload."""
    payload = dynamic_payload or {}
    embedded_snapshot = payload.get("evidence_snapshot") if isinstance(payload, dict) else None
    if (
        isinstance(embedded_snapshot, dict)
        and _clean_text(embedded_snapshot.get("snapshot_id"))
        and _clean_text(embedded_snapshot.get("checksum_sha256"))
    ):
        # Dynamic KB generation hashes its provider releases and normalized
        # source records. Reuse that immutable identity in every derived
        # report instead of producing a weaker report-time checksum.
        return dict(embedded_snapshot)
    timestamp = generated_at or _clean_text(payload.get("generated_at")) or utc_now_iso()
    normalized_records = sorted(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
        for record in payload.get("source_records", [])
        if isinstance(record, dict)
    )
    normalized_evidence_checksum = hashlib.sha256(
        "\n".join(normalized_records).encode("utf-8")
    ).hexdigest()
    embedded_provider_lookup = {
        _clean_text(provider.get("source_key")): provider
        for provider in (embedded_snapshot.get("providers", []) if isinstance(embedded_snapshot, dict) else [])
        if isinstance(provider, dict)
    }
    providers: list[dict[str, Any]] = []
    for status in payload.get("provider_statuses", []) if isinstance(payload, dict) else []:
        if not isinstance(status, dict):
            continue
        source_status = _clean_text(status.get("status")) or "not_assessed"
        source_key = _clean_text(status.get("source_key"))
        source_snapshot = embedded_provider_lookup.get(source_key, {})
        providers.append(
            {
                "source_key": source_key,
                "source": _clean_text(status.get("name")),
                "status": source_status,
                "assessment_status": "assessed" if source_status in {"ok", "imported"} else "not_assessed",
                "retrieved_at": timestamp,
                "queried_urls": [str(url) for url in _as_list(status.get("queried_urls")) if str(url).strip()],
                "record_count": _safe_int(status.get("record_count")) or 0,
                "source_release": _clean_text(source_snapshot.get("source_release")),
            }
        )
    identity = {
        "schema_version": INTERPRETATION_SCHEMA_VERSION,
        "gene": _clean_text(payload.get("gene_name")),
        "region": _clean_text(payload.get("region")),
        "genome_build": _clean_text(payload.get("genome_build")),
        "generated_at": timestamp,
        "providers": providers,
        "normalized_evidence_checksum_sha256": normalized_evidence_checksum,
        "normalized_evidence_record_count": len(normalized_records),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {
        **identity,
        "snapshot_id": f"evidence-{checksum[:16]}",
        "checksum_sha256": checksum,
        "refresh_policy": "explicit_user_triggered",
    }


def build_interpretation_payload(
    *,
    variants: pd.DataFrame,
    methylation: pd.DataFrame,
    gene_name: str,
    genome_build: str | None,
    interpretation_mode: str,
    knowledge_base: dict[str, Any],
    dynamic_payload: dict[str, Any] | None,
    sample_context: dict[str, Any] | None = None,
    requested_models: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the additive schema-v2 interpretation payload."""
    mode = normalize_interpretation_mode(interpretation_mode)
    context = dict(sample_context or {})
    snapshot = build_evidence_snapshot(dynamic_payload)
    findings = build_findings(
        variants=variants,
        gene_name=gene_name,
        genome_build=genome_build,
        knowledge_base=knowledge_base,
        dynamic_payload=dynamic_payload,
        sample_context=context,
    )
    methylation_assessment = build_methylation_assessment(
        methylation,
        sample_context=context,
        alleles=canonical_variant_alleles(variants, genome_build=genome_build),
    )
    model_assessments = build_model_assessments(
        sample_context=context,
        requested_models=requested_models,
    )
    clinical_eligible_count = sum(1 for finding in findings if finding["clinical_support"]["eligible"])
    clinical_eligible_keys = [
        finding["finding_key"] for finding in findings if finding["clinical_support"]["eligible"]
    ]
    research_only_keys = [
        finding["finding_key"] for finding in findings if not finding["clinical_support"]["eligible"]
    ]
    return {
        "schema_version": INTERPRETATION_SCHEMA_VERSION,
        "interpretation_context": {
            "mode": mode,
            "gene": str(gene_name or "").upper(),
            "genome_build": normalize_genome_build(genome_build),
            "sample_context": context,
            "clinical_support_disclaimer": "Decision support only; final clinical review is external to this application.",
        },
        "evidence_snapshot": snapshot,
        "findings": findings,
        "methylation_assessment": methylation_assessment,
        "model_assessments": model_assessments,
        "mode_summaries": {
            "research": {
                "finding_count": len(findings),
                "message": "Research findings preserve exact genotype and evidence provenance without individual trait forecasts.",
            },
            "clinical_support": {
                "eligible_finding_count": clinical_eligible_count,
                "downgraded_finding_count": len(findings) - clinical_eligible_count,
                "message": "Findings without all clinical prerequisites are shown as research-only.",
            },
        },
        "mode_reports": {
            "research": {
                "finding_keys": [finding["finding_key"] for finding in findings],
                "message": "All findings are research-context records with exact allele identity and source provenance.",
            },
            "clinical_support": {
                "eligible_finding_keys": clinical_eligible_keys,
                "research_only_downgrade_keys": research_only_keys,
                "message": "Only eligible findings belong in clinical-support output; all others are explicitly research-only.",
            },
        },
        "drd4_repeat_assay": {
            "status": "not_assessed",
            "markers": ["DRD4 exon III 48-bp VNTR", "DRD4 promoter duplication"],
            "haplotype_status": "not_inferred_from_nearby_snps",
            "reason": "Small-variant VCF evidence cannot establish repeat number, duplication state, or repeat haplotype phase.",
        }
        if str(gene_name or "").upper() == "DRD4"
        else None,
    }
