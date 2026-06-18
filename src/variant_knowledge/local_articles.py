"""Local PDF article evidence extraction for dynamic knowledge bases."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SourceResult, SourceSpec

LOCAL_ARTICLE_SOURCE_KEY = "local_pdf_articles"
LOCAL_ARTICLE_WORKFLOW_KEY = "local_pdf_article_evidence"
LOCAL_ARTICLE_LICENSE_NOTE = (
    "User-provided local PDFs are processed for local analysis only. The dynamic KB stores short "
    "gene-relevant snippets and provenance, not full article text."
)
LOCAL_ARTICLE_SOURCE_SPEC = SourceSpec(
    key=LOCAL_ARTICLE_SOURCE_KEY,
    name="Local PDF Articles",
    description="User-provided scientific article PDFs parsed locally for gene-relevant evidence.",
    lane="literature",
    access_type="local_user_files",
    connector_kind="local_pdf_extractor",
    homepage="",
    license_note=LOCAL_ARTICLE_LICENSE_NOTE,
    supports_variant=True,
    supports_gene=True,
    supports_region=False,
    supports_literature=True,
    ingestion_modes=("user_export", "linkout_only"),
    requires_export=False,
)

SUMMARY_FIELDS = [
    "record_id",
    "gene",
    "title",
    "doi",
    "pmid",
    "section",
    "page",
    "claim_type",
    "variant",
    "rsid",
    "phenotype",
    "drug",
    "confidence",
    "evidence_level",
    "source_file",
    "snippet",
]


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    parser: str


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _discover_pdf_files(folder: Path, *, recursive: bool, max_pdfs: int) -> list[Path]:
    iterator = folder.rglob("*.pdf") if recursive else folder.glob("*.pdf")
    files = sorted(path for path in iterator if path.is_file())
    return files[: max(0, int(max_pdfs or 0))]


def _decode_pdf_literal(value: str) -> str:
    value = re.sub(r"^\(", "", value.strip())
    value = re.sub(r"\)\s*T[Jj]$", "", value)
    value = re.sub(r"\)$", "", value)
    value = value.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
    value = re.sub(r"\\([nrtbf])", " ", value)
    value = re.sub(r"\\\d{1,3}", " ", value)
    return _clean_text(value)


def _text_from_pdf_bytes(raw: bytes) -> str:
    """Extract visible text from simple PDF text operators without third-party dependencies."""
    chunks: list[str] = []
    decoded = raw.decode("latin-1", errors="ignore")
    streams = re.findall(r"stream\r?\n(.*?)\r?\nendstream", decoded, flags=re.S)
    for stream in streams:
        stream_bytes = stream.encode("latin-1", errors="ignore")
        candidates = [stream]
        try:
            candidates.append(zlib.decompress(stream_bytes).decode("latin-1", errors="ignore"))
        except Exception:
            pass
        for candidate in candidates:
            chunks.extend(_decode_pdf_literal(match) for match in re.findall(r"\((?:\\.|[^\\()])*\)\s*T[Jj]", candidate))
            array_texts = re.findall(r"\[((?:\s*\((?:\\.|[^\\()])*\)\s*)+)\]\s*TJ", candidate)
            for array_text in array_texts:
                chunks.extend(_decode_pdf_literal(match) for match in re.findall(r"\((?:\\.|[^\\()])*\)", array_text))
    if not chunks:
        chunks.extend(_decode_pdf_literal(match) for match in re.findall(r"\((?:\\.|[^\\()]){4,}\)", decoded))
    return _clean_text(" ".join(chunk for chunk in chunks if chunk))


def _extract_with_fitz(path: Path) -> tuple[list[ExtractedPage], dict[str, str]] | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    pages: list[ExtractedPage] = []
    metadata: dict[str, str] = {}
    with fitz.open(path) as document:  # type: ignore[attr-defined]
        metadata = {
            "title": _clean_text((document.metadata or {}).get("title")),
            "author": _clean_text((document.metadata or {}).get("author")),
        }
        for index, page in enumerate(document, start=1):
            text = _clean_text(page.get_text("text"))
            if text:
                pages.append(ExtractedPage(page_number=index, text=text, parser="pymupdf"))
    return pages, metadata


def _extract_with_pypdf(path: Path) -> tuple[list[ExtractedPage], dict[str, str]] | None:
    reader_cls = None
    try:
        from pypdf import PdfReader  # type: ignore

        reader_cls = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader_cls = PdfReader
        except Exception:
            return None
    reader = reader_cls(str(path))
    metadata_obj = getattr(reader, "metadata", None) or {}
    metadata_get = metadata_obj.get if hasattr(metadata_obj, "get") else lambda _key, default="": default
    metadata = {
        "title": _clean_text(getattr(metadata_obj, "title", "") or metadata_get("/Title", "")),
        "author": _clean_text(getattr(metadata_obj, "author", "") or metadata_get("/Author", "")),
    }
    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append(ExtractedPage(page_number=index, text=text, parser="pypdf"))
    return pages, metadata


def _extract_pages(path: Path) -> tuple[list[ExtractedPage], dict[str, str], list[str]]:
    warnings: list[str] = []
    for extractor in (_extract_with_fitz, _extract_with_pypdf):
        try:
            extracted = extractor(path)
        except Exception as exc:
            warnings.append(f"{extractor.__name__} failed: {exc}")
            extracted = None
        if extracted is not None:
            pages, metadata = extracted
            if pages:
                return pages, metadata, warnings
    try:
        text = _text_from_pdf_bytes(path.read_bytes())
    except Exception as exc:
        return [], {}, warnings + [f"stdlib PDF text fallback failed: {exc}"]
    if not text:
        warnings.append("No text layer was extracted. OCR is not configured for this run.")
        return [], {}, warnings
    return [ExtractedPage(page_number=1, text=text, parser="stdlib_pdf_text")], {}, warnings


def _gene_terms(gene: str, aliases: list[str] | tuple[str, ...] | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in [gene, *(aliases or [])]:
        term = _clean_text(raw_term)
        if not term:
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def _term_pattern(terms: list[str]) -> re.Pattern[str]:
    alternatives = [re.escape(term) for term in sorted(terms, key=len, reverse=True)]
    return re.compile(r"(?<![A-Za-z0-9_-])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9_-])", re.I)


def _sentence_windows(text: str) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?;]{20,900}[.!?;]?", text):
        sentence = _clean_text(match.group(0))
        if sentence:
            windows.append((match.start(), match.end(), sentence))
    return windows or [(0, min(len(text), 900), _clean_text(text[:900]))]


def _section_for_position(text: str, position: int) -> str:
    before = text[max(0, position - 1500) : position].lower()
    sections = [
        ("abstract", "Abstract"),
        ("results", "Results"),
        ("discussion", "Discussion"),
        ("conclusion", "Conclusion"),
        ("methods", "Methods"),
        ("materials", "Methods"),
        ("figure", "Figure or Table Caption"),
        ("table", "Figure or Table Caption"),
    ]
    best_label = "Body"
    best_pos = -1
    for needle, label in sections:
        found = before.rfind(needle)
        if found > best_pos:
            best_pos = found
            best_label = label
    return best_label


def _claim_type(snippet: str) -> str:
    lower = snippet.lower()
    if re.search(r"\brs\d+\b|variant|mutation|pathogenic|benign|hgvs", lower):
        return "clinical_variant"
    if any(token in lower for token in ("methylation", "cpg", "chromatin", "enhancer", "promoter", "epigen")):
        return "regulatory_epigenomic"
    if any(token in lower for token in ("drug", "pharmaco", "dose", "therapy", "inhibitor", "response")):
        return "pharmacogenomics"
    if any(token in lower for token in ("gwas", "association", "odds ratio", "risk allele", "cohort")):
        return "population_association"
    if any(token in lower for token in ("expression", "knockdown", "protein", "assay", "pathway", "function")):
        return "functional_biology"
    return "gene_literature_context"


def _confidence(section: str, snippet: str) -> tuple[float, str]:
    score = 0.55
    if section in {"Abstract", "Results", "Figure or Table Caption"}:
        score += 0.15
    if re.search(r"\brs\d+\b|p\.|c\.", snippet):
        score += 0.10
    if any(token in snippet.lower() for token in ("significant", "reported", "observed", "associated", "showed")):
        score += 0.08
    score = min(score, 0.95)
    level = "high" if score >= 0.78 else "medium" if score >= 0.62 else "low"
    return round(score, 2), level


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.I)
    return _clean_text(match.group(0)) if match else ""


def _doi(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.I)
    return _clean_text(match.group(0).rstrip(".,);")) if match else ""


def _pmid(text: str) -> str:
    match = re.search(r"\bPMID[:\s]*(\d{4,})\b", text, flags=re.I)
    return match.group(1) if match else ""


def _title(metadata: dict[str, str], text: str, fallback: str) -> str:
    if metadata.get("title"):
        return metadata["title"][:240]
    for line in re.split(r"(?<=[.!?])\s+|\n", text[:1600]):
        cleaned = _clean_text(line)
        if 12 <= len(cleaned) <= 220 and not cleaned.lower().startswith(("abstract", "introduction", "results")):
            return cleaned
    return fallback


def _records_for_pdf(
    *,
    path: Path,
    gene: str,
    aliases: list[str] | tuple[str, ...] | None,
    generated_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    file_sha256 = _sha256_bytes(raw)
    pages, metadata, warnings = _extract_pages(path)
    all_text = _clean_text(" ".join(page.text for page in pages))
    terms = _gene_terms(gene, aliases)
    pattern = _term_pattern(terms)
    title = _title(metadata, all_text, path.stem)
    doi = _doi(all_text)
    pmid = _pmid(all_text)
    records: list[dict[str, Any]] = []
    seen_snippets: set[str] = set()
    for page in pages:
        for start, _end, sentence in _sentence_windows(page.text):
            if not pattern.search(sentence):
                continue
            snippet = sentence[:700]
            snippet_key = snippet.casefold()
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)
            section = _section_for_position(page.text, start)
            confidence, evidence_level = _confidence(section, snippet)
            rsid = _first_match(r"\brs\d+\b", snippet)
            variant = rsid or _first_match(r"\b[cp]\.[A-Za-z0-9_>+\-*]+", snippet)
            record_id = f"{LOCAL_ARTICLE_SOURCE_KEY}:{file_sha256[:12]}:p{page.page_number}:{len(records) + 1}"
            records.append(
                {
                    "source_key": LOCAL_ARTICLE_SOURCE_KEY,
                    "record_id": record_id,
                    "category": "literature",
                    "gene": gene,
                    "title": title,
                    "label": title,
                    "summary": snippet,
                    "snippet": snippet,
                    "section": section,
                    "page": page.page_number,
                    "doi": doi,
                    "pmid": pmid,
                    "citation": doi or (f"PMID:{pmid}" if pmid else title),
                    "source_file": path.name,
                    "source_file_hash": _sha256_text(path.name),
                    "source_file_sha256": file_sha256,
                    "pdf_path_hash": _sha256_text(str(path.resolve())),
                    "claim_type": _claim_type(snippet),
                    "variant": variant,
                    "rsid": rsid,
                    "phenotype": "",
                    "drug": "",
                    "confidence": confidence,
                    "evidence_level": evidence_level,
                    "license_note": LOCAL_ARTICLE_LICENSE_NOTE,
                    "imported_at": generated_at,
                    "url": "",
                }
            )
    file_provenance = {
        "source_file": path.name,
        "source_file_hash": _sha256_text(path.name),
        "source_file_sha256": file_sha256,
        "pdf_path_hash": _sha256_text(str(path.resolve())),
        "page_count": len(pages),
        "parser": pages[0].parser if pages else "",
        "text_extracted": bool(all_text),
        "matched_record_count": len(records),
        "warnings": warnings,
    }
    return records, file_provenance


def extract_local_article_evidence(
    *,
    gene: str,
    pdf_folder: str | Path | None,
    gene_aliases: list[str] | tuple[str, ...] | None = None,
    recursive: bool = True,
    max_pdfs: int = 100,
    generated_at: str,
) -> dict[str, Any]:
    """Extract short gene-relevant snippets from local scientific article PDFs."""
    normalized_gene = _clean_text(gene).upper()
    folder_text = _clean_text(pdf_folder)
    if not folder_text:
        return {
            "status": "needs_folder",
            "message": "Enable local article evidence by entering a folder containing legally obtained PDF articles.",
            "records": [],
            "provenance": {
                "source_key": LOCAL_ARTICLE_SOURCE_KEY,
                "generated_at": generated_at,
                "folder_path_hash": "",
                "pdf_count": 0,
                "parsed_pdf_count": 0,
                "matched_pdf_count": 0,
                "record_count": 0,
                "files": [],
                "warnings": ["No PDF folder was provided."],
                "errors": [],
            },
        }
    folder = Path(folder_text).expanduser()
    if not folder.exists() or not folder.is_dir():
        return {
            "status": "needs_folder",
            "message": f"Local article evidence folder was not found or is not a directory: {folder}",
            "records": [],
            "provenance": {
                "source_key": LOCAL_ARTICLE_SOURCE_KEY,
                "generated_at": generated_at,
                "folder_path_hash": _sha256_text(str(folder)),
                "pdf_count": 0,
                "parsed_pdf_count": 0,
                "matched_pdf_count": 0,
                "record_count": 0,
                "files": [],
                "warnings": [],
                "errors": ["Folder not found or not a directory."],
            },
        }
    files = _discover_pdf_files(folder, recursive=recursive, max_pdfs=max_pdfs)
    all_records: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for pdf_path in files:
        try:
            records, file_provenance = _records_for_pdf(
                path=pdf_path,
                gene=normalized_gene,
                aliases=gene_aliases,
                generated_at=generated_at,
            )
        except Exception as exc:
            errors.append(f"{pdf_path.name}: {exc}")
            continue
        all_records.extend(records)
        file_rows.append(file_provenance)
    matched_pdf_count = sum(1 for row in file_rows if int(row.get("matched_record_count", 0) or 0) > 0)
    parsed_pdf_count = sum(1 for row in file_rows if row.get("text_extracted"))
    status = "ok" if all_records else "skipped" if files else "needs_folder"
    message = (
        f"Extracted {len(all_records)} gene-relevant snippet(s) from {matched_pdf_count} local PDF article(s)."
        if all_records
        else "No queried-gene evidence snippets were extracted from the selected local PDF folder."
        if files
        else "No PDF files were found in the selected local article folder."
    )
    return {
        "status": status,
        "message": message,
        "records": all_records,
        "provenance": {
            "source_key": LOCAL_ARTICLE_SOURCE_KEY,
            "generated_at": generated_at,
            "folder_path_hash": _sha256_text(str(folder.resolve())),
            "recursive": bool(recursive),
            "max_pdfs": int(max_pdfs or 0),
            "pdf_count": len(files),
            "parsed_pdf_count": parsed_pdf_count,
            "matched_pdf_count": matched_pdf_count,
            "record_count": len(all_records),
            "files": file_rows,
            "warnings": [
                warning
                for row in file_rows
                for warning in row.get("warnings", [])
                if warning
            ],
            "errors": errors,
        },
    }


def source_result_from_local_article_evidence(extraction: dict[str, Any]) -> SourceResult:
    """Convert local article extraction output to a dynamic KB source result."""
    status = str(extraction.get("status") or "skipped")
    return SourceResult(
        source_key=LOCAL_ARTICLE_SOURCE_KEY,
        status=status,
        message=str(extraction.get("message") or ""),
        records=list(extraction.get("records") or []),
        warnings=list((extraction.get("provenance") or {}).get("warnings") or []),
        errors=list((extraction.get("provenance") or {}).get("errors") or []),
    )


def write_local_article_artifacts(output_dir: str | Path, extraction: dict[str, Any]) -> dict[str, str]:
    """Write local article evidence artifacts next to the dynamic KB."""
    target_dir = Path(output_dir) / "local_article_evidence"
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "article_evidence.json"
    csv_path = target_dir / "article_evidence_summary.csv"
    json_path.write_text(json.dumps(extraction, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    records = list(extraction.get("records") or [])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in SUMMARY_FIELDS})
    return {
        "article_evidence_json": str(json_path),
        "article_evidence_summary_csv": str(csv_path),
    }
