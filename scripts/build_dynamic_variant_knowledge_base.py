#!/usr/bin/env python3
"""Build a dynamic variant knowledge base for one gene/query interval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import AnalysisError, load_variants
from src.variant_knowledge.orchestrator import build_dynamic_knowledge_base


def _load_manifest_subset(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest subset not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def _load_variants(path: str | None, region: str) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        return load_variants(path, region)
    except AnalysisError as exc:
        raise RuntimeError(f"Could not load VCF variants for dynamic KB: {exc}") from exc


def _parse_selected_sources(raw: str) -> list[str] | None:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("["):
        payload = json.loads(cleaned)
        if not isinstance(payload, list):
            raise ValueError("--selected-sources JSON must be a list.")
        return [str(item) for item in payload]
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _parse_source_imports(raw_items: list[str] | None) -> dict[str, str]:
    imports: dict[str, str] = {}
    for item in raw_items or []:
        if "=" not in item:
            raise ValueError("--source-import must use SOURCE_KEY=PATH.")
        source_key, path = item.split("=", 1)
        source_key = source_key.strip()
        path = path.strip()
        if not source_key or not path:
            raise ValueError("--source-import requires both SOURCE_KEY and PATH.")
        imports[source_key] = path
    return imports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, help="Gene symbol used for the dynamic KB.")
    parser.add_argument("--region", required=True, help="Region in chrom:start-end format.")
    parser.add_argument("--genome-build", default="hg19", help="Genome build label, usually hg19 or hg38.")
    parser.add_argument("--vcf", default="", help="Optional VCF path for observed variants.")
    parser.add_argument("--manifest-subset", default="", help="Optional filtered manifest CSV.")
    parser.add_argument("--selected-sources", default="", help="Comma-separated source keys or JSON list. Defaults to all.")
    parser.add_argument(
        "--source-import",
        action="append",
        default=[],
        metavar="SOURCE_KEY=PATH",
        help="User-provided licensed-source CSV/JSON export. Repeat for multiple sources.",
    )
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".research-cache" / "variant_knowledge"))
    parser.add_argument("--output-dir", required=True, help="Directory that receives variant_kb.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = _load_variants(args.vcf, args.region)
    manifest_subset = _load_manifest_subset(args.manifest_subset)
    payload = build_dynamic_knowledge_base(
        gene=args.gene,
        region=args.region,
        genome_build=args.genome_build,
        variants=variants,
        manifest_subset=manifest_subset,
        selected_sources=_parse_selected_sources(args.selected_sources),
        source_imports=_parse_source_imports(args.source_import),
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    print(json.dumps({"artifact_path": payload.get("artifact_path"), "provider_count": len(payload.get("provider_statuses", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
