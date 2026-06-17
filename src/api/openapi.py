"""Small hand-authored OpenAPI document for the local API."""

from __future__ import annotations

from typing import Any


def build_openapi_document() -> dict[str, Any]:
    """Return the API's OpenAPI 3.1 description."""
    profile_schema = {
        "type": "object",
        "required": [
            "display_name",
            "default_genome_build",
            "idat_prefix",
            "manifest_path",
        ],
        "properties": {
            "id": {"type": "string", "readOnly": True},
            "display_name": {"type": "string"},
            "default_genome_build": {"enum": ["hg19", "hg38"]},
            "idat_prefix": {"type": "string"},
            "manifest_path": {"type": "string"},
            "population_statistics_path": {"type": "string"},
            "vcf_sources": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GenomicSource"},
            },
            "bam_sources": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GenomicSource"},
            },
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "NophiGene Local Gene Workflow API",
            "version": "1.0.0",
        },
        "servers": [{"url": "/api/v1"}],
        "paths": {
            "/": {
                "get": {"summary": "Discover API endpoints", "responses": {"200": {"description": "OK"}}}
            },
            "/profiles": {
                "get": {"summary": "List sample profiles", "responses": {"200": {"description": "OK"}}},
                "post": {
                    "summary": "Create a sample profile",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Profile"}}
                        },
                    },
                    "responses": {"201": {"description": "Created"}, "422": {"description": "Invalid profile"}},
                },
            },
            "/profiles/{profile_id}": {
                "get": {"summary": "Get a sample profile", "responses": {"200": {"description": "OK"}}},
                "put": {"summary": "Replace a sample profile", "responses": {"200": {"description": "OK"}}},
                "delete": {"summary": "Delete a sample profile", "responses": {"204": {"description": "Deleted"}}},
            },
            "/jobs": {
                "get": {"summary": "List jobs", "responses": {"200": {"description": "OK"}}},
                "post": {
                    "summary": "Submit an asynchronous workflow job",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/JobRequest"}}
                        },
                    },
                    "responses": {"202": {"description": "Queued"}, "422": {"description": "Invalid job"}},
                },
            },
            "/jobs/{job_id}": {
                "get": {"summary": "Get job status", "responses": {"200": {"description": "OK"}}}
            },
            "/jobs/{job_id}/cancel": {
                "post": {"summary": "Cancel a queued job", "responses": {"200": {"description": "Cancelled"}}}
            },
            "/jobs/{job_id}/result": {
                "get": {"summary": "Get a completed result manifest", "responses": {"200": {"description": "OK"}}}
            },
            "/jobs/{job_id}/artifacts/{path}": {
                "get": {"summary": "Download a job artifact", "responses": {"200": {"description": "Artifact"}}}
            },
            "/knowledge-sources": {
                "get": {
                    "summary": "List dynamic knowledge source cards, ingestion modes, and import schema",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/knowledge-sources/test": {
                "post": {
                    "summary": "Check source API/export/linkout readiness without storing credentials",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/health": {
                "get": {"summary": "Get worker and extraction health", "responses": {"200": {"description": "OK"}}}
            },
            "/openapi.json": {
                "get": {"summary": "Get this OpenAPI document", "responses": {"200": {"description": "OK"}}}
            },
        },
        "components": {
            "schemas": {
                "GenomicSource": {
                    "type": "object",
                    "required": ["path", "genome_build"],
                    "properties": {
                        "path": {"type": "string"},
                        "genome_build": {"enum": ["hg19", "hg38"]},
                        "reference_fasta": {"type": "string"},
                    },
                },
                "Profile": profile_schema,
                "JobRequest": {
                    "type": "object",
                    "required": ["operation", "genes"],
                    "properties": {
                        "operation": {
                            "enum": [
                                "resolve_regions",
                                "prepare_manifests",
                                "extract_variants",
                                "build_knowledge_bases",
                                "analyze",
                                "render_reports",
                                "full_workflow",
                            ]
                        },
                        "genes": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "maxItems": 100, "items": {"type": "string"}},
                            ]
                        },
                        "profile_id": {"type": "string"},
                        "source_job_id": {"type": "string"},
                        "analysis_scope": {
                            "enum": ["promoter_plus_gene", "promoter_only", "gene_only"]
                        },
                        "genome_build": {"enum": ["auto", "hg19", "hg38"]},
                        "region_overrides": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "update_general_database": {"type": "boolean", "default": False},
                                "overwrite_general_database": {"type": "boolean", "default": False},
                                "use_dynamic_knowledge_base": {"type": "boolean", "default": False},
                                "knowledge_sources": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                },
                                "knowledge_source_imports": {
                                    "type": "object",
                                    "description": "Map source keys to permitted local CSV/JSON export paths.",
                                    "additionalProperties": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "object",
                            "required": ["code", "message"],
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "details": {},
                            },
                        }
                    },
                },
            }
        },
    }
